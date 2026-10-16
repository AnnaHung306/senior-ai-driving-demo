"""Train-only/validation-only observable lamp classifier. No lockbox access."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from senior_ai.core import write_json
from senior_ai.dataset import safe_path
from senior_ai.sunny_data import png,sha,verify
from senior_ai.light_model import LightStateNet,crop_rgb
from senior_ai.light_labels import LIGHT_STATES
from senior_ai.segmentation_training import configure_cuda,seed_all
from train_sunny import require_idle_gpu,save_checkpoint


class LightCrops(Dataset):
    def __init__(self,root,split,config,allow_lockbox=False,supplement_root=None):
        if split=='lockbox' and not allow_lockbox: raise ValueError('Lockbox forbidden during training')
        if supplement_root is not None and split!='lockbox':raise ValueError('Supplement is heldout only')
        self.samples=[];self.crops=[];counts=Counter();groups=defaultdict(set)
        for data_root in [root]+([supplement_root] if supplement_root is not None else []):
            for line in (data_root/f'training/{split}.jsonl').read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                if row['split']!=split: raise ValueError('Split mismatch')
                labels=json.loads(safe_path(data_root,row['detection']).read_text(encoding='utf-8'))
                lights=[p for p in labels['positive_objects'] if p['category_id']==7]
                if not lights: continue
                rgb=png(safe_path(data_root,row['rgb']))
                for light in lights:
                    state=light['light_state_class'];box=light['visible_bbox_xyxy']
                    self.samples.append({**row,'data_root':str(data_root.resolve()),'controlled_supplement':data_root!=root,
                                         'box':box,'target':state,'world_phase':light['world_phase']})
                    self.crops.append(crop_rgb(rgb,box));counts[state]+=1;groups[state].add(row['group_id'])
        self.counts=[counts[c] for c in range(4)]
        self.coverage={LIGHT_STATES[c]:{'samples':counts[c],'groups':len(groups[c])} for c in range(4)}
        self.approved=all(counts[c]>=config['minimum_samples_per_class'][split] and len(groups[c])>=config['minimum_groups_per_class'] for c in range(4))

    def __len__(self): return len(self.samples)
    def __getitem__(self,index): return self.crops[index],self.samples[index]['target'],index


def scores(matrix):
    support=matrix.sum(1);predicted=matrix.sum(0);diag=matrix.diagonal()
    rows=[{'name':name,'support':int(support[c]),'precision':float(diag[c]/predicted[c]) if predicted[c] else None,
           'recall':float(diag[c]/support[c]) if support[c] else None} for c,name in enumerate(LIGHT_STATES)]
    return {'confusion_true_rows_predicted_columns':matrix.tolist(),'classes':rows,'all_classes_present':bool((support>0).all()),
            'accuracy':float(diag.sum()/support.sum()) if support.sum() else None,
            'macro_recall':float(np.mean([r['recall'] for r in rows if r['recall'] is not None])) if support.sum() else None}


def quality(result,config):
    checks={'all_classes':result['all_classes_present'],'macro_recall':(result['macro_recall'] or 0)>=config['minimum_macro_recall']}
    for c,row in enumerate(result['classes']): checks[row['name']]=(row['recall'] or 0)>=(config['minimum_unknown_recall'] if c==0 else config['minimum_per_class_recall'])
    m=np.array(result['confusion_true_rows_predicted_columns']);checks['red_as_green']=(m[1,3]/max(1,m[1].sum()))<=config['max_red_as_green_rate']
    return {'passed':all(checks.values()),'checks':checks}


@torch.inference_mode()
def evaluate(model,data,batch_size):
    model.eval();matrix=np.zeros((4,4),dtype=np.int64);frames=[]
    for rgb,target,indices in DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=0):
        logits=model(rgb.cuda().float()/255)
        if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite phase logits')
        confidence,pred=logits.softmax(1).max(1);p=pred.cpu().numpy();t=target.numpy();matrix+=np.bincount(t*4+p,minlength=16).reshape(4,4)
        frames.extend({**data.samples[index],'prediction':int(prediction),'softmax_max_uncalibrated':float(conf)} for index,prediction,conf in zip(indices.tolist(),p,confidence.cpu().tolist()))
    return scores(matrix),frames


def train(args):
    if args.run.exists(): raise FileExistsError('Use a new run directory')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if config['crop_wh']!=[32,64] or config['class_names']!=LIGHT_STATES:raise ValueError('Phase crop/schema differs from implementation')
    if config.get('heldout_supplement_plan') and sha(ROOT/config['heldout_supplement_plan'])!=config['heldout_supplement_plan_sha256']:
        raise ValueError('Frozen heldout phase coverage plan changed')
    args.run.mkdir(parents=True)
    data_signature=verify(args.root);train=LightCrops(args.root,'train',config);validation=LightCrops(args.root,'validation',config)
    write_json(args.run/'coverage.json',{'train':train.coverage,'validation':validation.coverage})
    if not train.approved or not validation.approved: raise ValueError('Phase class/group coverage failed; cannot train missing phases')
    require_idle_gpu();configure_cuda();seed_all(config['seed']);model=LightStateNet().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.0001)
    weights=torch.tensor(np.clip(sum(train.counts)/(4*np.array(train.counts)),.1,10),device='cuda',dtype=torch.float32)
    source={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/train_sunny.py',ROOT/'scripts/predict_sunny.py',ROOT/'scripts/evaluate_light_state.py',ROOT/'scripts/validate_light_joint.py',ROOT/'scripts/audit_light_supplement.py',ROOT/'scripts/audit_sunny_dataset.py',ROOT/'scripts/validate_dataset.py',ROOT/'scripts/collect_dataset.py',ROOT/'scripts/rig_capture.py']}
    for relative in source:
        destination=args.run/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/relative).read_bytes())
    write_json(args.run/'config.json',config);best=-1.;started=time.monotonic();updates=0
    for epoch in range(config['epochs']):
        model.train();losses=[]
        for rgb,target,indices in DataLoader(train,batch_size=config['batch_size'],shuffle=True,num_workers=0):
            images=rgb.cuda().float()/255
            contrast=torch.rand((len(images),1,1,1),device='cuda')*.2+.9;brightness=torch.rand_like(contrast)*.2+.9
            mean=images.mean((2,3),keepdim=True);images=((images-mean)*contrast+mean).mul(brightness).clamp_(0,1)
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(images),target.cuda(),weight=weights)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite phase loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss.detach()))
        result,frames=evaluate(model,validation,config['batch_size']);score=result['macro_recall']
        value={'task':'light_state','model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'config':config,
               'data':data_signature,'source':source,'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
               'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()}}
        save_checkpoint(args.run/'last.pt',value)
        if score>best:
            best=score;save_checkpoint(args.run/'best.pt',value);write_json(args.run/'validation.json',result)
            (args.run/'validation_frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')
        row={'epoch':epoch,'mean_loss':float(np.mean(losses)),'validation_macro_recall':score,'best':best,'updates':updates}
        with (args.run/'epochs.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    result=json.loads((args.run/'validation.json').read_text(encoding='utf-8'))
    selected=torch.load(args.run/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);model.eval()
    image=validation[0][0][None].cuda().float()/255
    with torch.inference_mode():
        before=model(image);model.load_state_dict(selected['model']);after=model(image)
    identical=torch.equal(before,after)
    if not identical:raise RuntimeError('Phase checkpoint reload differs')
    write_json(args.run/'state.json',{'status':'completed','best_sha256':sha(args.run/'best.pt'),'updates':updates,'wall_s':time.monotonic()-started,
                                   'validation_quality':quality(result,config),'lockbox_opened':False,'checkpoint_reload_identical':identical,
                                   'validation_sha256':sha(args.run/'validation.json'),'validation_frames_sha256':sha(args.run/'validation_frames.jsonl')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/light_state_v1.json');p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    try: train(a)
    except Exception as exc:
        if a.run.is_dir(): write_json(a.run/'failure.json',{'error':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
