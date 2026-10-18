"""Frozen sunny detector/segmentation training, validation selection, epoch resume.

No CARLA client. Lockbox is deliberately absent from this command.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import torchvision
import PIL
from torch.utils.data import DataLoader
from senior_ai.core import write_json
from senior_ai.detection_model import make_detector
from senior_ai.detection_metrics import evaluate_detection
from senior_ai.segmentation_model import TinyUNet
from senior_ai.mobile_segmentation import MobileUNet
from senior_ai.sunny_augmentation import paired_geometry
from senior_ai.segmentation_metrics import confusion, metrics
from senior_ai.segmentation_training import weighted_loss, seed_all, configure_cuda
from senior_ai.sunny_data import SunnySamples, SEGMENTATION_NAMES, DETECTION_CLASSES, sha, verify
from senior_ai.sunny_losses import segmentation_loss
from senior_ai.road_boundaries import boundary_counts,boundary_metrics
from senior_ai.segmentation_supplement import verify_scenery,append_scenery
from senior_ai.segmentation_expansion import verify_expansion,append_expansion
from senior_ai.detection_augmentation import bicycle_scale_batch
from senior_ai.perception_quality import quality,selection_key


def source_fingerprint():
    files=sorted((ROOT/'src').rglob('*.py'))+[ROOT/'scripts'/name for name in ['train_sunny.py','predict_sunny.py','evaluate_sunny.py',
        'prepare_segmentation_supplement.py','collect_segmentation_supplement.py','audit_segmentation_supplement.py','collect_light_supplement.py',
        'collect_dataset.py','rig_capture.py','validate_dataset.py']]
    values={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    return {'files':values,'sha256':hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()}


def build_model(task,config,pretrained=True):
    c=config[task]; path=ROOT/c['pretrained_path']
    if pretrained and c.get('initial_checkpoint_path'):
        initial=ROOT/c['initial_checkpoint_path']
        if sha(initial)!=c['initial_checkpoint_sha256']:raise ValueError('Initial project checkpoint changed')
        saved=torch.load(initial,map_location='cpu',weights_only=False)
        if saved['task']!=task or saved['provenance']['data']['audit_sha256']!=c['initial_data_audit_sha256']:
            raise ValueError('Initial checkpoint task or training audit differs')
        model=build_model(task,config,pretrained=False)
        model.load_state_dict(saved['model'])
        return model
    if pretrained and sha(path)!=c['pretrained_sha256']: raise ValueError('Starting weights changed')
    if task=='detection': return make_detector(weights=path if pretrained else None,
        min_size=c.get('min_size',360),max_size=c.get('max_size',640))
    if c.get('architecture')=='mobilenet_unet':
        model=MobileUNet(classes=len(SEGMENTATION_NAMES),precision=c['precision'])
        if pretrained:model.load_encoder(path)
        return model
    model=TinyUNet(width=c['width'],classes=len(SEGMENTATION_NAMES))
    if pretrained:
        old=torch.load(path,map_location='cpu',weights_only=False)['model']
        state=model.state_dict()
        for name,value in old.items():
            if not name.startswith('head.'): state[name]=value
        # Explicit old-to-new class transfer; other new classes retain seeded init.
        for new,prior in {0:0,1:1,2:2,3:3,5:4,8:5,9:6,10:7}.items():
            state['head.weight'][new]=old['head.weight'][prior]
            state['head.bias'][new]=old['head.bias'][prior]
        model.load_state_dict(state)
    return model


def batches(data,size,shuffle=False):
    return DataLoader(data,batch_size=size,shuffle=shuffle,num_workers=0,collate_fn=lambda samples:list(zip(*samples)))


def inputs(batch,augment=False,photometric_range=(.9,1.1)):
    rgb=torch.stack(batch).cuda().float().div_(255)
    if augment:
        low,high=photometric_range
        contrast=torch.rand((len(rgb),1,1,1),device='cuda')*(high-low)+low
        brightness=torch.rand_like(contrast)*(high-low)+low
        mean=rgb.mean((2,3),keepdim=True); rgb=((rgb-mean)*contrast+mean).mul(brightness).clamp_(0,1)
    return rgb


def step(model,task,rgb,target,optimizer,weights,clip):
    model.train(); optimizer.zero_grad(set_to_none=True)
    if task=='detection':
        labels=[{k:v.cuda() for k,v in t.items()} for t in target]
        parts=model(list(rgb),labels); loss=sum(parts.values())
        if not model.rpn.last_supervised_samples and not model.roi_heads.last_supervised_samples:
            return 0.,0.,{'_skipped_empty_supervision':True}
    else:
        truth=target if torch.is_tensor(target) else torch.stack(target)
        loss=segmentation_loss(model(rgb),truth.cuda().long(),weights); parts={'weighted_ce_plus_foreground_dice':loss}
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()),float(norm),{k:float(v.detach()) for k,v in parts.items()}


@torch.inference_mode()
def evaluate(model,task,data,size):
    model.eval(); grouped={}; frames=[]; matrices=[];margin_counts=Counter()
    for rgb,target,indices in batches(data,size):
        images=inputs(rgb)
        if task=='detection':
            predictions=model(list(images))
            for p,t,index in zip(predictions,target,indices):
                row={'index':index,**data.samples[index],
                    'predictions':[{'box':b,'label':int(l),'score':float(s)} for b,l,s in zip(p['boxes'].cpu().tolist(),p['labels'].cpu().tolist(),p['scores'].cpu().tolist())],
                    'targets':[{'box':b,'label':int(l),'visible_pixels':int(v),'truncated':bool(tr),'light_state_class':int(phase)} for b,l,v,tr,phase in zip(t['boxes'].tolist(),t['labels'].tolist(),t['visible_pixels'].tolist(),t['truncated'].tolist(),t['light_state'].tolist())],
                    'ignore_boxes':t['ignore_boxes'].tolist()}
                frames.append(row)
        else:
            logits=model(images)
            if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite inference logits')
            predicted=logits.argmax(1).cpu().numpy()
            for p,t,index in zip(predicted,target,indices):
                m=confusion(t.numpy(),p,classes=len(SEGMENTATION_NAMES)); matrices.append(m)
                border=boundary_counts(t.numpy(),p);margin_counts.update(border)
                row=data.samples[index]; grouped.setdefault(row['scene_id'],np.zeros_like(m))[:] += m
                frames.append({'index':index,**row,**metrics(m,SEGMENTATION_NAMES),'visible_margin':boundary_metrics(border)})
    if task=='detection':
        result=evaluate_detection(frames,DETECTION_CLASSES)
        result['by_scene']={name:evaluate_detection([f for f in frames if f['scene_id']==name],DETECTION_CLASSES)
                            for name in sorted({f['scene_id'] for f in frames})}
    else:
        result=metrics(sum(matrices),SEGMENTATION_NAMES)
        result['visible_margin']=boundary_metrics(margin_counts)
        result['by_scene']={k:metrics(v,SEGMENTATION_NAMES) for k,v in grouped.items()}
    return result,frames


def save_checkpoint(path,value):
    temp=path.with_suffix('.pt.tmp'); torch.save(value,temp); temp.replace(path)


def make_optimizer(model,c):
    if c.get('encoder_learning_rate') is not None:
        encoder=list(model.encoder.parameters());ids={id(p) for p in encoder}
        groups=[{'params':encoder,'lr':c['encoder_learning_rate']},{'params':[p for p in model.parameters() if id(p) not in ids],'lr':c['learning_rate']}]
        return torch.optim.AdamW(groups,weight_decay=c['weight_decay'])
    return torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])


def require_idle_gpu():
    # Model training must not compete with the 32-camera server on 8GB VRAM.
    processes=subprocess.check_output(['powershell','-NoProfile','-Command',
        "Get-Process -Name 'CarlaUE4*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ProcessName"],text=True)
    memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if 'CarlaUE4' in processes or memory>2048:
        raise RuntimeError('CARLA must be stopped and at least 6GiB GPU memory free before training')


def run(args):
    config=json.loads(args.config.read_text(encoding='utf-8')); task=args.task; c=config[task]
    root=args.dataset_root.resolve() if args.dataset_root else ROOT/config['dataset']; run=args.run.resolve()
    require_idle_gpu()
    configure_cuda(); seed_all(config['seed'])
    if c.get('precision')=='bfloat16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required by this experiment')
    provenance={'data':verify(root),'source':source_fingerprint(),'config_sha256':sha(args.config),
                'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
                'numpy':np.__version__,'pillow':PIL.__version__,'cudnn':torch.backends.cudnn.version(),
                'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
                'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
                'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    if c.get('initial_checkpoint_path') and provenance['data']['audit_sha256']!=c['initial_data_audit_sha256']:
        raise ValueError('Fine-tuning initialization requires the explicitly bound main training audit')
    scenery=c.get('training_scenery')
    if scenery:
        if task!='segmentation':raise ValueError('Static scenery is segmentation training only')
        provenance['training_supplement']=verify_scenery(ROOT/scenery['root'],provenance['data']['audit_sha256'],scenery['plan_sha256'])
        provenance['training_supplement']['sampling_repeats']=scenery['sampling_repeats']
    expansions=c.get('training_expansions',[])
    if expansions:
        if task!='segmentation' or len({str((ROOT/e['root']).resolve()) for e in expansions})!=len(expansions):raise ValueError('Unique segmentation-only expansions required')
        provenance['training_expansions']=[{**verify_expansion(ROOT/e['root'],root,e['plan_sha256'],e['audit_sha256'],e['review_sha256']),
            'sampling_repeats':e['sampling_repeats']} for e in expansions]
    if run.exists() and not args.resume: raise FileExistsError('New experiment requires new run directory')
    if args.resume:
        prior=json.loads((run/'provenance.json').read_text(encoding='utf-8'))
        # Git may advance for docs alone. Exact source bytes, config, data and
        # runtime are authoritative; retain the original Git attribution.
        if {k:v for k,v in prior.items() if k!='git_revision'}!={k:v for k,v in provenance.items() if k!='git_revision'}:
            raise ValueError('Resume source/config/data/runtime differs')
        provenance=prior
        state=json.loads((run/'state.json').read_text(encoding='utf-8'))
        if state['status']=='completed' or (run/'lockbox_started.json').exists(): raise ValueError('Completed or evaluated run cannot resume')
        if sha(run/'last.pt')!=state['last_sha256']: raise ValueError('Checkpoint integrity failure')
    else:
        run.mkdir(parents=True);(run/'config.json').write_bytes(args.config.read_bytes());write_json(run/'provenance.json',provenance)
        if sha(run/'config.json')!=provenance['config_sha256']:raise ValueError('Frozen config bytes differ')
        for relative in provenance['source']['files']:
            destination=run/'source'/relative; destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes((ROOT/relative).read_bytes())
    started=time.monotonic()
    train=SunnySamples(root,'train',task,config['cache_budget_gib_per_split'])
    validation=SunnySamples(root,'validation',task,config['cache_budget_gib_per_split'])
    sampling=append_scenery(train,ROOT/scenery['root'],scenery['sampling_repeats'],config['cache_budget_gib_per_split']) if scenery else None
    expansion_sampling=[append_expansion(train,ROOT/e['root'],e['sampling_repeats'],config['cache_budget_gib_per_split']) for e in expansions]
    write_json(run/'dataset_statistics.json',{'train_views':len(train),'validation_views':len(validation),
        'train_groups':sorted({s['group_id'] for s in train.samples}),'validation_groups':sorted({s['group_id'] for s in validation.samples}),
        'train_segmentation_pixels':train.support.tolist(),'cache_budget_gib_per_split':config['cache_budget_gib_per_split'],
        'static_training_supplement':sampling,'static_training_expansions':expansion_sampling,
        'detection_bicycle_scale_policy':c.get('bicycle_scale_geometry'),
        'sample_count_note':'Repeated supplemental views are optimization samples, not independent scenes'})
    weights=torch.tensor(np.clip(np.sqrt(max(1,train.support.max())/np.maximum(1,train.support)),1,20),device='cuda',dtype=torch.float32)
    if not args.resume:
        model=build_model(task,config).cuda(); optimizer=make_optimizer(model,c)
        if task=='detection':
            choices=sorted(range(len(train)),key=lambda i:(-len(train.targets[i]['boxes']),i))[:2]
        else: choices=[0,len(train)//2]
        rgb=inputs([train[i][0] for i in choices]); target=[train[i][1] for i in choices]; losses=[]
        for _ in range(c['overfit_steps']): losses.append(step(model,task,rgb,target,optimizer,weights,c['gradient_clip'])[0])
        ratio=float(np.mean(losses[-10:])/np.mean(losses[:10])); sanity={'losses':losses,'ratio':ratio,'indices':choices,'passed':ratio<=c['overfit_loss_ratio_max']}
        write_json(run/'overfit.json',sanity)
        if not sanity['passed']: raise RuntimeError('Train-only overfit sanity failed')
        del model,optimizer,rgb,target; gc.collect(); torch.cuda.empty_cache(); seed_all(config['seed'])
    model=build_model(task,config,pretrained=not args.resume).cuda(); optimizer=make_optimizer(model,c)
    total_steps=math.ceil(len(train)/c['batch_size'])*c['epochs']
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1,(s+1)/max(1,c['warmup_steps']))*.5*(1+math.cos(math.pi*min(s,total_steps)/total_steps)))
    first_epoch=0; best=-math.inf; best_qualified=False;updates=0
    selection_policy=config.get('selection_policy','score_only')
    selection_key(best,best_qualified,selection_policy)
    if args.resume:
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); schedule.load_state_dict(saved['scheduler'])
        first_epoch=saved['epoch']+1; best=saved['best_score'];best_qualified=saved.get('best_validation_qualified',False); updates=saved['updates']
        if task=='detection':
            model.rpn.ignore_assignment_calls=saved.get('rpn_ignore_calls',0);model.roi_heads.ignore_assignment_calls=saved.get('roi_ignore_calls',0)
        random.setstate(saved['rng']['python']); np.random.set_state(saved['rng']['numpy']); torch.set_rng_state(saved['rng']['torch']); torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    else:
        baseline,_=evaluate(model,task,validation,c['batch_size']); write_json(run/'initial_validation.json',baseline)
    for epoch in range(first_epoch,c['epochs']):
        losses=[]; max_norm=0.; epoch_start=time.monotonic();skipped=0;geometry_counts=Counter()
        for rgb,target,indices in batches(train,c['batch_size'],shuffle=True):
            images=inputs(rgb,augment=True,photometric_range=c.get('photometric_range',(.9,1.1)))
            if task=='segmentation' and c.get('paired_geometry'):
                images,target=paired_geometry(images,torch.stack(target).cuda(),**c['paired_geometry'])
            if task=='detection' and c.get('bicycle_scale_geometry'):
                images,target,ledger=bicycle_scale_batch(images,target,**c['bicycle_scale_geometry'])
                geometry_counts.update({'views':len(ledger),'bicycle_positive_views':sum(r['bicycle_positive'] for r in ledger),'zoomed_views':sum(r['zoomed'] for r in ledger)})
            loss,norm,parts=step(model,task,images,target,optimizer,weights,c['gradient_clip'])
            if parts.get('_skipped_empty_supervision'):
                skipped+=1;continue
            losses.append(loss); max_norm=max(max_norm,norm); updates+=1; schedule.step()
            if updates%100==0:
                write_json(run/'progress.json',{'epoch':epoch,'updates':updates,'mean_loss':float(np.mean(losses)),'lr':optimizer.param_groups[0]['lr']})
        if not losses: raise ValueError('Epoch has no supervised training batches')
        result,_=evaluate(model,task,validation,c['batch_size'])
        score=result['mAP50_95'] if task=='detection' else result['focus_macro_iou']
        if score is None or not math.isfinite(score): raise FloatingPointError('Nonfinite validation score')
        validation_quality=quality(task,result,config['quality_gates'])
        improved=selection_key(score,validation_quality['passed'],selection_policy)>selection_key(best,best_qualified,selection_policy)
        if improved: best=score;best_qualified=validation_quality['passed']; write_json(run/'best_validation.json',result)
        checkpoint={'task':task,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':schedule.state_dict(),
            'epoch':epoch,'updates':updates,'best_score':best,'best_validation_qualified':best_qualified,'config':config,'provenance':provenance,
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
        if task=='detection':checkpoint.update({'rpn_ignore_calls':model.rpn.ignore_assignment_calls,'roi_ignore_calls':model.roi_heads.ignore_assignment_calls})
        save_checkpoint(run/'last.pt',checkpoint)
        if improved: save_checkpoint(run/'best.pt',checkpoint)
        row={'epoch':epoch,'updates':updates,'mean_loss':float(np.mean(losses)),'max_gradient_norm':max_norm,
             'skipped_fully_ignored_batches':skipped,
             'validation_score':score,'best_score':best,'improved':improved,'wall_s':time.monotonic()-epoch_start}
        row['validation_quality']=validation_quality;row['selected_validation_qualified']=best_qualified
        if geometry_counts:row['training_geometry_counts']=dict(geometry_counts)
        with (run/'epochs.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
        write_json(run/'state.json',{'status':'training','task':task,'last_sha256':sha(run/'last.pt'),'epoch':epoch,'updates':updates})
        print(json.dumps(row),flush=True)
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False); model.load_state_dict(checkpoint['model'])
    # Reload roundtrip must produce exactly the same RGB-only output in eval mode.
    model.eval(); images=inputs([validation[0][0]])
    with torch.inference_mode():
        before=model(list(images) if task=='detection' else images)
        model.load_state_dict(checkpoint['model']); after=model(list(images) if task=='detection' else images)
    if task=='detection': identical=all(torch.equal(before[0][k],after[0][k]) for k in before[0])
    else: identical=torch.equal(before,after)
    if not identical: raise RuntimeError('Checkpoint reload RGB inference differs')
    result,frames=evaluate(model,task,validation,c['batch_size']); write_json(run/'validation.json',result)
    (run/'validation_frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')
    receipt={'status':'completed','task':task,'best_epoch':checkpoint['epoch'],'best_score':best,'updates':updates,
             'last_sha256':sha(run/'last.pt'),'best_sha256':sha(run/'best.pt'),'checkpoint_reload_identical':identical,
             'training_wall_s':time.monotonic()-started,'max_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
             'parameter_count':sum(p.numel() for p in model.parameters()),
             'selected_validation_qualified':best_qualified,'selection_policy':selection_policy,
             'max_cuda_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
             'rpn_ignore_calls':getattr(getattr(model,'rpn',None),'ignore_assignment_calls',None),
             'roi_ignore_calls':getattr(getattr(model,'roi_heads',None),'ignore_assignment_calls',None),'lockbox_opened':False}
    receipt['validation_sha256']=sha(run/'validation.json');receipt['validation_frames_sha256']=sha(run/'validation_frames.jsonl')
    write_json(run/'state.json',receipt); print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--task',choices=['detection','segmentation'],required=True); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset-root',type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try: run(a)
    except Exception as exc:
        if a.run.is_dir(): write_json(a.run/'failure.json',{'error':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
