"""Open heldout splits once for a validation-qualified, completed checkpoint."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image,ImageDraw
import torch

from train_sunny import ROOT, evaluate, configure_cuda
from predict_sunny import load_model,PALETTE
from senior_ai.core import write_json
from senior_ai.sunny_data import SunnySamples,DETECTION_CLASSES,SEGMENTATION_NAMES,sha,verify
from senior_ai.detection_metrics import evaluate_detection
from senior_ai.segmentation_supplement import verify_scenery
from senior_ai.perception_quality import quality


def gallery(model,task,data,frames,output):
    if task=='detection':
        scored=[]
        for row in frames:
            r=evaluate_detection([row],DETECTION_CLASSES)
            counts=r['per_class'];tp=sum(c['true_positives'] for c in counts.values());fp=sum(c['false_positives'] for c in counts.values());fn=sum(c['false_negatives'] for c in counts.values())
            denominator=2*tp+fp+fn
            scored.append((2*tp/denominator if denominator else 1.,row['index'],{'tp':tp,'fp':fp,'fn':fn}))
    else:
        scored=[(f['focus_macro_iou'] if f['focus_macro_iou'] is not None else 1.,f['index'],{'classes':f['classes']}) for f in frames]
    scored.sort(key=lambda r:(r[0],r[1])); selected=scored[:10]
    if task=='segmentation':
        # A focus-empty negative image has undefined focus IoU. Explicitly
        # include its false positives rather than ranking it as a failure-free 1.
        def false_pixels(row):
            m=np.array(row['confusion_true_rows_predicted_columns']);return int((m.sum(0)-m.diagonal())[1:].sum())
        most_fp=max(frames,key=lambda row:(false_pixels(row),-row['index']))
        if most_fp['index'] not in {s[1] for s in selected}:selected.append(next(v for v in scored if v[1]==most_fp['index']))
    # Include first view with GT support for each object/semantic class, so rare
    # categories cannot disappear from a gallery chosen only by whole-image score.
    for category in range(1,9 if task=='detection' else len(SEGMENTATION_NAMES)):
        candidates=[r for r in frames if (any(t['label']==category for t in r['targets']) if task=='detection' else r['classes'][category]['support_pixels']>0)]
        if candidates:
            worst=min(candidates,key=lambda r:(next(v[0] for v in scored if v[1]==r['index']),r['index']))
            if worst['index'] not in {s[1] for s in selected}: selected.append(next(v for v in scored if v[1]==worst['index']))
    output.mkdir(parents=True,exist_ok=False); catalog=[]; names={v:k for k,v in DETECTION_CLASSES.items()}
    for rank,(score,index,detail) in enumerate(selected):
        rgb,target,_=data[index]; image=rgb.permute(1,2,0).numpy(); tensor=rgb.cuda().float()/255
        with torch.inference_mode(): prediction=model([tensor])[0] if task=='detection' else model(tensor[None])[0].argmax(0).cpu().numpy()
        if task=='detection':
            gt=Image.fromarray(image); pred=gt.copy(); dg,dp=ImageDraw.Draw(gt),ImageDraw.Draw(pred)
            for b,l in zip(target['boxes'].tolist(),target['labels'].tolist()): dg.rectangle(b,outline='lime',width=2);dg.text((b[0],max(0,b[1]-12)),names[l],fill='lime')
            for b,l,s in zip(prediction['boxes'].cpu().tolist(),prediction['labels'].cpu().tolist(),prediction['scores'].cpu().tolist()):
                if s>=.5: dp.rectangle(b,outline='yellow',width=2);dp.text((b[0],max(0,b[1]-12)),f'{names[l]} {s:.2f}',fill='yellow')
        else:
            target=target.numpy(); colors=np.zeros((*target.shape,3),dtype=np.uint8); valid=target!=255;colors[valid]=PALETTE[target[valid]]
            gt=Image.fromarray(colors);pred=Image.fromarray(PALETTE[prediction])
        panel=Image.new('RGB',(1920,384));panel.paste(Image.fromarray(image),(0,24));panel.paste(gt,(640,24));panel.paste(pred,(1280,24))
        d=ImageDraw.Draw(panel);d.text((4,5),'RGB',fill='white');d.text((644,5),'GROUND TRUTH',fill='lime');d.text((1284,5),'LEARNED PREDICTION',fill='yellow')
        path=output/f'{rank:02d}.png';panel.save(path)
        catalog.append({'rank':rank,'selection_score':score,'sample':data.samples[index],'counts':detail,'panel':str(path),'sha256':sha(path)})
    write_json(output/'error_catalog.json',catalog)


def main(args):
    run=args.run.resolve(); state=json.loads((run/'state.json').read_text(encoding='utf-8'))
    if state['status']!='completed' or not state['checkpoint_reload_identical']: raise ValueError('Completed, verified training required')
    if sha(run/'best.pt')!=state['best_sha256']: raise ValueError('Selected model changed')
    model,task,saved=load_model(run/'best.pt','cpu');config=saved['config'];root=args.dataset_root.resolve() if args.dataset_root else ROOT/config['dataset']
    for relative,digest in saved['provenance']['source']['files'].items():
        if sha(ROOT/relative)!=digest: raise ValueError('Evaluation source differs from frozen training source: '+relative)
    if verify(root)!=saved['provenance']['data']: raise ValueError('Training dataset differs')
    supplement=saved['provenance'].get('training_supplement')
    if supplement:
        signature=verify_scenery(Path(supplement['root']),saved['provenance']['data']['audit_sha256'],supplement['plan_sha256'])
        if any(signature[key]!=value for key,value in supplement.items() if key!='sampling_repeats'):
            raise ValueError('Training scenery provenance differs')
    if sha(run/'validation.json')!=state['validation_sha256'] or sha(run/'validation_frames.jsonl')!=state['validation_frames_sha256']:
        raise ValueError('Validation results changed after training')
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'));gates={**plan['quality_gates'],**config.get('quality_gates',{})}
    val=json.loads((run/'validation.json').read_text(encoding='utf-8')); gate=quality(task,val,gates)
    write_json(run/'validation_quality.json',gate)
    if not gate['passed']:
        print(json.dumps({'status':'validation_quality_failed','lockbox_opened':False,**gate}));return 2
    marker=run/'lockbox_started.json'
    if marker.exists(): raise FileExistsError('Heldout evaluation already started; cannot silently reuse lockbox')
    write_json(marker,{'best_sha256':state['best_sha256'],'task':task,'source_sha256':sha(Path(__file__))})
    configure_cuda();model.cuda().eval()
    receipts={}
    for split in ['development','lockbox']:
        data=SunnySamples(root,split,task,allow_lockbox=split=='lockbox');result,frames=evaluate(model,task,data,config[task]['batch_size'])
        write_json(run/(split+'.json'),result);(run/(split+'_frames.jsonl')).write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')
        gallery(model,task,data,frames,run/(split+'_errors'))
        receipts[split]=quality(task,result,gates);del data,frames
    receipt={'status':'passed' if receipts['lockbox']['passed'] else 'quality_failed','task':task,
             'checkpoint_sha256':state['best_sha256'],'quality':receipts,'source_sha256':sha(Path(__file__)),
             'scope':'ClearNoon ideal simulation only; no driving or rainy/night generalization claim'}
    write_json(run/'final_evaluation.json',receipt);print(json.dumps(receipt));return 0 if receipt['status']=='passed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset-root',type=Path);sys.exit(main(p.parse_args()))
