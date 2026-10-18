"""Exercise candidate geometry against real training bytes and a CPU detector.

Finite-loss/backward smoke check and actual panels, not model training/quality.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
import torch
from experimental_detector_geometry import transform_one
from predict_sunny import load_model
from senior_ai.sunny_data import sha,png
from senior_ai.core import write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    out=args.output.resolve();root=args.root.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'artifacts/runs'):raise ValueError('New artifact run required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(16)
    model,task,saved=load_model(args.checkpoint,'cpu')
    if task!='detection':raise ValueError('Detection checkpoint required')
    for rel,h in saved['provenance']['source']['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Parent implementation differs')
    if sha(root/'audit.json')!=saved['provenance']['data']['audit_sha256']:raise ValueError('Main audit changed')
    metadata_path=root/'metadata_inventory.json'
    if sha(metadata_path)!=saved['provenance']['data']['metadata_inventory_sha256']:raise ValueError('Metadata changed')
    metadata={r['path']:r['sha256'] for r in read(metadata_path)}
    images_path=root/'image_inventory.jsonl'
    if sha(images_path)!=saved['provenance']['data']['image_inventory_sha256']:raise ValueError('RGB inventory changed')
    inventory={r['path']:r['sha256'] for r in map(json.loads,images_path.read_text().splitlines())}
    if sha(root/'training/train.jsonl')!=metadata['training/train.jsonl']:raise ValueError('Train index changed')
    rows=[json.loads(line) for line in (root/'training/train.jsonl').read_text().splitlines()];selected=[]
    for row in rows:
        if row['split']!='train':raise ValueError('Train-only samples required')
        source=root/row['detection']
        if sha(source)!=metadata[row['detection']]:raise ValueError('Train labels changed')
        labels=read(source)
        if any(o['category_id']==3 for o in labels['positive_objects']):
            if not selected or selected[0][0]['group_id']!=row['group_id']:selected.append((row,labels))
        if len(selected)==2:break
    if len(selected)!=2:raise ValueError('Two original training groups with bicycle required')
    out.mkdir();batch=[];truth=[];records=[];parent_sha=sha(args.checkpoint)
    for i,(row,labels) in enumerate(selected):
        if sha(root/row['rgb'])!=inventory[row['rgb']]:raise ValueError('Train RGB changed')
        rgb=torch.from_numpy(png(root/row['rgb']).transpose(2,0,1).copy()).float()/255
        target={'boxes':torch.tensor([o['visible_bbox_xyxy'] for o in labels['positive_objects']],dtype=torch.float32).reshape(-1,4),
            'labels':torch.tensor([o['category_id'] for o in labels['positive_objects']],dtype=torch.int64),
            'ignore_boxes':torch.tensor([r['bbox_xyxy'] for r in labels['ignore_regions']],dtype=torch.float32).reshape(-1,4)}
        augmented,t,ledger=transform_one(rgb,target,scale=.5,flip=bool(i),offset=(40,80))
        batch.append(augmented);truth.append(t)
        canvas=Image.new('RGB',(1280,384))
        for side,(pixels,gt) in enumerate([(rgb,target),(augmented,t)]):
            picture=Image.fromarray((pixels.permute(1,2,0).numpy()*255).round().astype(np.uint8));draw=ImageDraw.Draw(picture)
            for box,label in zip(gt['boxes'].tolist(),gt['labels'].tolist()):
                draw.rectangle(box,outline='cyan',width=2);draw.text((box[0],box[1]),str(label),fill='cyan')
            for box in gt['ignore_boxes'].tolist():draw.rectangle(box,outline='yellow',width=1)
            canvas.paste(picture,(side*640,24))
        ImageDraw.Draw(canvas).text((4,4),'TRAIN original / paired scale.5 + optional flip; cyan positive, yellow ignore',fill='white')
        panel=out/f'geometry-{i}.png';canvas.save(panel)
        records.append({'sample':row,'rgb_sha256':sha(root/row['rgb']),'GT_sha256':sha(root/row['detection']),
            'transform':ledger,'panel':panel.name,'panel_sha256':sha(panel)})
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5);model.train();optimizer.zero_grad(set_to_none=True)
    losses=model(batch,truth);loss=sum(losses.values())
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    if sha(args.checkpoint)!=parent_sha:raise ValueError('Parent checkpoint changed')
    write_json(out/'report.json',{'status':'CPU_real_train_geometry_finite_step_only','checkpoint_sha256':parent_sha,
        'tool_sha256':sha(Path(__file__)),'geometry_source_sha256':sha(ROOT/'scripts/experimental_detector_geometry.py'),
        'main_audit_sha256':sha(root/'audit.json'),'records':records,'device':'cpu','threads':2,
        'loss':float(loss.detach()),'parts':{k:float(v.detach()) for k,v in losses.items()},'preclip_gradient_norm':float(norm),
        'RPN_supervised_samples':model.rpn.last_supervised_samples,'ROI_supervised_samples':model.roi_heads.last_supervised_samples,
        'formal_checkpoints_written':0,'limits':'One ephemeral CPU optimizer step with real train bytes; not CUDA capacity, quality, or formal training. Original checkpoint and active v5 implementation untouched.'})
    print(json.dumps({'status':'CPU_finite_step_passed','loss':float(loss.detach()),'output':str(out)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
