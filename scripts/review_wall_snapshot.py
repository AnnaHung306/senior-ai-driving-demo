"""CPU-only wall diagnostics for an intermediate checkpoint; no acceptance scoring."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
import torch
from predict_sunny import load_model,PALETTE
from senior_ai.sunny_data import sha,png,segmentation_target,SEGMENTATION_NAMES
from senior_ai.segmentation_metrics import confusion,metrics
from senior_ai.dataset import safe_path
from senior_ai.core import write_json


def main(args):
    if args.output.exists():raise FileExistsError('Use a new diagnostic version')
    state=json.loads((args.selection_run/'state.json').read_text(encoding='utf-8'))
    frames_path=args.selection_run/'validation_frames.jsonl'
    if state['status']!='completed' or sha(frames_path)!=state['validation_frames_sha256']:raise ValueError('Completed, intact original validation selection required')
    old=[json.loads(line) for line in frames_path.read_text(encoding='utf-8').splitlines()]
    if any(row['split']!='validation' for row in old):raise ValueError('Validation only')
    selected={}
    for scene in sorted({r['scene_id'] for r in old}):
        rows=[r for r in old if r['scene_id']==scene]
        for reason,key in [('wall_false_positive',lambda r:sum(r['confusion_true_rows_predicted_columns'][i][12] for i in range(17) if i!=12)),
            ('wall_as_fence',lambda r:r['confusion_true_rows_predicted_columns'][12][13]),
            ('wall_GT_support',lambda r:r['classes'][12]['support_pixels'])]:
            row=max(rows,key=lambda r:(key(r),-r['index']))
            if key(row)>0:selected.setdefault(row['index'],{'row':row,'reasons':[]})['reasons'].append(reason)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    checkpoint_sha=sha(args.checkpoint);model,task,saved=load_model(args.checkpoint,'cpu')
    if task!='segmentation':raise ValueError('Segmentation checkpoint required')
    for relative,digest in saved['provenance']['source']['files'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Diagnostic source differs from snapshot: '+relative)
    signature=saved['provenance']['data']
    for file,key in [('audit.json','audit_sha256'),('visual_review.json','review_sha256'),('image_inventory.jsonl','image_inventory_sha256'),('metadata_inventory.json','metadata_inventory_sha256')]:
        if sha(args.root/file)!=signature[key]:raise ValueError('Main data binding differs')
    metadata={r['path']:r['sha256'] for r in json.loads((args.root/'metadata_inventory.json').read_text(encoding='utf-8'))}
    if sha(args.root/'training/validation.jsonl')!=metadata['training/validation.jsonl']:raise ValueError('Validation index changed')
    canonical=[json.loads(line) for line in (args.root/'training/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    required={case['row'][k] for case in selected.values() for k in ['rgb','semantic','instance']}
    inventory={}
    for line in (args.root/'image_inventory.jsonl').read_text(encoding='utf-8').splitlines():
        item=json.loads(line)
        if item['path'] in required:
            p=safe_path(args.root,item['path'])
            if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise ValueError('Selected image changed')
            inventory[item['path']]=item['sha256']
    if set(inventory)!=required:raise ValueError('Selected image inventory incomplete')
    args.output.mkdir(parents=True);records=[]
    for index,case in sorted(selected.items()):
        row=case['row']
        if any(row[k]!=v for k,v in canonical[index].items()):raise ValueError('Selection is not canonical validation')
        rgb=png(safe_path(args.root,row['rgb']));truth=segmentation_target(png(safe_path(args.root,row['semantic']))[...,0],png(safe_path(args.root,row['instance'])),row['verified_obstacle_actor_ids'])
        tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255
        with torch.inference_mode():prediction=model(tensor)[0].argmax(0).numpy().astype(np.uint8)
        colors=np.zeros_like(rgb);valid=truth!=255;colors[valid]=PALETTE[truth[valid]]
        errors=rgb.copy();actual=truth==12;predicted=prediction==12
        errors[valid&predicted&~actual]=[255,100,0];errors[valid&actual&~predicted]=[0,255,255];errors[valid&actual&predicted]=[0,255,0]
        panel=Image.new('RGB',(1920,310));draw=ImageDraw.Draw(panel)
        for j,(pixels,label) in enumerate([(rgb,'RGB'),(colors,'GROUND TRUTH'),(PALETTE[prediction],'CPU FP32 PREDICTION'),(errors,'WALL TP=GREEN FP=ORANGE FN=CYAN')]):
            panel.paste(Image.fromarray(pixels).resize((480,270),Image.Resampling.NEAREST if j in [1,2] else Image.Resampling.BILINEAR),(480*j,40));draw.text((480*j+4,22),label,fill='white')
        draw.text((4,4),f'{row["scene_id"]}/{row["camera"]} frame{row["frame"]} | '+','.join(case['reasons']),fill='white')
        path=args.output/f'{len(records):02d}.png';panel.save(path)
        records.append({'index':index,'sample':canonical[index],'reasons':case['reasons'],'CPU_wall':metrics(confusion(truth,prediction,17),SEGMENTATION_NAMES)['classes'][12],
            'panel':path.name,'sha256':sha(path)})
        write_json(args.output/'catalog.json',{'status':'in_progress','checkpoint_sha256':checkpoint_sha,'epoch':saved['epoch'],'updates':saved['updates'],'records':records})
    pages=[]
    for start in range(0,len(records),3):
        subset=records[start:start+3];page=Image.new('RGB',(1920,310*len(subset)))
        for j,record in enumerate(subset):page.paste(Image.open(args.output/record['panel']),(0,310*j))
        p=args.output/f'page-{start//3}.png';page.save(p);pages.append({'path':p.name,'sha256':sha(p)})
    write_json(args.output/'catalog.json',{'status':'completed_diagnostic_only','checkpoint_sha256':checkpoint_sha,'epoch':saved['epoch'],'updates':saved['updates'],
        'selection_run':str(args.selection_run),'selection_frames_sha256':sha(frames_path),'selected_raw_sha256':inventory,'tool_sha256':sha(Path(__file__)),
        'scope':'CPU FP32 selected validation examples; never substitutes for full GPU BF16 gates, no heldout opened. Selection fixed from the completed v4 baseline, not new model scores.',
        'records':records,'pages':pages})
    print(json.dumps({'status':'completed_diagnostic_only','views':len(records),'checkpoint_sha256':checkpoint_sha}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--selection-run',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
