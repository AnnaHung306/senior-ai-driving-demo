"""GT-selected seen-train/validation CPU bicycle diagnostics, never acceptance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
import torch
from predict_sunny import load_model
from senior_ai.sunny_data import sha,png,DETECTION_CLASSES
from senior_ai.detection_metrics import evaluate_detection,iou
from senior_ai.core import write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    if args.output.exists():raise FileExistsError('Use a new diagnostic version')
    root=args.root.resolve();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.output.resolve().is_relative_to(root) or args.output.resolve().is_relative_to(args.checkpoint.resolve().parent):
        raise ValueError('Diagnostic output must be outside immutable data/checkpoint snapshot')
    model,task,saved=load_model(args.checkpoint,'cpu')
    if task!='detection':raise ValueError('Detector required')
    for rel,digest in saved['provenance']['source']['files'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Snapshot source differs: '+rel)
    for file,key in [('audit.json','audit_sha256'),('visual_review.json','review_sha256'),('metadata_inventory.json','metadata_inventory_sha256'),('image_inventory.jsonl','image_inventory_sha256')]:
        if sha(root/file)!=saved['provenance']['data'][key]:raise ValueError('Main signature differs')
    metadata={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')};samples=[]
    for split in ['train','validation']:
        index=root/f'training/{split}.jsonl'
        if sha(index)!=metadata[f'training/{split}.jsonl']:raise ValueError('Canonical index changed')
        rows=[json.loads(line) for line in index.read_text(encoding='utf-8').splitlines()]
        if any(r['split']!=split for r in rows):raise ValueError('Train/validation only')
        samples.extend({**r,'split_index':i} for i,r in enumerate(rows))
    def labels(row):
        path=(root/row['detection']).resolve()
        if not path.is_relative_to(root) or sha(path)!=metadata[row['detection']]:raise ValueError('Audited GT changed')
        source=read(path);bikes=[o for o in source['positive_objects'] if o['category_id']==3]
        if not bikes:return None
        obj=max(bikes,key=lambda o:o['visible_pixels']);b=obj['visible_bbox_xyxy']
        return {'sample':row,'largest_bike':obj,'box_area':(b[2]-b[0])*(b[3]-b[1])}
    with ThreadPoolExecutor(max_workers=4) as pool:candidates=[r for r in pool.map(labels,samples) if r]
    selected={}
    for split,group in sorted({(r['sample']['split'],r['sample']['group_id']) for r in candidates}):
        subset=[r for r in candidates if r['sample']['split']==split and r['sample']['group_id']==group]
        for reason,key in [('max_visible_bicycle_support',lambda r:(-r['largest_bike']['visible_pixels'],r['sample']['split_index'])),
            ('near_16x16_box_area',lambda r:(abs(r['box_area']-256),r['sample']['split_index']))]:
            row=min(subset,key=key);sample=row['sample'];identity=(split,sample['split_index'])
            selected.setdefault(identity,{**row,'selection_reasons':[]})['selection_reasons'].append(reason)
    inventory={r['path']:r['sha256'] for r in map(json.loads,(root/'image_inventory.jsonl').read_text().splitlines())}
    args.output.mkdir(parents=True)
    write_json(args.output/'contract.json',{'checkpoint_sha256':sha(args.checkpoint),'checkpoint_epoch':saved['epoch'],'checkpoint_updates':saved['updates'],
        'tool_sha256':sha(Path(__file__)),'main_audit_sha256':saved['provenance']['data']['audit_sha256'],
        'selection':'Per original train/validation group: maximum GT bicycle visible support and nearest256px box area, deterministic earliest index tie. Before inference, no model-score selection.',
        'selected':[{'sample':r['sample'],'reasons':r['selection_reasons']} for r in selected.values()],
        'scope':'Biased selected seen-train/validation CPU FP32 only; no full-split accuracy, official GPU acceptance, physical occlusion fractions, or heldout.'})
    records=[];panels={}
    for row in selected.values():
        sample=row['sample'];path=(root/sample['rgb']).resolve()
        if not path.is_relative_to(root) or sha(path)!=inventory[sample['rgb']]:raise ValueError('Audited RGB changed')
        gt_path=(root/sample['detection']).resolve()
        if not gt_path.is_relative_to(root) or sha(gt_path)!=metadata[sample['detection']]:raise ValueError('Selected GT changed after selection')
        rgb=png(path);source=read(gt_path)
        with torch.inference_mode():p=model([torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255])[0]
        predictions=[{'box':b,'label':int(label),'score':float(score)} for b,label,score in zip(p['boxes'].tolist(),p['labels'].tolist(),p['scores'].tolist())]
        targets=[]
        for obj in source['positive_objects']:
            b=obj['visible_bbox_xyxy']
            targets.append({'box':b,'label':obj['category_id'],'visible_pixels':obj['visible_pixels'],
                'truncated':b[0]==0 or b[1]==0 or b[2]==640 or b[3]==360,
                'light_state_class':int(obj.get('light_state_class',-1))})
        frame={'split':sample['split'],'group_id':sample['group_id'],'predictions':predictions,'targets':targets,'ignore_boxes':[r['bbox_xyxy'] for r in source['ignore_regions']]}
        bike_box=row['largest_bike']['visible_bbox_xyxy'];same=[p for p in predictions if p['label']==3 and iou(p['box'],[bike_box])[0]>=.5]
        correct=any(p['score']>=.5 for p in same)
        other=[p for p in predictions if p['label']!=3 and p['score']>=.5 and iou(p['box'],[bike_box])[0]>=.5]
        reason='detected' if correct else 'correct_box_low_score' if same else 'wrong_class_overlap' if other else 'no_matching_bicycle_box'
        record={'sample':sample,'selection_reasons':row['selection_reasons'],'selected_bicycle_box':bike_box,'selected_bicycle_visible_pixels':row['largest_bike']['visible_pixels'],
            'selected_bicycle_diagnostic':reason,'wrong_class_overlaps':other,'highest_matching_bicycle_score':max((p['score'] for p in same),default=None),'evaluation_frame':frame}
        records.append(record);gt=Image.fromarray(rgb.copy());prediction=Image.fromarray(rgb.copy())
        ImageDraw.Draw(gt).rectangle(bike_box,outline='cyan',width=2)
        for obj in predictions:
            if obj['score']>=.5:
                draw=ImageDraw.Draw(prediction);draw.rectangle(obj['box'],outline='yellow' if obj['label']==3 else 'gray',width=2)
                draw.text((obj['box'][0],max(0,obj['box'][1]-11)),f"{obj['label']} {obj['score']:.2f}",fill='yellow')
        panel=Image.new('RGB',(1280,384));panel.paste(gt,(0,24));panel.paste(prediction,(640,24))
        ImageDraw.Draw(panel).text((4,4),f"{sample['split']} {sample['group_id']}/{sample['camera']} frame{sample['frame']} {reason} | GTbike / CPU predictions",fill='white')
        panels[len(records)-1]=panel;write_json(args.output/'progress.json',{'completed_views':len(records),'total':len(selected)})
    # All validation selections plus up to three hardest seen-train contexts.
    display=[i for i,r in enumerate(records) if r['sample']['split']=='validation']
    train=sorted([i for i,r in enumerate(records) if r['sample']['split']=='train'],key=lambda i:(records[i]['selected_bicycle_diagnostic']=='detected',records[i]['highest_matching_bicycle_score'] or 0,i))
    display+=train[:3];pages=[]
    for start in range(0,len(display),3):
        indices=display[start:start+3];page=Image.new('RGB',(1280,384*len(indices)))
        for j,index in enumerate(indices):page.paste(panels[index],(0,j*384))
        p=args.output/f'page-{start//3:02d}.png';page.save(p);pages.append({'path':p.name,'sha256':sha(p),'record_indices':indices})
    results={split:evaluate_detection([r['evaluation_frame'] for r in records if r['sample']['split']==split],DETECTION_CLASSES) for split in ['train','validation']}
    write_json(args.output/'report.json',{'status':'completed_selected_CPU_diagnostic_only','checkpoint_sha256':sha(args.checkpoint),'views':len(records),'results':results,'records':records,'pages':pages,
        'limits':'Deliberately GT-selected seen train and validation views; CPU outputs can differ numerically from CUDA. No new model/gates/heldout or full-split train generalization estimate.'})
    print(json.dumps({'views':len(records),'selected_bicycle_results':{split:r['per_class']['bicycle'] for split,r in results.items()},'pages':len(pages)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
