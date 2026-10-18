"""Resumable GT-only train/validation box-size and anchor-shape diagnostic.

Uses official TorchVision rounded base anchors on CPU, with freely aligned
centres: the IoU is an optimistic shape upper bound, not actual grid coverage,
RPN positives, trained accuracy or a recommendation to change an active run.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import box_iou
from survey_sunny_materials import write
from senior_ai.dataset import safe_path
from senior_ai.sunny_data import DETECTION_CLASSES


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def centered_anchor_iou(widths_heights,sizes,ratios=(.5,1.,2.)):
    values=torch.tensor(widths_heights,dtype=torch.float32,device='cpu').reshape(-1,2)
    if not len(values):return np.empty(0,dtype=np.float32)
    anchors=AnchorGenerator().generate_anchors(list(sizes),list(ratios),device=torch.device('cpu'))
    gt=torch.cat((-values/2,values/2),1)
    return box_iou(gt,anchors).max(1).values.numpy()


def summarize(records,identity,sizes,status):
    output={}
    for split in ['train','validation']:
        subset=[r for r in records if r['split']==split];classes={}
        for name,category in DETECTION_CLASSES.items():
            rows=[obj for r in subset for obj in r['objects'] if obj[0]==category]
            if not rows:continue
            wh=np.array([[r[1],r[2]] for r in rows]);area=wh.prod(1)
            base=centered_anchor_iou(wh,sizes);small=centered_anchor_iou(wh,sorted(set([8]+sizes)))
            tall=centered_anchor_iou(wh,sizes,(.25,.5,1.,2.,4.))
            classes[name]={'objects':len(rows),'bbox_width_q10_q50_q90':np.quantile(wh[:,0],[.1,.5,.9]).tolist(),
                'bbox_height_q10_q50_q90':np.quantile(wh[:,1],[.1,.5,.9]).tolist(),
                'bbox_area_below_32_squared_fraction':float((area<32**2).mean()),
                'bbox_short_side_below_8_fraction':float((wh.min(1)<8).mean()),
                'truncated_fraction':sum(r[4] for r in rows)/len(rows),
                'optimistic_shape_IoU_fraction_at_least':{str(t):{
                    'current':float((base>=t).mean()),'add_8px_size':float((small>=t).mean()),
                    'add_quarter_four_aspect_ratios':float((tall>=t).mean())} for t in [.3,.5,.7]}}
        output[split]={'views':len(subset),'classes':classes}
    return {'status':status,'identity':identity,'scanned_views':len(records),'results':output,
            'scope':'GT labels only at original640x360; no model inference or heldout indexes. Shape IoU assumes exactly aligned centres and ignores feature-grid offsets, ranking, budget and regression.',
            'limits':'TorchVision RPN allows low-quality best matches, so an IoU below .7 does not mean a GT receives no positive anchor. Candidate shapes are untrained what-if geometry only; no active model/config/threshold changed.'}


def main(args):
    root,out=args.root.resolve(),args.output.resolve()
    if out.is_relative_to(root):raise ValueError('Write outside immutable dataset')
    audit=read(root/'audit.json');review=read(root/'visual_review.json')
    if audit['status']!='passed' or not review['training_approved'] or review['audit_sha256']!=sha(root/'audit.json'):raise ValueError('Bound data approval required')
    if sha(root/'metadata_inventory.json')!=audit['metadata_inventory_sha256']:raise ValueError('Metadata inventory changed')
    metadata={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
    config=read(args.config);sizes=config['detection']['anchor_sizes_pixels'];selected=[]
    for split in ['train','validation']:
        name=f'training/{split}.jsonl'
        if sha(root/name)!=metadata[name]:raise ValueError('Index changed')
        rows=[json.loads(line) for line in (root/name).read_text(encoding='utf-8').splitlines()]
        if any(r['split']!=split for r in rows):raise ValueError('Split mismatch')
        selected.extend(rows)
    identity={'tool_sha256':sha(Path(__file__)),'io_helper_sha256':sha(ROOT/'scripts/survey_sunny_materials.py'),
              'data_audit_sha256':sha(root/'audit.json'),'config_sha256':sha(args.config),
              'torchvision':torchvision.__version__,'current_anchor_sizes':sizes,'current_ratios':[.5,1.,2.],
              'selection_sha256':hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()}
    out.mkdir(parents=True,exist_ok=True);chunks=out/'chunks';chunks.mkdir(exist_ok=True);report_path=out/'report.json'
    if report_path.exists() and read(report_path)['identity']!=identity:raise ValueError('Resume identity differs')
    torch.set_num_threads(2);records=[]
    for start in range(0,len(selected),100):
        batch=selected[start:start+100];path=chunks/f'{start//100:04d}.json';cached=read(path) if path.exists() else None
        if cached and (cached['identity']!=identity or cached['sources']!=[r['detection'] for r in batch]):raise ValueError('Cached chunk differs')
        computed=[]
        for row in batch:
            source=safe_path(root,row['detection'])
            if sha(source)!=metadata[row['detection']]:raise ValueError('Audited GT label changed')
            if cached:continue
            labels=read(source);objects=[]
            for obj in labels['positive_objects']:
                x1,y1,x2,y2=obj['visible_bbox_xyxy'];w,h=x2-x1,y2-y1
                if not 0<=x1<x2<=640 or not 0<=y1<y2<=360 or obj['category_id'] not in DETECTION_CLASSES.values():raise ValueError('Invalid positive box')
                objects.append([obj['category_id'],w,h,obj['visible_pixels'],x1==0 or y1==0 or x2==640 or y2==360])
            computed.append({'split':row['split'],'group_id':row['group_id'],'source':row['detection'],'objects':objects})
        if cached:records.extend(cached['records'])
        else:
            write(path,{'identity':identity,'sources':[r['detection'] for r in batch],'records':computed});records.extend(computed)
        if len(records)%1000==0:
            write(report_path,summarize(records,identity,sizes,'in_progress'))
            print(json.dumps({'scanned_views':len(records),'total':len(selected)}),flush=True)
    report=summarize(records,identity,sizes,'completed_data_diagnostic')
    for split in ['train','validation']:
        for name,c in report['results'][split]['classes'].items():
            if c['objects']!=audit['visible_objects'][split][name]:raise ValueError('Object count differs from independent audit')
    write(report_path,report);print(json.dumps({'status':'completed_data_diagnostic','views':len(records)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
