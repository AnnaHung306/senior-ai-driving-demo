"""Saved validation PR diagnostics; original score=.5 gates stay unchanged."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from senior_ai.sunny_data import sha,DETECTION_CLASSES
from senior_ai.detection_metrics import iou
from senior_ai.visibility_policy import ignore_background_proposals
from senior_ai.core import write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def analyze(run):
    state=read(run/'state.json');config=read(run/'config.json')
    if state['status']!='completed' or state['task']!='detection' or not state['checkpoint_reload_identical']:
        raise ValueError('Completed reloaded detector required')
    for name,key in [('best.pt','best_sha256'),('validation.json','validation_sha256'),('validation_frames.jsonl','validation_frames_sha256')]:
        if sha(run/name)!=state[key]:raise ValueError('Frozen detector/validation changed')
    baseline=read(run/'validation.json')
    if baseline['score_threshold']!=.5:raise ValueError('Expected original score=.5 operating point')
    frames=[json.loads(line) for line in (run/'validation_frames.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(frames)!=baseline['images'] or any(f['split']!='validation' for f in frames):raise ValueError('Saved validation only')
    grid=[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95];classes={}
    for name,label in DETECTION_CLASSES.items():
        gt={i:[t for t in f['targets'] if t['label']==label] for i,f in enumerate(frames)}
        support=sum(map(len,gt.values()));matched={i:set() for i in gt};ranked=[]
        for index,frame in enumerate(frames):
            for rank,p in enumerate(sorted(frame['predictions'],key=lambda p:-p['score'])[:100]):
                if not math.isfinite(p['score']) or not 0<=p['score']<=1:raise ValueError('Invalid score')
                if p['label']==label:ranked.append((p['score'],index,rank,p['box']))
        ranked.sort(key=lambda r:(-r[0],r[1],r[2]));scores=[];hits=[];ignored=0
        for score,index,rank,box in ranked:
            overlaps=iou(box,[t['box'] for t in gt[index]])
            eligible=[j for j,v in enumerate(overlaps) if v>=.5 and j not in matched[index]]
            if eligible:
                matched[index].add(max(eligible,key=lambda j:overlaps[j]));hit=True
            else:
                known=max(iou(box,[t['box'] for t in frames[index]['targets']]),default=0)
                is_ignore=ignore_background_proposals([box],[0],frames[index].get('ignore_boxes',[]))[0]==-1
                if known<.5 and is_ignore:ignored+=1;continue
                hit=False
            scores.append(score);hits.append(hit)
        scores=np.asarray(scores);hits=np.asarray(hits,dtype=np.int64);points=[]
        for threshold in grid:
            selected=scores>=threshold;tp=int(hits[selected].sum());fp=int(selected.sum())-tp
            p=tp/(tp+fp) if tp+fp else 0.;r=tp/support if support else None
            points.append({'score_threshold':threshold,'true_positives':tp,'false_positives':fp,'false_negatives':support-tp,
                'precision':p,'recall':r,'F1':2*p*(r or 0)/(p+(r or 0)) if p+(r or 0) else 0.})
        original=next(p for p in points if p['score_threshold']==.5);reference=baseline['per_class'][name]
        if support!=reference['ground_truth_objects'] or any(original[k]!=reference[k] for k in ['true_positives','false_positives','false_negatives']):
            raise ValueError('Score=.5 greedy/ignore accounting differs from canonical report: '+name)
        classes[name]={'ground_truth_objects':support,'ignored_predictions_all_scores':ignored,'points':points,
            'diagnostic_grid_has_P_R_at_least_half':any(p['precision']>=.5 and (p['recall'] or 0)>=.5 for p in points)}
    return {'status':'completed_saved_validation_diagnostic_only','checkpoint_sha256':state['best_sha256'],
        'validation_frames_sha256':state['validation_frames_sha256'],'validation_sha256':state['validation_sha256'],
        'tool_sha256':sha(Path(__file__)),'matching_metric_source_sha256':sha(ROOT/'src/senior_ai/detection_metrics.py'),
        'ignore_policy_source_sha256':sha(ROOT/'src/senior_ai/visibility_policy.py'),'original_quality_gates':config['quality_gates'],
        'images':len(frames),'groups':sorted({f['group_id'] for f in frames}),'fixed_diagnostic_thresholds':grid,'per_class':classes,
        'scope':'Existing validation predictions only, no new inference/training or heldout. Same max100, class-aware IoU50 greedy matches and positive-first ignore policy; exact TP/FP/FN agreement at original .5 required.',
        'limits':'Exploratory operating points are not adopted. Deployment/display/quality gates remain score=.5. Grid findings do not prove calibrated probabilities, new recognition ability, or unseen-test generalization.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.run.resolve()):raise ValueError('New output outside immutable run required')
    report=analyze(args.run.resolve());write_json(args.output,report)
    print(json.dumps({'images':report['images'],'classes_with_a_diagnostic_P_R_half_point':{k:v['diagnostic_grid_has_P_R_at_least_half'] for k,v in report['per_class'].items()},'original_score_threshold_unchanged':.5}))
