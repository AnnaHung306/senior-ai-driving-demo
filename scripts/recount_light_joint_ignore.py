"""Correct legacy joint FP counting using saved phase decisions, CPU only.

Keeps original reports intact; proves GT confusion/recall unchanged and FP
count agrees with canonical detector evaluation before emitting a v2 report.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from evaluate_light_state import joint_scores,joint_quality,error_gallery
from senior_ai.light_joint_policy import ignored_unmatched_light,JOINT_PROTOCOL
from senior_ai.sunny_data import sha
from senior_ai.core import write_json


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main(args):
    out=args.output.resolve();phase=args.phase_run.resolve();det=args.detector_run.resolve();root=args.root.resolve()
    if out.exists() or any(out.is_relative_to(p) for p in [phase,det,root]):raise ValueError('New separate recount output required')
    ps,ds=read(phase/'state.json'),read(det/'state.json');receipt=read(phase/'validation_joint_quality.json');original=read(phase/'validation_joint.json')
    if ps['status']!='completed' or ds['status']!='completed' or sha(phase/'best.pt')!=ps['best_sha256'] or sha(det/'best.pt')!=ds['best_sha256']:
        raise ValueError('Completed intact model pair required')
    if receipt['phase_sha256']!=ps['best_sha256'] or receipt['detector_sha256']!=ds['best_sha256'] or receipt['joint_report_sha256']!=sha(phase/'validation_joint.json'):
        raise ValueError('Legacy report/model binding differs')
    if sha(det/'validation_frames.jsonl')!=ds['validation_frames_sha256'] or sha(det/'validation.json')!=ds['validation_sha256'] or sha(phase/'validation_frames.jsonl')!=ps['validation_frames_sha256']:
        raise ValueError('Saved validation changed')
    if ps['data']!=read(det/'provenance.json')['data'] or sha(root/'audit.json')!=ps['data']['audit_sha256'] or sha(root/'image_inventory.jsonl')!=ps['data']['image_inventory_sha256']:
        raise ValueError('Original shared data signature changed')
    frames=[json.loads(line) for line in (det/'validation_frames.jsonl').read_text().splitlines()]
    if any(r['split']!='validation' for r in frames):raise ValueError('Validation only')
    lookup={(f['scene_id'],f['camera'],f['frame']):f for f in frames}
    records=[json.loads(line) for line in (phase/'validation_joint_frames.jsonl').read_text().splitlines()]
    matrix=np.zeros((4,5),dtype=np.int64);old_fp=Counter();new_fp=Counter();corrected=[];ignored=[]
    for r in records:
        if r.get('ignored'):raise ValueError('This tool handles legacy v1 records only')
        f=lookup[(r['scene_id'],r['camera'],r['frame'])]
        if r['rgb']!=f['rgb']:raise ValueError('Record RGB identity differs')
        item=dict(r)
        if 'target' in r:matrix[r['target'],4 if r['prediction'] is None else r['prediction']]+=1
        else:
            if not any(p['label']==7 and p['score']>=.5 and p['box']==r['box'] for p in f['predictions']):raise ValueError('Legacy prediction not present in detector frame')
            old_fp[r['prediction']]+=1
            if ignored_unmatched_light(r['box'],f['targets'],f.get('ignore_boxes',[])):
                item.update(ignored=True,legacy_phase_prediction=r['prediction'],prediction=None);ignored.append(item)
            else:new_fp[r['prediction']]+=1
        corrected.append(item)
    before=joint_scores(matrix,[old_fp[c] for c in range(4)])
    if any(before[k]!=original[k] for k in before) or sum(old_fp.values())!=original['unmatched_detector_light_predictions']:
        raise ValueError('Legacy records do not reproduce original report exactly')
    result={**joint_scores(matrix,[new_fp[c] for c in range(4)]),'protocol':JOINT_PROTOCOL,
        'unmatched_detector_light_predictions':sum(new_fp.values()),'ignored_detector_light_predictions':len(ignored)}
    metric=read(det/'validation.json')['per_class']['traffic_light'];matched=int(matrix[:,:4].sum());missed=int(matrix[:,4].sum())
    if matched!=metric['true_positives'] or missed!=metric['false_negatives'] or sum(new_fp.values())!=metric['false_positives']:
        raise ValueError('Corrected matched/missed/FP counts disagree with detector AP50')
    if result['macro_correct_state_recall']!=original['macro_correct_state_recall'] or result['red_as_green_rate_matched_red']!=original['red_as_green_rate_matched_red']:
        raise ValueError('Ignore-only correction must not alter GT confusion or recall')
    out.mkdir();write_json(out/'legacy_joint.json',original);write_json(out/'joint_v2.json',result)
    (out/'joint_v2_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in corrected),encoding='utf-8')
    q=joint_quality(result,read(phase/'config.json'));write_json(out/'quality.json',q)
    write_json(out/'receipt.json',{'scope':__doc__,'tool_sha256':sha(Path(__file__)),'policy_sha256':sha(ROOT/'src/senior_ai/light_joint_policy.py'),
        'phase_sha256':ps['best_sha256'],'detector_sha256':ds['best_sha256'],'legacy_joint_sha256':sha(phase/'validation_joint.json'),
        'legacy_records_sha256':sha(phase/'validation_joint_frames.jsonl'),'detector_frames_sha256':ds['validation_frames_sha256'],
        'matrix_unchanged':True,'legacy_report_recomputed_exact':True,'corrected_FP_matches_detector':True,
        'old_unmatched':sum(old_fp.values()),'ignored_at_score_0_5':len(ignored),'new_unmatched':sum(new_fp.values()),'model_inference_performed':False,'heldout_opened':False})
    crop=[json.loads(line) for line in (phase/'validation_frames.jsonl').read_text().splitlines()]
    error_gallery(root,None,crop,corrected,out/'validation_phase_errors')
    print(json.dumps({'macro_recall_unchanged':result['macro_correct_state_recall'],'old_false_lights':sum(old_fp.values()),
        'ignored':len(ignored),'corrected_false_lights':sum(new_fp.values()),'quality':q}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase-run',type=Path,required=True);p.add_argument('--detector-run',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
