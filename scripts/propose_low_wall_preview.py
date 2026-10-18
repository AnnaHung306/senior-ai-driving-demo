"""Offline train-town camera shortlist; preview must precede a formal data plan."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.sunny_data import sha
from senior_ai.core import write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    if args.output.exists():raise FileExistsError('Preserve old candidate versions')
    parent=read(args.parent_plan);plan=copy.deepcopy(parent);summary={}
    if parent['source_audit_sha256']!=sha(args.root/'audit.json') or set(parent['towns'])!={'Town01','Town03'}:
        raise ValueError('Bound original training towns required')
    scouts={}
    for scout in args.scout:
        report=read(scout/'report.json');town=report['town']
        if report['status']!='completed_metadata_only' or town not in {'Town01','Town03'} or town in scouts or report['source_audit_sha256']!=parent['source_audit_sha256']:
            raise ValueError('Distinct bound training-town inventory required')
        if report['opendrive_sha256']!=parent['towns'][town]['opendrive_sha256']:raise ValueError('Training map differs')
        for name in ['walls.json','road_spawn_points.json']:
            if sha(scout/name)!=report['files'][name]['sha256']:raise ValueError('Native inventory changed')
        scouts[town]=scout
    if set(scouts)!={'Town01','Town03'}:raise ValueError('Both training towns required')
    for town,scout in scouts.items():
        spawns=read(scout/'road_spawn_points.json');walls=read(scout/'walls.json')
        positions=np.asarray([r['native_transform_matrix'] for r in spawns])[:,:2,3]
        old=parent['towns'][town]['anchors'];old_indices={r['spawn_index'] for r in old};hints={}
        for wall in walls:
            if wall['native_semantic_type']!='Walls':raise ValueError('Expected native Walls inventory')
            box=wall['bounding_box_as_returned'];height=2*box['extent_m'][2]
            if not .4<=height<=2.5 or max(abs(box['rotation_deg'][k]) for k in ['pitch','roll'])>5:continue
            origin=np.asarray(wall['native_transform_matrix'])[:2,3]
            distances=np.linalg.norm(positions-origin,axis=1)
            # Native transform origin is world-space. Do not assume bbox centre.
            eligible=np.flatnonzero((distances>=6)&(distances<=30))
            if not len(eligible):continue
            i=int(min(eligible,key=lambda i:distances[i]));index=spawns[i]['spawn_index']
            if index in old_indices:continue
            hints.setdefault(i,[]).append({'native_id':wall['native_id'],'native_name':wall['native_name'],
                'reported_bbox_height_m_hint':height,'distance_to_native_origin_m':float(distances[i])})
        chosen=[];existing=np.asarray([r['native_transform_matrix'] for r in old])[:,:2,3]
        while len(chosen)<args.additional:
            candidates=[i for i in hints if i not in chosen and np.linalg.norm(existing-positions[i],axis=1).min()>=8]
            if not candidates:break
            # Cover spatially separated hints; not model scores or material names.
            i=max(candidates,key=lambda i:(float(np.linalg.norm(existing-positions[i],axis=1).min()),len(hints[i]),-i))
            chosen.append(i);existing=np.concatenate([existing,positions[i:i+1]],axis=0)
        added=[{**spawns[i],'selection':{'native_semantic_type':'Walls','geometry_hints_not_material_GT':hints[i],
            'purpose':'actual RGB/semantic preview of low-height wall candidates, not confirmed material coverage'}} for i in chosen]
        plan['towns'][town]['anchors']=added+old
        summary[town]={'old_anchors':len(old),'candidate_new_anchors':len(added),'eligible_new_spawns':len(hints),
            'new_anchor_spawn_indices':[r['spawn_index'] for r in added]}
    plan.update(version='sunny-low-wall-candidate-preview-v1',training_approved=False,requires_actual_RGB_preview=True,
        parent_plan_sha256=sha(args.parent_plan),preparer_sha256=sha(Path(__file__)),minimum_anchor_separation_m=8,
        selection='New native Walls geometry hints first, then all original frozen anchors. Not material labels. Original8 cameras, no pose fabrication.',
        scope='Candidate PREVIEW ONLY until actual RGB/GT is inspected and a distinct formal capture plan frozen. No approved training data or independent dynamic episodes.',
        candidate_summary=summary)
    write_json(args.output,plan)
    print(json.dumps({'candidate_plan_sha256':sha(args.output),'training_approved':False,'towns':summary}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent-plan',type=Path,required=True);p.add_argument('--scout',type=Path,nargs='+',required=True);p.add_argument('--additional',type=int,default=24);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not 1<=args.additional<=32:raise ValueError('Bound candidate count to1..32 per training town')
    main(args)
