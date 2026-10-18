"""Independent exact-frame/pose/PNG check of a bounded static capture probe."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_segmentation_supplement import local_camera
from validate_dataset import read_png
from senior_ai.core import write_json
from senior_ai.sunny_data import sha
from senior_ai.dataset import safe_path
from senior_ai.segmentation_supplement import resolved_class_alignment


def main(args):
    root=args.root;plan=json.loads((root/'plan.json').read_text(encoding='utf-8'));rig=plan['rig'];town=args.town
    report=json.loads((root/town/'report.json').read_text(encoding='utf-8'));rows=json.loads((root/town/'records.json').read_text(encoding='utf-8'))
    if report['status']!='passed' or not report['cleanup'] or not report['probe_only'] or report['plan_sha256']!=sha(root/'plan.json'):raise ValueError('Completed bound probe required')
    for relative,digest in report['source_files'].items():
        if sha(root/town/'source_snapshot'/relative)!=digest:raise ValueError('Probe capture source changed')
    expected={(a['spawn_index'],c['name']):np.asarray(a['native_transform_matrix'])@local_camera(c)
        for a in plan['towns'][town]['anchors'][:args.anchors] for c in rig['cameras']}
    actual=[(r['anchor_spawn_index'],r['camera']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected) or len(rows)!=report['records']:raise ValueError('Probe matrix incomplete')
    max_pose=0.;max_disagreement=0.;files=[]
    for row in rows:
        if row['town']!=town or row['split']!='train' or row['task']!='segmentation_only':raise ValueError('Probe scope mismatch')
        if set(row['images'])!={'rgb','semantic','depth','instance'}:raise ValueError('Four modalities required')
        images={}
        for modality,item in row['images'].items():
            p=safe_path(root,item['path']);image,_,_=read_png(p,rig['width'],rig['height']);images[modality]=image
            if item['frame']!=row['frame'] or abs(item['timestamp_s']-row['timestamp_s'])>.0001:raise ValueError('Probe exact frame/time failed')
            delta=float(np.abs(np.asarray(item['T_world_from_camera'])-expected[(row['anchor_spawn_index'],row['camera'])]).max());max_pose=max(max_pose,delta)
            if delta>=.0001:raise ValueError('Probe actual world pose failed')
            files.append({'path':item['path'],'sha256':sha(p)})
        sem=images['semantic'];instance=images['instance']
        if sem[...,1:].any() or sem[...,0].max()>28:raise ValueError('Semantic encoding failed')
        disagreement=resolved_class_alignment(sem[...,0],instance[...,0])['resolved_disagreement'];max_disagreement=max(max_disagreement,disagreement)
        if disagreement>.02:raise ValueError('Semantic/instance alignment failed')
    result={'status':'passed_probe_only','training_approved':False,'scope':'Engineering probe only, not the full training supplement',
        'views':len(rows),'checked_images':len(files),'max_pose_error':max_pose,'max_semantic_instance_disagreement':max_disagreement,
        'rejected_capture_attempts':sum(len(r['rejected_capture_attempts']) for r in rows),'capture_report_sha256':sha(root/town/'report.json'),
        'plan_sha256':sha(root/'plan.json'),'verifier_sha256':sha(Path(__file__)),'files':files}
    write_json(root/'probe-verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files'}));return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--town',choices=['Town01','Town03'],required=True)
    p.add_argument('--anchors',type=int,required=True);raise SystemExit(main(p.parse_args()))
