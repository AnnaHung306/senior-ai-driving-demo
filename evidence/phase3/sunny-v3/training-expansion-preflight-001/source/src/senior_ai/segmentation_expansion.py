"""Audited reserved Town06/07 segmentation training; no heldout or RGB inference API."""
import json
from pathlib import Path
import re
import numpy as np
import torch
from .dataset import safe_path
from .sunny_data import sha,png,segmentation_target,SEGMENTATION_NAMES


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def verify_expansion(root,main_root,plan_sha256,audit_sha256,review_sha256):
    root=Path(root).resolve();main_root=Path(main_root).resolve()
    for name,digest in [('plan.json',plan_sha256),('audit.json',audit_sha256),('visual_review.json',review_sha256)]:
        if sha(root/name)!=digest:raise ValueError('Frozen expansion '+name+' changed')
    plan=read(root/'plan.json');audit=read(root/'audit.json');review=read(root/'visual_review.json');main=read(main_root/'plan.json')
    if plan.get('version')!='sunny-static-segmentation-expansion-v2' or plan['task']!='segmentation_only' or plan['split']!='train' or plan['weather']!='ClearNoon' or plan['training_approved'] is not False:
        raise ValueError('Explicit train-only sunny expansion required')
    if plan['source_audit_sha256']!=sha(main_root/'audit.json') or plan['source_plan_sha256']!=sha(main_root/'plan.json') or plan['rig']!=read(main_root/'rig.json'):
        raise ValueError('Expansion main dataset/rig binding differs')
    if audit['status']!='passed' or not audit['checks'] or not all(audit['checks'].values()) or audit['source_audit_sha256']!=plan['source_audit_sha256'] or audit['plan_sha256']!=plan_sha256 or audit['mapping_source_sha256']!=sha(Path(__file__).with_name('sunny_data.py')):
        raise ValueError('Expansion audit/mapping differs')
    if review['status']!='passed' or review['training_approved'] is not True or review['audit_sha256']!=audit_sha256 or not review['reviewed_images']:
        raise ValueError('Bound actual expansion visual review required')
    towns=set(plan['towns'])
    if not towns or not towns.issubset({'Town06','Town07'}):raise ValueError('Only reserved unused full towns allowed')
    used_hashes={v['opendrive_sha256'] for v in main['maps'].values()}
    used_families={re.sub(r'_opt$','',s['town'].split('/')[-1],flags=re.I).casefold() for s in main['scenes']}
    expected=set()
    for town,entry in plan['towns'].items():
        if town.casefold() in used_families or entry['opendrive_sha256'] in used_hashes:raise ValueError('Original data split/map aliases excluded')
        policy=read(root/'native_inventories'/town/'policy.json')
        if sha(root/'native_inventories'/town/'policy.json')!=plan['policy_sha256'] or town not in policy['candidate_training_towns'] or policy['source_audit_sha256']!=plan['source_audit_sha256'] or policy['source_plan_sha256']!=plan['source_plan_sha256']:
            raise ValueError('Reserved training policy changed')
        report=read(root/town/'report.json')
        if report['status']!='passed' or not report['cleanup'] or report.get('probe_only') is not False or report.get('explicit_success_readback_frames') is not True or report['plan_sha256']!=plan_sha256:
            raise ValueError('Complete formal acquisition required; probes excluded')
        capture=read(root/town/'records.json')
        if len(capture)!=report['records'] or any(r.get('readback_frame')!=r['frame']+1 for r in capture):raise ValueError('Explicit captured F+1 records differ')
        for anchor in entry['anchors']:
            for camera in plan['rig']['cameras']:expected.add((town,f'{town.lower()}-static-anchor-{anchor["spawn_index"]:03}',camera['name']))
    inventory={r['path']:r for r in audit['files']}
    if len(inventory)!=len(audit['files']) or not {'plan.json','training/train.jsonl','qa/catalog.json'}.issubset(inventory):raise ValueError('Complete unique audited inventory required')
    for item in audit['files']+review['reviewed_images']:
        path=safe_path(root,item['path'])
        if sha(path)!=item['sha256'] or 'bytes' in item and path.stat().st_size!=item['bytes']:raise ValueError('Expansion bytes changed: '+item['path'])
    rows=[json.loads(line) for line in (root/'training/train.jsonl').read_text(encoding='utf-8').splitlines()]
    keys=[(r['town'],r['group_id'],r['camera']) for r in rows]
    if len(rows)!=audit['views'] or len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('Full frozen anchor/camera matrix required')
    for row in rows:
        if row['split']!='train' or row['scene_id']!=row['group_id'] or row['verified_obstacle_actor_ids']!=[]:raise ValueError('Training scope differs')
        for modality in ['rgb','semantic','instance','depth']:
            if row[modality] not in inventory:raise ValueError('Training image absent from audited inventory')
    if audit['groups']!=len({r['group_id'] for r in rows}):raise ValueError('Group count differs')
    return {'root':str(root),'plan_sha256':plan_sha256,'audit_sha256':audit_sha256,'review_sha256':review_sha256,
        'views':len(rows),'groups':audit['groups'],'segmentation_pixels':audit['segmentation_pixels'],'scope':plan['scope']}


def append_expansion(train,root,repeats,cache_budget_gib):
    if train.task!='segmentation' or any(r['split']!='train' for r in train.samples):raise ValueError('Expansion is segmentation training only')
    if type(repeats) is not int or not 1<=repeats<=8:raise ValueError('Explicit sampling repeats must be1..8')
    root=Path(root).resolve();rows=[json.loads(line) for line in (root/'training/train.jsonl').read_text(encoding='utf-8').splitlines()]
    if (len(train)+len(rows)*repeats)*640*360*4>cache_budget_gib*1024**3:raise MemoryError('Combined explicit cache budget exceeded')
    rgbs=[];targets=[];support=np.zeros(len(SEGMENTATION_NAMES),dtype=np.int64)
    for row in rows:
        if row['split']!='train' or row['town'] not in {'Town06','Town07'} or row['verified_obstacle_actor_ids']!=[]:raise ValueError('Expansion split boundary violation')
        rgb=png(safe_path(root,row['rgb']));raw=png(safe_path(root,row['semantic']))[...,0];instance=png(safe_path(root,row['instance']))
        if rgb.shape!=(360,640,3) or raw.shape!=(360,640) or instance.shape!=rgb.shape:raise ValueError('Expansion rig shape differs')
        target=segmentation_target(raw,instance,[]);support+=np.bincount(target[target!=255],minlength=len(support))
        rgbs.append(torch.from_numpy(rgb.transpose(2,0,1).copy()));targets.append(torch.from_numpy(target))
    if support.tolist()!=read(root/'audit.json')['segmentation_pixels']:raise ValueError('Loaded expansion support differs')
    train.rgb.extend(rgbs*repeats);train.targets.extend(targets*repeats)
    train.samples.extend([{**r,'data_root':str(root),'static_expansion':True} for r in rows]*repeats);train.support+=support*repeats
    return {'unique_views':len(rows),'sampling_repeats':repeats,'sampled_views_per_epoch':len(rows)*repeats,'unique_groups':len({r['group_id'] for r in rows})}
