"""Explicitly bound train-only scenery; never available to validation/heldout."""
import json
from pathlib import Path
import numpy as np
import torch
from .dataset import safe_path
from .sunny_data import png,sha,segmentation_target,SEGMENTATION_NAMES


def resolved_class_alignment(semantic,instance_class):
    """Compare nonzero annotated IDs; retain raw disagreement/missing support.

    Raw0 is unlabelled. Semantic9 vegetation remains annotated other0 even if
    the instance camera has no class on a leaf pixel; no target is changed.
    """
    resolved=(semantic!=0)&(instance_class!=0)
    return {'raw_disagreement':float((semantic!=instance_class).mean()),
        'resolved_disagreement':float(((semantic!=instance_class)&resolved).sum()/max(1,int(resolved.sum()))),
        'compared_pixels':int(resolved.sum()),'unlabelled_instance_pixels':int((instance_class==0).sum()),
        'unlabelled_semantic_pixels':int((semantic==0).sum())}


def verify_scenery(root,main_audit,plan_sha256):
    root=Path(root).resolve();audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    review=json.loads((root/'visual_review.json').read_text(encoding='utf-8'));plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    if audit['status']!='passed' or not all(audit['checks'].values()) or audit['source_audit_sha256']!=main_audit:
        raise ValueError('Scenery audit must pass and bind the main audit')
    if plan['task']!='segmentation_only' or plan['split']!='train' or set(plan['towns'])!={'Town01','Town03'} or plan['weather']!='ClearNoon':
        raise ValueError('Only original train towns, sunny segmentation scenery allowed')
    if sha(root/'plan.json')!=plan_sha256 or audit['plan_sha256']!=plan_sha256:raise ValueError('Scenery plan changed')
    if review['status']!='passed' or not review['training_approved'] or review['audit_sha256']!=sha(root/'audit.json') or not review['reviewed_images']:
        raise ValueError('Actual bound visual review required')
    for item in audit['files']+review['reviewed_images']:
        path=safe_path(root,item['path'])
        if sha(path)!=item['sha256'] or 'bytes' in item and path.stat().st_size!=item['bytes']:raise ValueError('Scenery bytes changed: '+item['path'])
    rows=[json.loads(line) for line in (root/'training/train.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=audit['views'] or len({(r['group_id'],r['camera']) for r in rows})!=len(rows):raise ValueError('Scenery row matrix mismatch')
    if any(r['split']!='train' or r['town'] not in {'Town01','Town03'} or r['verified_obstacle_actor_ids']!=[] for r in rows):raise ValueError('Scenery training boundary violation')
    return {'root':str(root),'audit_sha256':sha(root/'audit.json'),'review_sha256':sha(root/'visual_review.json'),
            'plan_sha256':plan_sha256,'views':len(rows),'groups':len({r['group_id'] for r in rows}),
            'segmentation_pixels':audit['segmentation_pixels'],'scope':plan['scope']}


def append_scenery(train,root,repeats,cache_budget_gib):
    if train.task!='segmentation' or any(r['split']!='train' for r in train.samples):raise ValueError('Scenery is training segmentation only')
    if type(repeats) is not int or not 1<=repeats<=8:raise ValueError('Explicit sampling repeats must be 1..8')
    root=Path(root).resolve();rows=[json.loads(line) for line in (root/'training/train.jsonl').read_text(encoding='utf-8').splitlines()]
    if (len(train)+len(rows)*repeats)*640*360*4>cache_budget_gib*1024**3:raise MemoryError('Combined cache exceeds explicit conservative budget')
    rgbs=[];targets=[];support=np.zeros(len(SEGMENTATION_NAMES),dtype=np.int64)
    for row in rows:
        if row['split']!='train' or row['town'] not in {'Town01','Town03'}:raise ValueError('Scenery split boundary violation')
        rgb=png(safe_path(root,row['rgb']));raw=png(safe_path(root,row['semantic']))[...,0];instance=png(safe_path(root,row['instance']))
        if rgb.shape!=(360,640,3) or raw.shape!=(360,640) or instance.shape!=rgb.shape:raise ValueError('Scenery rig shape differs')
        target=segmentation_target(raw,instance,[]);support+=np.bincount(target[target!=255],minlength=len(support))
        rgbs.append(torch.from_numpy(rgb.transpose(2,0,1).copy()));targets.append(torch.from_numpy(target))
    audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    if support.tolist()!=audit['segmentation_pixels']:raise ValueError('Loaded scenery class support differs from audit')
    # Repeated sampling uses references, preserving explicit unique view/group counts.
    train.rgb.extend(rgbs*repeats);train.targets.extend(targets*repeats)
    train.samples.extend([{**r,'data_root':str(root),'static_supplement':True} for r in rows]*repeats)
    train.support+=support*repeats
    return {'unique_views':len(rows),'sampling_repeats':repeats,'sampled_views_per_epoch':len(rows)*repeats,'unique_groups':len({r['group_id'] for r in rows})}
