"""Audited sunny data, bounded uint8 cache, separate RGB-only inference boundary."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .dataset import safe_path
from .instance_labels import decode_instance_rgb

DETECTION_CLASSES={'car':1,'pedestrian':2,'bicycle':3,'motorcycle':4,'truck':5,'bus':6,'traffic_light':7,'temporary_obstacle':8}
SEGMENTATION_NAMES=['other_annotated','road','sidewalk','pedestrian','rider','car','truck','bus','motorcycle','bicycle','road_line','traffic_light','wall','fence','pole','temporary_obstacle','terrain_or_ground']
# CARLA 0.9.16 rpc/ObjectLabel.h: raw4 Walls,5 Fences,6 Poles,11 Sky.
RAW_TO_SEG={1:1,2:2,12:3,13:4,14:5,15:6,16:7,18:8,19:9,24:10,7:11,4:12,5:13,6:14,10:16,25:16}


def png(path):
    with Image.open(path) as image:
        if image.format!='PNG' or image.mode!='RGB': raise ValueError('Expected raw RGB PNG')
        return np.array(image,dtype=np.uint8)


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    root=Path(root).resolve(); audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    if audit['status']!='passed' or not audit['automated_checks_passed']: raise ValueError('Independent audit/coverage must pass')
    review=json.loads((root/'visual_review.json').read_text(encoding='utf-8'))
    if review['status']!='passed' or not review['training_approved'] or review['audit_sha256']!=sha(root/'audit.json'): raise ValueError('Missing bound visual review')
    if not review.get('reviewed_images'): raise ValueError('Review must identify actual inspected images')
    for item in review['reviewed_images']:
        if sha(safe_path(root,item['path']))!=item['sha256']: raise ValueError('Reviewed QA image changed')
    for filename,key in [('image_inventory.jsonl','image_inventory_sha256'),('metadata_inventory.json','metadata_inventory_sha256')]:
        if sha(root/filename)!=audit[key]: raise ValueError('Audited inventory changed')
    for item in json.loads((root/'metadata_inventory.json').read_text(encoding='utf-8')):
        if sha(safe_path(root,item['path']))!=item['sha256']: raise ValueError('Audited metadata changed: '+item['path'])
    count=0
    for line in (root/'image_inventory.jsonl').read_text(encoding='utf-8').splitlines():
        item=json.loads(line); path=safe_path(root,item['path'])
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']: raise ValueError('Raw image changed: '+item['path'])
        count+=1
    if count!=audit['checked_images']: raise ValueError('Incomplete inventory')
    return {'audit_sha256':sha(root/'audit.json'),'review_sha256':sha(root/'visual_review.json'),
            'image_inventory_sha256':audit['image_inventory_sha256'],'metadata_inventory_sha256':audit['metadata_inventory_sha256'],'verified_images':count}


def segmentation_target(raw, instance, verified_obstacle_actor_ids):
    if raw.dtype!=np.uint8 or raw.max()>28: raise ValueError('Invalid semantic class')
    lut=np.zeros(29,dtype=np.uint8)
    for raw_id,model_id in RAW_TO_SEG.items(): lut[raw_id]=model_id
    lut[0]=255; lut[21]=255  # unknown Dynamic is not automatically an obstacle
    target=lut[raw]
    classes,ids=decode_instance_rgb(instance)
    verified=np.isin(ids,verified_obstacle_actor_ids)&(classes==21)&(raw==21)
    target[verified]=15
    return target


class SunnySamples(Dataset):
    def __init__(self,root,split,task,cache_budget_gib=16,allow_lockbox=False):
        self.root=Path(root).resolve(); self.task=task
        if split not in ['train','validation','development','lockbox'] or (split=='lockbox' and not allow_lockbox): raise ValueError('Lockbox unavailable during training/selection')
        self.samples=[json.loads(line) for line in (self.root/f'training/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        if any(s['split']!=split for s in self.samples): raise ValueError('Split boundary error')
        estimated=len(self.samples)*640*360*(4 if task=='segmentation' else 3)
        if estimated>cache_budget_gib*1024**3: raise MemoryError('Decoded cache exceeds explicit budget; use streaming')
        self.rgb=[]; self.targets=[]; self.support=np.zeros(len(SEGMENTATION_NAMES),dtype=np.int64)
        for s in self.samples:
            rgb=png(safe_path(self.root,s['rgb']))
            if rgb.dtype!=np.uint8 or rgb.shape!=(360,640,3): raise ValueError('RGB schema error')
            self.rgb.append(torch.from_numpy(rgb.transpose(2,0,1).copy()))
            if task=='segmentation':
                raw=png(safe_path(self.root,s['semantic']))[...,0]
                instance=png(safe_path(self.root,s['instance']))
                target=segmentation_target(raw,instance,s['verified_obstacle_actor_ids']); self.support+=np.bincount(target[target!=255],minlength=len(SEGMENTATION_NAMES))
                self.targets.append(torch.from_numpy(target))
            elif task=='detection':
                labels=json.loads(safe_path(self.root,s['detection']).read_text(encoding='utf-8'))
                self.targets.append({'boxes':torch.tensor([p['visible_bbox_xyxy'] for p in labels['positive_objects']],dtype=torch.float32).reshape(-1,4),
                    'labels':torch.tensor([p['category_id'] for p in labels['positive_objects']],dtype=torch.int64),
                    'light_state':torch.tensor([p.get('light_state_class',-1) for p in labels['positive_objects']],dtype=torch.int64),
                    'visible_pixels':torch.tensor([p['visible_pixels'] for p in labels['positive_objects']],dtype=torch.int64),
                    'truncated':torch.tensor([p['visible_bbox_xyxy'][0]==0 or p['visible_bbox_xyxy'][1]==0 or p['visible_bbox_xyxy'][2]==640 or p['visible_bbox_xyxy'][3]==360 for p in labels['positive_objects']],dtype=torch.bool),
                    'ignore_boxes':torch.tensor([r['bbox_xyxy'] for r in labels['ignore_regions']],dtype=torch.float32).reshape(-1,4)})
            else: raise ValueError('Unknown task')

    def __len__(self): return len(self.samples)

    def __getitem__(self,index):
        return self.rgb[index],self.targets[index],index
