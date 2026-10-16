"""Independent PNG/geometry/cadence/phase coverage audit for a controlled supplement."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
from audit_sunny_dataset import ROOT,lamp_targets,transform,sha
from validate_dataset import read_png
from senior_ai.core import write_json
from senior_ai.dataset import safe_path
from senior_ai.light_labels import LIGHT_STATES
from senior_ai.light_labels import phase_target


def verify_supplement(root,source_audit_sha256):
    root=Path(root).resolve();audit=json.loads((root/'audit.json').read_text(encoding='utf-8'));review=json.loads((root/'visual_review.json').read_text(encoding='utf-8'))
    if audit['status']!='passed' or audit['source_audit_sha256']!=source_audit_sha256:raise ValueError('Unapproved supplement audit')
    if review['status']!='passed' or review['audit_sha256']!=sha(root/'audit.json') or not review['reviewed_images']:raise ValueError('Missing supplement visual review')
    for item in audit['files']+review['reviewed_images']:
        if sha(safe_path(root,item['path']))!=item['sha256']:raise ValueError('Supplement bytes changed: '+item['path'])
    return {'audit_sha256':sha(root/'audit.json'),'review_sha256':sha(root/'visual_review.json')}


def audit(root):
    if (root/'audit.json').exists():raise FileExistsError('Immutable audit')
    report=json.loads((root/'report.json').read_text(encoding='utf-8'));plan=json.loads((root/'plan.json').read_text(encoding='utf-8'));rig=plan['rig']
    if report['status']!='passed' or not report['cleanup'] or report['plan_sha256']!=sha(root/'plan.json'):raise ValueError('Capture incomplete')
    records=json.loads((root/'records.json').read_text(encoding='utf-8'));lamps=json.loads((root/'lamps.json').read_text(encoding='utf-8'));cal=json.loads((root/'calibration.json').read_text(encoding='utf-8'))
    expected={(p['id'],phase) for p in plan['poses'] for phase in plan['phases']}
    if len(records)!=len(expected) or {(r['id'],r['phase']) for r in records}!=expected:raise ValueError('Matrix incomplete/duplicated')
    for attrs in cal['sensor_attributes'].values():
        if int(attrs['image_size_x'])!=rig['width'] or int(attrs['image_size_y'])!=rig['height'] or float(attrs['fov'])!=rig['horizontal_fov_deg']:raise ValueError('Sensor optics differ')
    counts=Counter();groups=defaultdict(set);rows=[];qa=[];max_pose=0.;raw_count=0
    poses={p['id']:p for p in plan['poses']};off={r['id']:r for r in records if r['phase']=='Off'}
    for row in records:
        images={};camera=None
        if any(row[k]!=poses[row['id']][k] for k in poses[row['id']]):raise ValueError('Record differs from frozen viewpoint')
        if not row['all_lights_frozen'] or not row['states'] or any(v!=row['phase'] for v in row['states'].values()):raise ValueError('Phase mismatch')
        for modality,item in row['images'].items():
            image,_,_=read_png(safe_path(root,item['path']),rig['width'],rig['height']);images[modality]=image;raw_count+=1
            if item['frame']!=row['frame'] or abs(item['timestamp_s']-row['timestamp_s'])>.0001:raise ValueError('Frame mismatch')
            actual=np.asarray(item['T_world_from_camera']);max_pose=max(max_pose,float(np.abs(actual-row['T_world_from_camera']).max()))
            if camera is not None and np.abs(camera-actual).max()>1e-5:raise ValueError('Modality pose mismatch')
            camera=actual
        if images['semantic'][...,1:].any() or images['semantic'][...,0].max()>28:raise ValueError('Invalid raw semantic')
        packed=images['depth'].astype(np.uint32);depth=(packed[...,0]+256*packed[...,1]+65536*packed[...,2])*(1000/16777215)
        positives,_=lamp_targets(images['semantic'][...,0],depth,{'K':cal['K'],'T_world_from_camera':camera},lamps,rgb=images['rgb'],states=row['states'])
        off_rgb,_,_=read_png(safe_path(root,off[row['id']]['images']['rgb']['path']),rig['width'],rig['height'])
        changed=(images['rgb'].astype(np.int16)-off_rgb.astype(np.int16)).max(2)>30
        for p in positives:
            if p['light_state_class']:
                b=p['visible_bbox_xyxy'];mask=np.zeros(changed.shape,dtype=bool);mask[b[1]:b[3],b[0]:b[2]]=True
                mask &= (images['semantic'][...,0]==7)&changed
                if phase_target(images['rgb'],mask,row['phase'])['light_state_class']!=p['light_state_class']:
                    raise ValueError('Active color lacks >=2 phase-compatible pixels brighter than paired Off by >30')
                p['paired_off_brightness_verified']=True
        relative=f'visible/{row["id"]}/{row["phase"]}.json';write_json(root/relative,{'positive_objects':positives,'ignore_regions':[]})
        rows.append({'scene_id':row['scene_id'],'group_id':row['group_id'],'split':'lockbox','camera':row['camera'],'frame':row['frame'],
                     'rgb':row['images']['rgb']['path'],'detection':relative,'controlled_phase':row['phase'],'source_pose':row['id']})
        for p in positives:counts[p['light_state_class']]+=1;groups[p['light_state_class']].add(row['group_id'])
        qa.extend((row,p) for p in positives)
    path=root/'training/lockbox.jsonl';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    # Identical viewpoints rendered under each state: inspect the largest lamp per route.
    for group in sorted({r['group_id'] for r in records}):
        by_pose=defaultdict(list)
        for r,p in qa:
            if r['group_id']==group:by_pose[r['id']].append((r,p))
        if not by_pose:raise ValueError('No lamps in group')
        key=max(by_pose,key=lambda k:max(p['visible_pixels'] for r,p in by_pose[k]))
        canvas=Image.new('RGB',(640,384));d=ImageDraw.Draw(canvas)
        for j,phase in enumerate(plan['phases']):
            candidates=[(r,p) for r,p in by_pose[key] if r['phase']==phase]
            r,p=max(candidates,key=lambda x:x[1]['visible_pixels']);im=Image.open(root/r['images']['rgb']['path']).crop(tuple(p['visible_bbox_xyxy']));im.thumbnail((152,320));scale=min(152/im.width,320/im.height);im=im.resize((int(im.width*scale),int(im.height*scale)),Image.Resampling.NEAREST)
            canvas.paste(im,(j*160,48));d.text((j*160+2,2),phase,fill='white');d.text((j*160+2,18),LIGHT_STATES[p['light_state_class']],fill='white')
        p=root/f'qa/{group}.png';p.parent.mkdir(exist_ok=True);canvas.save(p)
    checks={'raw_image_count':raw_count==len(records)*3,'pose_alignment':max_pose<.0001,
        **{LIGHT_STATES[c]:counts[c]>=plan['coverage']['minimum_samples_per_class'] and len(groups[c])>=plan['coverage']['minimum_groups_per_class'] for c in range(4)}}
    files=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['audit.json','visual_review.json']]
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'counts':{LIGHT_STATES[c]:counts[c] for c in range(4)},
        'groups':{LIGHT_STATES[c]:len(groups[c]) for c in range(4)},'source_audit_sha256':plan['source_audit_sha256'],'max_pose_error':max_pose,'checked_images':raw_count,
        'auditor_sha256':sha(Path(__file__)),'lamp_geometry_source_sha256':sha(ROOT/'scripts/audit_sunny_dataset.py'),'files':files,
        'limits':plan['static_scope_limits'],'training_approved':False,'phase_evaluation_only':True}
    write_json(root/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);sys.exit(audit(p.parse_args().root))
