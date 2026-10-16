"""Independent raw-image audit and versioned visible-box export for sunny v3."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from collect_dataset import ROOT, transform, sha
from validate_dataset import read_png
from senior_ai.core import write_json
from senior_ai.dataset import read_sensor_frame, safe_path, validate_calibration
from senior_ai.instance_labels import decode_instance_rgb, validate_actor_ids, visible_candidate, project_box, mask_bbox
from senior_ai.visibility_policy import export_view
from senior_ai.light_labels import phase_target,LIGHT_STATES

CLASSES={'car':1,'pedestrian':2,'bicycle':3,'motorcycle':4,'truck':5,'bus':6,'traffic_light':7,'temporary_obstacle':8}
FOCUS=[1,2,4,5,6,10,12,13,14,15,16,18,19,24,7]


def lamp_targets(semantic, depth, camera, lamps, min_pixels=32,rgb=None,states=None):
    """Associate visible semantic-7 pixels to verified world-space lamp OBBs.

    A signal trigger volume and its actor semantic tag are never a lamp box.
    Simulator phase is retained as GT metadata, not trained as visible state.
    """
    claimed=np.zeros(semantic.shape,dtype=bool); positive=[]
    y,x=np.nonzero(semantic==7)
    if not len(x): return positive,claimed
    k=np.asarray(camera['K']); rays=np.stack((np.ones_like(x),(x+.5-k[0,2])/k[0,0],-(y+.5-k[1,2])/k[1,1]),1)
    world=(np.asarray(camera['T_world_from_camera']) @ np.c_[rays*depth[y,x,None],np.ones(len(x))].T).T
    for lamp in lamps:
        projection=project_box(lamp['world_vertices_m'],camera['T_world_from_camera'],k,semantic.shape[1],semantic.shape[0])
        if projection['bbox_xyxy'] is None or projection['near_plane_clipped']: continue
        box_world=np.asarray(transform(lamp['world_pose']).get_matrix())
        local=(np.linalg.inv(box_world) @ world.T).T[:,:3]
        inside=(np.abs(local)<=np.asarray(lamp['extent_m'])+.15).all(1)
        mask=np.zeros_like(claimed); mask[y[inside],x[inside]]=True
        if (mask & claimed).any(): raise ValueError('Ambiguous overlapping lamp OBB association')
        b=mask_bbox(mask)
        if b is None or int(mask.sum())<min_pixels or min(b[2]-b[0],b[3]-b[1])<4: continue
        # The observed mask must also agree with the projected geometric envelope.
        p=projection['bbox_xyxy']
        if b[0]<p[0]-2 or b[1]<p[1]-2 or b[2]>p[2]+2 or b[3]>p[3]+2: continue
        claimed |= mask
        positive.append({'kind':'traffic_light','category_id':7,'visible_bbox_xyxy':b,'visible_pixels':int(mask.sum()),
            'lamp_actor_id':lamp['actor_id'],'lamp_index':lamp['lamp_index'],'source':'semantic7_axial_depth_in_verified_lamp_world_OBB',
            'state_label':'unassigned; lamp facing and emissive visibility not verified','status':'positive'})
        if rgb is not None and states is not None:
            positive[-1].update(phase_target(rgb,mask,states[str(lamp['actor_id'])]))
            positive[-1]['state_label']=LIGHT_STATES[positive[-1]['light_state_class']]
    return positive,claimed


def audit(root):
    acquisition=json.loads((root/'report.json').read_text(encoding='utf-8'))
    if acquisition['task']!='sunny-cross-town-collect-v3' or acquisition['status']!='passed': raise ValueError('Complete acquisition required')
    if (root/'audit.json').exists(): raise FileExistsError('Audit immutable; use a new version')
    plan,rig,criteria=[json.loads((root/(n+'.json')).read_text(encoding='utf-8')) for n in ['plan','rig','criteria']]
    policy=json.loads((ROOT/'configs/visibility_policy_v1_1.json').read_text(encoding='utf-8'))
    policy={**policy,'version':'sunny-visible-policy-v3.1','class_ids':CLASSES}
    write_json(root/'visibility_policy.json',policy)
    write_json(root/'light_state_protocol.json',json.loads((ROOT/'configs/light_state_v1.json').read_text(encoding='utf-8')))
    scenes={s['id']:s for s in plan['scenes']}
    if len(scenes)!=18 or any(s['weather']!='ClearNoon' for s in scenes.values()): raise ValueError('Frozen scenario matrix changed')
    towns={split:{s['town'] for s in scenes.values() if s['split']==split} for split in ['train','validation','development','lockbox']}
    if any(towns[a]&towns[b] for a in towns for b in towns if a!=b): raise ValueError('Town leakage')
    inventory=[]; support=defaultdict(Counter); positive_views=defaultdict(Counter); groups=defaultdict(lambda:defaultdict(set))
    semantic_pixels=defaultdict(Counter); semantic_views=defaultdict(Counter); rows=defaultdict(list); seen=set(); rgb_splits={}; qa_rows=[]
    phase_counts=defaultdict(Counter); phase_groups=defaultdict(lambda:defaultdict(set))
    verified_obstacle_pixels=Counter();verified_obstacle_views=Counter()
    mismatch=0; total_pixels=0; max_pose=0.; max_translation=0.; max_rotation=0.
    for scene in scenes.values():
        folder=root/'scenes'/scene['id']; report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if report['status']!='passed' or not all(report['checks'].values()): raise ValueError('Scene acceptance failed')
        indexes=json.loads((folder/'index.json').read_text(encoding='utf-8'))
        if len(indexes)!=150 or report['motion']['label_duration_s']<29.5: raise ValueError('Long episode incomplete')
        cal=json.loads((root/f'calibration/{scene["id"]}.json').read_text(encoding='utf-8')); validate_calibration(cal,rig)
        cameras={c['name']:c for c in cal['cameras']}; lamps=[]
        for camera in cameras.values():
            if set(camera.get('sensor_attributes',{}))!=set(criteria['modalities']): raise ValueError('Missing actual sensor attributes')
            for attributes in camera['sensor_attributes'].values():
                if int(attributes['image_size_x'])!=rig['width'] or int(attributes['image_size_y'])!=rig['height'] or abs(float(attributes['fov'])-rig['horizontal_fov_deg'])>1e-5:
                    raise ValueError('Actual sensor optics/dimensions differ from calibration')
        light_path=root/f'labels/traffic_lights/{scene["id"]}.json'
        if light_path.exists():
            for a in json.loads(light_path.read_text(encoding='utf-8')):
                lamps.extend([{**b,'actor_id':a['actor_id'],'lamp_index':i} for i,b in enumerate(a['lamp_boxes_world'])])
        previous=None; known_ids=None
        for entry in indexes:
            if entry['split']!=scene['split'] or entry['group_id']!=scene['group_id']: raise ValueError('Split/group mismatch')
            feature=read_sensor_frame(root,entry['input']); label=json.loads(safe_path(root,entry['label']).read_text(encoding='utf-8'))
            if feature['frame']!=label['frame'] or abs(feature['timestamp_s']-label['timestamp_s'])>1e-4: raise ValueError('GT/feature mismatch')
            if previous and (feature['frame']-previous['frame']!=4 or abs(feature['timestamp_s']-previous['timestamp_s']-.2)>1e-4): raise ValueError('Cadence error')
            previous=feature; validate_actor_ids(label['objects_3d'])
            ids_now={a['actor_id'] for a in label['objects_3d']}
            if known_ids is not None and ids_now!=known_ids: raise ValueError('Actor identity changed')
            known_ids=ids_now
            if Counter((a['blueprint'],a['kind']) for a in label['objects_3d'])!=Counter((a['blueprint'],a['kind']) for a in scene['participants']): raise ValueError('Taxonomy inventory changed')
            objects=[]
            for a in label['objects_3d']:
                if a['kind']=='temporary_obstacle' and a['blueprint']!='static.prop.trafficcone01': raise ValueError('Unverified obstacle asset')
                objects.append({**a,'T_world_from_box':(np.asarray(transform(a['world_pose']).get_matrix()) @ np.asarray(transform(a['bbox_local_pose']).get_matrix())).tolist()})
            ego=np.asarray(transform(label['ego_world_pose']).get_matrix())
            for name,camera in cameras.items():
                poses=label['capture_world_from_camera'][name]; actual=np.asarray(poses['rgb'])
                max_pose=max(max_pose,max(float(np.abs(actual-poses[m]).max()) for m in criteria['modalities']))
                expected=ego @ np.asarray(camera['T_ego_from_camera_local'])
                max_translation=max(max_translation,float(np.linalg.norm(actual[:3,3]-expected[:3,3])))
                max_rotation=max(max_rotation,float(np.abs(actual[:3,:3]-expected[:3,:3]).max()))
                images={}; paths={}
                for modality in criteria['modalities']:
                    src=feature['cameras'][name] if modality=='rgb' else label[modality][name]
                    if src['frame']!=feature['frame'] or abs(src['timestamp_s']-feature['timestamp_s'])>1e-4: raise ValueError('Sensor alignment error')
                    relative=src['image']; path=safe_path(root,relative)
                    prefix='inputs/rgb' if modality=='rgb' else 'labels/'+modality
                    if not path.is_relative_to((root/prefix).resolve()) or relative in seen: raise ValueError('Boundary/duplicate image')
                    seen.add(relative); image,digest,size=read_png(path,rig['width'],rig['height']); images[modality]=image; paths[modality]=relative
                    inventory.append({'path':relative,'sha256':digest,'bytes':size})
                    if modality=='rgb':
                        if digest in rgb_splits and rgb_splits[digest]!=scene['split']: raise ValueError('Exact RGB split leakage')
                        rgb_splits[digest]=scene['split']
                classes,ids=decode_instance_rgb(images['instance_segmentation']); semantic=images['semantic_segmentation'][...,0]
                if images['semantic_segmentation'][...,1:].any() or semantic.max()>28: raise ValueError('Raw semantic encoding invalid')
                mismatch+=int((classes!=semantic).sum()); total_pixels+=semantic.size
                packed=images['depth'].astype(np.uint32); depth=(packed[...,0]+256*packed[...,1]+65536*packed[...,2])*(1000/16777215)
                geometry={'K':camera['K'],'T_world_from_camera':actual}
                candidates=[visible_candidate(a,classes,ids,semantic,depth,geometry) for a in objects]
                verified_obstacle_ids=[c['actor_id'] for c in candidates if c['kind']=='temporary_obstacle' and c['visible_pixels']
                    and not c['unexpected_semantic_tags'] and c['depth_geometry']['axial_in_box_fraction']>=policy['min_axial_depth_in_box_fraction']
                    and c['instance_semantic_disagreement_pixels']/c['visible_pixels']<=policy['max_instance_semantic_disagreement_fraction']]
                verified_mask=np.isin(ids,verified_obstacle_ids)&(classes==21)&(semantic==21)
                verified_obstacle_pixels[scene['split']]+=int(verified_mask.sum());verified_obstacle_views[scene['split']]+=int(verified_mask.any())
                light_positives,claimed=lamp_targets(semantic,depth,geometry,lamps,rgb=images['rgb'],states=label.get('traffic_light_states',{}))
                for light in light_positives:
                    phase_counts[scene['split']][light['light_state_class']]+=1
                    phase_groups[scene['split']][light['light_state_class']].add(scene['group_id'])
                export,ignore=export_view(candidates,np.where(claimed,0,classes),ids,policy)
                export['positive_objects'].extend(light_positives)
                relative=f'labels/visible/{scene["id"]}/{name}/{feature["frame"]:08d}.json'
                write_json(root/relative,export)
                names=Counter(c['kind'] for c in export['positive_objects']); split=scene['split']
                support[split].update(names); positive_views[split].update(names.keys())
                for kind in names: groups[split][kind].add(scene['group_id'])
                tags,counts=np.unique(semantic,return_counts=True); semantic_pixels[split].update({int(k):int(v) for k,v in zip(tags,counts)}); semantic_views[split].update(int(t) for t in tags)
                row={'scene_id':scene['id'],'group_id':scene['group_id'],'split':split,'camera':name,'frame':feature['frame'],
                     'input_manifest':entry['input'],'rgb':paths['rgb'],'semantic':paths['semantic_segmentation'],'detection':relative,
                     'instance':paths['instance_segmentation'],
                     'verified_obstacle_actor_ids':verified_obstacle_ids}
                rows[split].append(row)
                if entry==indexes[len(indexes)//2] and names:
                    panel=Image.fromarray(images['rgb']); draw=ImageDraw.Draw(panel)
                    for obj in export['positive_objects']:
                        b=obj['visible_bbox_xyxy']; draw.rectangle((b[0],b[1],b[2]-1,b[3]-1),outline='yellow',width=2); draw.text((b[0],max(0,b[1]-12)),obj['kind'],fill='yellow')
                    q=root/f'qa-visible/{scene["id"]}-{name}.png'; q.parent.mkdir(parents=True,exist_ok=True); panel.save(q); qa_rows.append(str(q.relative_to(root)))
        print(json.dumps({'audited_scene':scene['id'],'images':len(inventory),'positive_objects':dict(support[scene['split']])}),flush=True)
    quotas=plan['coverage_quotas']; checks={'images_86400':len(inventory)==86400,'pose_modalities':max_pose<1e-5,
        'pose_snapshot_translation':max_translation<.02,'pose_snapshot_rotation':max_rotation<1e-4,'semantic_parity':mismatch/total_pixels<.005}
    for split in towns:
        for kind in CLASSES:
            checks[f'{split}/{kind}/positive_views']=positive_views[split][kind]>=quotas['per_object_class_positive_views']
            checks[f'{split}/{kind}/positive_groups']=len(groups[split][kind])>=quotas['per_object_class_positive_groups']
        for tag in FOCUS:
            checks[f'{split}/semantic-{tag}']=semantic_pixels[split][tag]>=quotas['per_semantic_focus_class_pixels'] and semantic_views[split][tag]>=quotas['per_semantic_focus_class_positive_views']
        checks[f'{split}/verified_obstacle_semantic']=verified_obstacle_pixels[split]>=quotas['per_semantic_focus_class_pixels'] and verified_obstacle_views[split]>=quotas['per_semantic_focus_class_positive_views']
        path=root/f'training/{split}.jsonl'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(''.join(json.dumps(r)+'\n' for r in rows[split]),encoding='utf-8')
    inventory_path=root/'image_inventory.jsonl'; inventory_path.write_text(''.join(json.dumps(r)+'\n' for r in inventory),encoding='utf-8')
    metadata=[{'path':str(p.relative_to(root)).replace('\\','/'),'sha256':sha(p)} for p in sorted(root.rglob('*.json')) if p.name!='audit.json']
    metadata.extend({'path':str(p.relative_to(root)).replace('\\','/'),'sha256':sha(p)} for p in sorted((root/'training').glob('*.jsonl')))
    write_json(root/'metadata_inventory.json',metadata)
    result={'task':'sunny-independent-audit-v3','status':'passed' if all(checks.values()) else 'failed',
        'training_approved':False,'automated_checks_passed':all(checks.values()),'checks':checks,'checked_images':len(inventory),'image_inventory_sha256':sha(inventory_path),
        'metadata_inventory_sha256':sha(root/'metadata_inventory.json'),'source_sha256':sha(Path(__file__)),
        'geometry_sha256':sha(ROOT/'src/senior_ai/instance_labels.py'),'visible_objects':dict(support),
        'light_label_sha256':sha(ROOT/'src/senior_ai/light_labels.py'),
        'verified_obstacle_segmentation_pixels':dict(verified_obstacle_pixels),'verified_obstacle_segmentation_views':dict(verified_obstacle_views),
        'positive_views':dict(positive_views),'positive_group_counts':{s:{k:len(v) for k,v in d.items()} for s,d in groups.items()},
        'semantic_disagreement_fraction':mismatch/total_pixels,'max_modality_pose_difference':max_pose,
        'light_state_crop_counts':{s:{LIGHT_STATES[k]:v for k,v in values.items()} for s,values in phase_counts.items()},
        'light_state_positive_groups':{s:{LIGHT_STATES[k]:len(v) for k,v in values.items()} for s,values in phase_groups.items()},
        'max_capture_snapshot_translation_m':max_translation,'max_capture_snapshot_rotation_difference':max_rotation,
        'qa_images':qa_rows,'visual_review':'pending; training requires separate assistant review receipt'}
    write_json(root/'audit.json',result); print(json.dumps({k:v for k,v in result.items() if k not in ['checks','qa_images']}),flush=True)
    return 0 if all(checks.values()) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); sys.exit(audit(p.parse_args().root.resolve()))
