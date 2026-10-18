"""Independently audit reserved unused-town scenery bytes; original auditor unchanged."""
import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from validate_dataset import read_png
from collect_dataset import ROOT
from senior_ai.core import write_json
from senior_ai.dataset import safe_path
from senior_ai.sunny_data import sha,segmentation_target,SEGMENTATION_NAMES
from predict_sunny import PALETTE
from senior_ai.segmentation_supplement import resolved_class_alignment
from prepare_unused_scenery import validate_expansion_plan


def local_camera(camera):
    angle=math.radians(camera['yaw_deg']);c,s=math.cos(angle),math.sin(angle)
    matrix=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],dtype=float)
    matrix[:3,3]=camera['position_m'];return matrix


def audit(root):
    if (root/'audit.json').exists():raise FileExistsError('Immutable audit: use new data version')
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'));rig=plan['rig']
    source=ROOT/'data/sunny-cross-town-v3-007'
    if sha(source/'audit.json')!=plan['source_audit_sha256'] or sha(source/'plan.json')!=plan['source_plan_sha256']:raise ValueError('Original source changed')
    main=json.loads((source/'plan.json').read_text(encoding='utf-8'))
    validate_expansion_plan(plan,archive_root=root,check_runtime=False)
    cameras={c['name']:c for c in rig['cameras']};expected={};records=[]
    if len(cameras)!=8:raise ValueError('All eight rig cameras required')
    for town,entry in plan['towns'].items():
        # Unused family and OpenDrive exclusions were checked independently above.
        report=json.loads((root/town/'report.json').read_text(encoding='utf-8'));cal=json.loads((root/town/'calibration.json').read_text(encoding='utf-8'))
        if report['status']!='passed' or not report['cleanup'] or report['plan_sha256']!=sha(root/'plan.json'):raise ValueError('Capture incomplete')
        if report.get('probe_only'):raise ValueError('Probe capture cannot enter training audit')
        if not report.get('explicit_success_readback_frames'):raise ValueError('Formal expansion requires recorded success F+1 counters')
        if report.get('version')!=plan['version'] or report.get('runtime_provenance')!=plan['runtime'] or not report.get('source_files'):
            raise ValueError('Expansion capture requires explicit runtime and source archive')
        if report['source_files'].get('scripts/collect_unused_scenery.py')!=report['collector_sha256']:
            raise ValueError('Collector source identity mismatch')
        if sha(root/town/'runtime_launch.json')!=report['runtime_launch_sha256']:raise ValueError('Native launch receipt bytes changed')
        launch=json.loads((root/town/'runtime_launch.json').read_text(encoding='utf-8'))
        runtime=json.loads((root/'runtime_provenance/state.json').read_text(encoding='utf-8'))
        if Path(launch['executable']).resolve()!=Path(runtime['runtime']).resolve()/'CarlaUE4.exe' or launch['server_binary_sha256']!=runtime['shipping_binary_sha256'] or launch['graphics_api']!='dx12' or launch['quality_level']!='Epic' or launch['port']!=2000:
            raise ValueError('Capture runtime differs from prepared isolated server')
        if report.get('source_files'):
            for relative,digest in report['source_files'].items():
                if sha(root/town/'source_snapshot'/relative)!=digest:raise ValueError('Capture source snapshot changed')
        else:
            archive=root/town/'source_archive'
            if sha(archive/'scripts/collect_segmentation_supplement.py')!=report['collector_sha256']:
                raise ValueError('Reused collector source archive differs')
            for item in json.loads((archive/'receipt.json').read_text(encoding='utf-8'))['files']:
                if sha(safe_path(archive,item['path']))!=item['sha256']:raise ValueError('Reused source archive digest differs')
        if cal['opendrive_sha256']!=entry['opendrive_sha256'] or cal['server']!='0.9.16':raise ValueError('Map/runtime mismatch')
        weather={key:float(value) for key,value in re.findall(r'(\w+)=(-?[\d.]+)',cal['weather'])}
        if any(weather[key]!=0 for key in ['precipitation','precipitation_deposits','wetness']) or weather['sun_altitude_angle']<=0:
            raise ValueError('Observed capture weather is not ideal sunny daytime')
        focal=rig['width']/(2*math.tan(math.radians(rig['horizontal_fov_deg'])/2))
        expected_k=np.array([[focal,0,rig['width']/2],[0,focal,rig['height']/2],[0,0,1]])
        if not np.allclose(np.asarray(cal['K']),expected_k,atol=1e-8,rtol=0):raise ValueError('Calibration intrinsics differ from actual optics')
        for attrs in cal['sensor_attributes'].values():
            if int(attrs['image_size_x'])!=rig['width'] or int(attrs['image_size_y'])!=rig['height'] or float(attrs['fov'])!=rig['horizontal_fov_deg'] or float(attrs['sensor_tick'])!=0:raise ValueError('Sensor optics/cadence mismatch')
        for anchor in entry['anchors']:
            for name,camera in cameras.items():expected[(town,anchor['spawn_index'],name)]=np.asarray(anchor['native_transform_matrix'])@local_camera(camera)
        rows=json.loads((root/town/'records.json').read_text(encoding='utf-8'))
        if report['records']!=len(rows):raise ValueError('Record count mismatch')
        records.extend(rows)
    keys=[(r['town'],r['anchor_spawn_index'],r['camera']) for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('Incomplete or duplicate anchor-camera matrix')
    counts=np.zeros(len(SEGMENTATION_NAMES),dtype=np.int64);views=Counter();candidates=[];training=[];max_pose=0.;max_disagreement=0.;max_raw_disagreement=0.;unlabelled_instance_pixels=0;retries=0
    compared_pixels=0;class_pairs=np.zeros((29,29),dtype=np.int64)
    for index,row in enumerate(records):
        key=(row['town'],row['anchor_spawn_index'],row['camera']);matrix=expected[key]
        group=f'{row["town"].lower()}-static-anchor-{row["anchor_spawn_index"]:03}'
        if row['split']!='train' or row['task']!='segmentation_only' or row['group_id']!=group or row['scene_id']!=group or row['verified_obstacle_actor_ids']!=[]:raise ValueError('Scope/group mismatch')
        if row.get('readback_frame')!=row['frame']+1:raise ValueError('Explicit success readback mismatch')
        if np.abs(np.asarray(row['T_world_from_camera'])-matrix).max()>=.0001:raise ValueError('Planned pose mismatch')
        if set(row['images'])!={'rgb','semantic','depth','instance'}:raise ValueError('Four modalities required')
        images={}
        for modality,item in row['images'].items():
            if item['path']!=f'raw/{group}/{row["camera"]}/{modality}.png':raise ValueError('Image path does not bind the planned anchor/camera/modality')
            image,_,_=read_png(safe_path(root,item['path']),rig['width'],rig['height']);images[modality]=image
            if item['frame']!=row['frame'] or abs(item['timestamp_s']-row['timestamp_s'])>.0001:raise ValueError('Sensor frame/timestamp mismatch')
            error=float(np.abs(np.asarray(item['T_world_from_camera'])-matrix).max());max_pose=max(max_pose,error)
            if error>=.0001:raise ValueError('Readback camera pose mismatch')
        raw=images['semantic'][...,0]
        if images['semantic'][...,1:].any() or raw.max()>28:raise ValueError('Raw semantic format mismatch')
        if images['instance'][...,0].max()>28:raise ValueError('Instance raw class out of range')
        alignment=resolved_class_alignment(raw,images['instance'][...,0]);disagreement=alignment['resolved_disagreement']
        max_disagreement=max(max_disagreement,disagreement);max_raw_disagreement=max(max_raw_disagreement,alignment['raw_disagreement'])
        unlabelled_instance_pixels+=alignment['unlabelled_instance_pixels']
        compared_pixels+=alignment['compared_pixels']
        class_pairs+=np.bincount(raw.astype(np.int64).ravel()*29+images['instance'][...,0].ravel(),minlength=29*29).reshape(29,29)
        if disagreement>.02:raise ValueError('Instance/semantic alignment differs by >2%')
        packed=images['depth'].astype(np.uint32);depth=(packed[...,0]+256*packed[...,1]+65536*packed[...,2])*(1000/16777215)
        if not np.isfinite(depth).all() or depth.min()<0 or depth.max()>1000:raise ValueError('Invalid packed axial depth')
        target=segmentation_target(raw,images['instance'],[]);support=np.bincount(target[target!=255],minlength=len(counts));counts+=support
        for c in plan['target_classes']:views[c]+=int(support[c]>=512)
        retries+=len(row['rejected_capture_attempts'])
        training.append({k:row[k] for k in ['split','town','group_id','scene_id','camera','frame','verified_obstacle_actor_ids']}|{m:row['images'][m]['path'] for m in images})
        candidates.append({'index':index,'support':support.tolist(),'rgb':row['images']['rgb']['path'],'semantic':row['images']['semantic']['path'],'instance':row['images']['instance']['path']})
    path=root/'training/train.jsonl';path.parent.mkdir(exist_ok=True);path.write_text(''.join(json.dumps(r)+'\n' for r in training),encoding='utf-8')
    selected=[]
    for c in plan['target_classes']:
        # Stratify material contexts by town and anchor; repeated angles are correlated.
        seen=set()
        for entry in sorted(candidates,key=lambda r:(-r['support'][c],r['index'])):
            group=training[entry['index']]['group_id']
            if group not in seen and entry['support'][c]>=512:
                seen.add(group);selected.append(entry['index'])
                if len(seen)>=8:break
    # Fence negatives and low native-height candidates must remain visible in QA,
    # rather than only the largest tall-wall positives dominating the gallery.
    seen=set()
    for entry in sorted(candidates,key=lambda r:(-r['support'][13],r['index'])):
        group=training[entry['index']]['group_id']
        if group not in seen and entry['support'][13]>=512:
            selected.append(entry['index']);seen.add(group)
            if len(seen)>=8:break
    low={(town,a['spawn_index']) for town,e in plan['towns'].items() for a in e['anchors']
        if 0<a['selection'].get('native_bbox_height_hint_m',99)<=1.6}
    low_cases=[]
    for town,anchor in sorted(low):
        choices=[c for c in candidates if records[c['index']]['town']==town and records[c['index']]['anchor_spawn_index']==anchor]
        best=max(choices,key=lambda c:c['support'][12])
        if best['support'][12]>=512:low_cases.append(best)
    selected.extend(c['index'] for c in sorted(low_cases,key=lambda c:(-c['support'][12],c['index']))[:8])
    picket_anchors={(town,a['spawn_index']) for town,entry in plan['towns'].items() for a in entry['anchors']
        if a['selection'].get('name_hint_not_material_GT')=='picket'}
    for town,anchor in sorted(picket_anchors):
        possible=[c for c in candidates if records[c['index']]['town']==town and records[c['index']]['anchor_spawn_index']==anchor]
        if possible:selected.append(max(possible,key=lambda c:c['support'][13])['index'])
    selected=list(dict.fromkeys(selected+[i for i in range(0,len(training),max(1,len(training)//8))]))
    (root/'qa').mkdir(exist_ok=True);catalog=[]
    for start in range(0,len(selected),8):
        canvas=Image.new('RGB',(640,8*204))
        for j,index in enumerate(selected[start:start+8]):
            row=training[index];rgb=Image.open(root/row['rgb']).convert('RGB');raw=np.asarray(Image.open(root/row['semantic']))[...,0];instance=np.asarray(Image.open(root/row['instance']))
            target=segmentation_target(raw,instance,[]);color=np.zeros((*target.shape,3),dtype=np.uint8);valid=target!=255;color[valid]=PALETTE[target[valid]]
            rgb=rgb.resize((320,180));gt=Image.fromarray(color).resize((320,180),Image.Resampling.NEAREST)
            y=j*204;canvas.paste(rgb,(0,y+24));canvas.paste(gt,(320,y+24));ImageDraw.Draw(canvas).text((4,y+3),row['group_id']+'/'+row['camera']+' RGB / GT',fill='white')
            catalog.append({'index':index,'sample':row,'support':candidates[index]['support'],'page':f'qa/review-{start//8}.png'})
        canvas.save(root/f'qa/review-{start//8}.png')
    write_json(root/'qa/catalog.json',catalog)
    checks={'matrix_complete':len(records)==len(expected),'pose_alignment':max_pose<.0001,'semantic_instance_alignment':max_disagreement<=.02,
        **{SEGMENTATION_NAMES[c]:int(counts[c])>=plan['minimum_pixels_per_target_class'] and views[c]>=plan['minimum_views_per_target_class'] for c in plan['target_classes']}}
    files=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['audit.json','visual_review.json']]
    result={'version':plan['version'],'status':'passed' if all(checks.values()) else 'failed','checks':checks,'training_approved':False,'task':'segmentation_only','split':'train',
        'source_audit_sha256':plan['source_audit_sha256'],'source_plan_sha256':plan['source_plan_sha256'],'plan_sha256':sha(root/'plan.json'),
        'views':len(records),'groups':len({r['group_id'] for r in training}),'checked_images':len(records)*4,'segmentation_pixels':counts.tolist(),
        'target_class_views_ge512px':dict(views),'rejected_capture_attempts':retries,'max_pose_error':max_pose,'max_resolved_semantic_instance_disagreement':max_disagreement,
        'max_raw_semantic_instance_disagreement':max_raw_disagreement,'unlabelled_instance_pixels':unlabelled_instance_pixels,
        'alignment_compared_pixels':compared_pixels,'raw_class_pairs_semantic_rows_instance_columns':class_pairs.tolist(),
        'alignment_eligibility':'Both raw IDs nonzero. Raw0 means unlabelled, not a conflicting class. Raw disagreement/unresolved support retained; semantic targets unchanged.',
        'auditor_sha256':sha(Path(__file__)),'mapping_source_sha256':sha(ROOT/'src/senior_ai/sunny_data.py'),'files':files,'scope':plan['scope']}
    write_json(root/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
    return 0 if result['status']=='passed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);raise SystemExit(audit(p.parse_args().root))
