"""Causal Phase7 adapter over frozen RGB WorldFrames. No GT or future-data readers."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha,strict_world
from senior_ai.motion_forecast import MotionTracker,appearance_histogram
from senior_ai.dataset import read_sensor_frame,safe_path
from senior_ai.core import write_json

def main(a):
    if a.output.exists():raise FileExistsError('New motion run required')
    config=read(a.config);state=read(a.world/'state.json');proof=read(a.integrity);bev_proof=read(a.phase4_integrity)
    if proof['status']!='passed_artifact_integrity_not_formal_quality' or bev_proof['status']!='passed_artifact_integrity_not_formal_quality':raise ValueError('Upstream integrity seals required')
    for root,guard in [(a.world,proof),(a.phase4,bev_proof)]:
        if guard['run_contract_sha256']!=sha(root/'contract.json'):raise ValueError('Upstream run identity changed')
        for rel,digest in guard['output_manifest'].items():
            if sha(safe_path(root,rel))!=digest:raise ValueError('Upstream prediction changed')
    inventory=read(a.inputs/'rgb_only_inventory.json');stage=a.inputs/'rgb_only'
    if set(inventory)!={p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}:raise ValueError('Contaminated RGB root')
    for rel,digest in inventory.items():
        if not rel.startswith(('inputs/','calibration/')) or sha(safe_path(stage,rel))!=digest:raise ValueError('RGB-only inventory mismatch')
    a.output.mkdir(parents=True);sources={name:sha(ROOT/name) for name in ['src/senior_ai/motion_forecast.py','scripts/run_phase7_motion.py',a.config.resolve().relative_to(ROOT).as_posix()]}
    contract={'task':'causal_RGB_ground_motion_forecast','sources':sources,'config_sha256':sha(a.config),
              'world_contract_sha256':sha(a.world/'contract.json'),'phase4_contract_sha256':sha(a.phase4/'contract.json'),
              'RGB_inventory_sha256':sha(a.inputs/'rgb_only_inventory.json'),'upstream_world_integrity_sha256':sha(a.integrity),
              'upstream_BEV_integrity_sha256':sha(a.phase4_integrity),'GT_or_future_used':False,'control_issued':False}
    write_json(a.output/'contract.json',contract);write_json(a.output/'config.json',config)
    for name in sources:
        p=a.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    tracker=MotionTracker(config);records=[]
    for row in state['records']:
        folder=Path(row['folder']);world=read(a.world/folder/'world.json');parent=read(a.phase4/folder/'frame.json')
        if world['parent_frame_sha256']!=sha(a.phase4/folder/'frame.json'):raise ValueError('Motion/frame binding mismatch')
        with np.load(a.world/folder/'occupancy.npz',allow_pickle=False) as n:strict_world(world,n['semantic'])
        manifest=read_sensor_frame(stage,world['parent_manifest']);started=time.perf_counter()
        rgb={name:np.asarray(Image.open(safe_path(stage,src['image'])).convert('RGB')) for name,src in manifest['cameras'].items()}
        features=[appearance_histogram(rgb,o['sources']) for o in world['objects']]
        objects,tracking=tracker.update(world['objects'],world['scene_id'],world['timestamp_s'],sha(stage/manifest['calibration']),parent['motion'],features)
        elapsed=(time.perf_counter()-started)*1000
        overlay={'schema_version':1,'scene_id':world['scene_id'],'frame':world['frame'],'timestamp_s':world['timestamp_s'],
            'source':'causal_frozen_RGB_cuboids_and_RGB_SE2','objects':objects,'tracking':tracking,
            'coordinate_frame':'current_ego','axes':'x_forward_y_right_z_up','world_source_sha256':sha(a.world/folder/'world.json'),
            'motion_source_sha256':sha(a.phase4/folder/'frame.json'),'timing_ms':elapsed,
            'GT_used_in_inference':False,'future_observations_used':False,'control_issued':False}
        write_json(a.output/folder/'motion.json',overlay)
        record={'scene_id':world['scene_id'],'frame':world['frame'],'folder':folder.as_posix(),'objects':len(objects),
                'velocity_available':sum(o['motion']['velocity_available'] for o in objects),'held_tracks':len(tracking['held_unobserved_tracks']),
                'risk_summary':tracking['risk_summary'],'motion_ms':elapsed}
        records.append(record);write_json(a.output/'progress.json',{'frames':len(records),'latest':record})
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest:raise ValueError('Motion source changed during replay')
    write_json(a.output/'state.json',{'status':'completed_causal_motion_pending_independent_evaluation','records':records,'frames':len(records),
                                    'GT_used_in_inference':False,'future_observations_used':False,'control_issued':False})
    print({'status':'completed','frames':len(records),'objects':sum(r['objects'] for r in records),'forecasts':sum(r['velocity_available'] for r in records)})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['world','phase4','inputs','integrity','phase4-integrity','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/phase7_motion_v1.json');main(p.parse_args())
