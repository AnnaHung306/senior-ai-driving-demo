"""Whitelist export of causal Phase7 predictions; GT aggregate metrics only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha,display_payload,relative_path
from senior_ai.dataset import read_sensor_frame,safe_path
from senior_ai.core import write_json

def main(a):
    if a.output.exists():raise FileExistsError('New display bundle required')
    contract=read(a.motion/'contract.json');state=read(a.motion/'state.json');world_contract=read(a.world/'contract.json');config=read(a.phase4/'config.json')
    assert contract['world_contract_sha256']==sha(a.world/'contract.json') and contract['phase4_contract_sha256']==sha(a.phase4/'contract.json')
    assert contract['RGB_inventory_sha256']==sha(a.inputs/'rgb_only_inventory.json')
    ev=read(a.evaluation/'report.json');assert ev['motion_contract_sha256']==sha(a.motion/'contract.json')
    for name,digest in contract['sources'].items():assert sha(a.motion/'source'/name)==digest
    for r,g in [(a.world,read(a.world_integrity)),(a.phase4,read(a.phase4_integrity))]:
        for path,digest in g['output_manifest'].items():assert sha(safe_path(r,path))==digest
    inventory=read(a.inputs/'rgb_only_inventory.json');stage=a.inputs/'rgb_only'
    assert set(inventory)=={p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    for name,digest in inventory.items():assert sha(safe_path(stage,name))==digest
    a.output.mkdir(parents=True);files={};frames=[]
    def save(name,data):
        relative_path(name);p=a.output/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():raise FileExistsError(p)
        p.write_bytes(data);files[name]=sha(p)
    fields=['motion','forecast','risk','history_ego_xy_m','lifecycle','track_id','track_age_measurements']
    for row in state['records']:
        folder=Path(row['folder']);overlay=read(a.motion/folder/'motion.json');world=read(a.world/folder/'world.json')
        assert overlay['world_source_sha256']==sha(a.world/folder/'world.json')
        assert overlay['motion_source_sha256']==sha(a.phase4/folder/'frame.json')
        assert overlay['future_observations_used'] is False and overlay['GT_used_in_inference'] is False
        with np.load(a.world/folder/'occupancy.npz',allow_pickle=False) as n:occ=n['semantic']
        with np.load(a.phase4/folder/'bev.npz',allow_pickle=False) as n:payload=display_payload(world,occ,n,config)
        for d,m,w in zip(payload['objects'],overlay['objects'],world['objects'],strict=True):
            assert m['upstream_track_id']==w['track_id']
            for name in ['center_ego_m','size_lwh_m','yaw_rad','class_id']:assert m[name]==w[name]
            d.update({name:m[name] for name in fields})
        key=world['scene_id']+'/'+str(world['frame']);binary='frames/'+key+'.bin';save(binary,occ.tobytes(order='C'))
        sensor=read_sensor_frame(stage,world['parent_manifest']);cameras={}
        for name,v in sensor['cameras'].items():
            relative=f"rgb/{world['scene_id']}/{name}/{world['frame']}.png";dest=a.output/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(safe_path(stage,v['image']),dest);files[relative]=sha(dest);cameras[name]={'file':relative,'sha256':files[relative]}
        payload.update(dynamics=overlay['tracking'],future_observations_used=False,motion_prediction_source_sha256=sha(a.motion/folder/'motion.json'),
            world_source_sha256=sha(a.world/folder/'world.json'),parent_bev_sha256=sha(a.phase4/folder/'bev.npz'),cameras=cameras,
            occupancy={'file':binary,'sha256':files[binary],'shape':list(occ.shape),'dtype':'uint8','order':'C'})
        payload['quality']['velocity_available']=any(o['motion']['velocity_available'] for o in payload['objects'])
        payload['held_track_count']=len(overlay['tracking']['held_unobserved_tracks']);payload['tracking_status']=overlay['tracking']['reset_reason']
        payload['timing_ms']['causal_motion_forecasting']=overlay['timing_ms']
        relative='frames/'+key+'.json';save(relative,(json.dumps(payload,separators=(',',':'),allow_nan=False)+'\n').encode())
        frames.append({'key':key,'scene_id':world['scene_id'],'frame':world['frame'],'timestamp_s':world['timestamp_s'],'objects':len(payload['objects']),
                       'file':relative,'sha256':files[relative]})
    scenes=[]
    for scene in dict.fromkeys(f['scene_id'] for f in frames):
        records=[f for f in frames if f['scene_id']==scene];duration=records[-1]['timestamp_s']-records[0]['timestamp_s']
        scenes.append({'id':scene,'frames':[r['key'] for r in records],'duration_s':duration,'source_frequency_Hz':(len(records)-1)/duration})
    # Synthetic trace is a separate explicitly labelled page; never a WorldFrame scene.
    synthetic='frames/synthetic-diagnostics.json';save(synthetic,(a.synthetic/'traces.json').read_bytes())
    val=ev['by_split']['validation'];forecast=val['horizons_s'];tracking=val['tracking']
    write_json(a.output/'bundle.json',{'schema_version':1,'source_mode':'recorded_inference_replay','label':'晴天RGB動態研究 · 已計算因果推論',
        'run_contract_sha256':sha(a.world/'contract.json'),'checkpoint_sha256':world_contract['checkpoint_sha256'],'phase4_contract_sha256':sha(a.phase4/'contract.json'),
        'motion_contract_sha256':sha(a.motion/'contract.json'),'motion_profile':'phase7_kinematic_v1','evaluation_report_sha256':sha(a.evaluation/'report.json'),
        'formal_model_quality_passed':False,'GT_geometry_exported':False,'control_issued':False,'frames':frames,'scenes':scenes,'file_inventory':files,
        'quality_context':{'scope':'known_model_selection_validation_long_sequences_not_heldout','position_precision_at_2m':tracking['precision_at_2m'],
            'position_recall_at_2m':tracking['recall_at_2m'],'ID_switches':tracking.get('ID_switches',0)},
        'motion_evaluation':{'scope':'known_model_selection_validation_long_sequences_not_heldout','velocity_vector_MAE_mps':val['velocity']['vector_MAE_mps'],
            'ADE1_m':forecast['1']['ADE_m'],'FDE2_m':forecast['2']['FDE_m'],'forecast_coverage_2s':forecast['2']['forecast_coverage'],
            'ID_switches':tracking.get('ID_switches',0),'baseline_ID_switches':val['baseline'].get('ID_switches',0),'formal_quality_passed':False},
        'synthetic_diagnostics':{'file':synthetic,'sha256':files[synthetic],'scope':'synthetic_geometry_not_AI_or_CARLA'},
        'exporter_sha256':sha(Path(__file__)),'adapter_sha256':sha(ROOT/'src/senior_ai/world_display.py')})
    print({'status':'exported_causal_motion_viewer','frames':len(frames),'scenes':len(scenes),'files':len(files),'images':len(frames)*8})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['world','phase4','inputs','motion','evaluation','world-integrity','phase4-integrity','synthetic','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
