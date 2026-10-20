"""Recompute all causal motion outputs, prefix, transfer geometry and saved metrics."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.world_viewer_service import ViewerBundle
from senior_ai.motion_forecast import MotionTracker,appearance_histogram
from senior_ai.cuboid import corners
from senior_ai.core import write_json

def main():
    motion=ROOT/'artifacts/runs/phase7-motion-v1-004';world=ROOT/'artifacts/runs/phase7-world-v1-001';bev=ROOT/'artifacts/runs/phase7-bev-v1-001'
    stage=ROOT/'artifacts/runs/phase7-inputs-v1-001/rgb_only';out=ROOT/'artifacts/reports/phase7-integrity-final.json'
    config=read(motion/'config.json');contract=read(motion/'contract.json');state=read(motion/'state.json')
    for rel,digest in contract['sources'].items():assert sha(motion/'source'/rel)==sha(ROOT/rel)==digest
    bundle=ViewerBundle(ROOT/'artifacts/runs/phase7-viewer-bundle-v1-001');tracker=MotionTracker(config);prefix=MotionTracker(config)
    geometry=[];counts={'objects':0,'forecasts':0,'held':0,'prefix_frames':0};sources={};times=[]
    for row,index in zip(state['records'],bundle.manifest['frames'],strict=True):
        folder=Path(row['folder']);w=read(world/folder/'world.json');p=read(bev/folder/'frame.json');m=read(motion/folder/'motion.json');display=read(bundle.file(index['file']))
        assert m['world_source_sha256']==sha(world/folder/'world.json')==display['world_source_sha256']
        assert m['motion_source_sha256']==sha(bev/folder/'frame.json') and display['motion_prediction_source_sha256']==sha(motion/folder/'motion.json')
        sensor=read(stage/w['parent_manifest']);rgb={n:np.asarray(Image.open(stage/v['image']).convert('RGB')) for n,v in sensor['cameras'].items()}
        features=[appearance_histogram(rgb,o['sources']) for o in w['objects']]
        args=(w['objects'],w['scene_id'],w['timestamp_s'],sha(stage/sensor['calibration']),p['motion'],features)
        objects,tracking=tracker.update(*args);assert objects==m['objects'] and tracking==m['tracking']
        if counts['prefix_frames']<80:
            po,pt=prefix.update(*args);assert po==m['objects'] and pt==m['tracking'];counts['prefix_frames']+=1
        assert display['dynamics']==tracking and display['future_observations_used'] is False
        with np.load(world/folder/'occupancy.npz',allow_pickle=False) as n:assert bundle.file(display['occupancy']['file']).read_bytes()==n['semantic'].tobytes(order='C')
        checks=[]
        for o,raw,d in zip(objects,w['objects'],display['objects'],strict=True):
            assert o['track_id']==raw['track_id']==d['track_id']
            assert o['track_age_measurements']==raw['track_age_measurements']==d['track_age_measurements']
            for field in ['class_id','center_ego_m','size_lwh_m','yaw_rad']:assert o[field]==raw[field]==d[field]
            for field in ['motion','forecast','risk','lifecycle','history_ego_xy_m']:assert o[field]==d[field]
            assert not set(d)&{'actor_id','world_pose','world_vertices_m','sources','upstream_track_id'}
            forecast=o['forecast'];v=o['motion'];counts['objects']+=1
            if forecast['status']=='available':
                counts['forecasts']+=1;tt=np.array(forecast['times_s']);pred=np.array(forecast['positions_ego_xy_m'])
                np.testing.assert_allclose(pred,np.array(o['center_ego_m'][:2])+tt[:,None]*v['velocity_ground_ego_mps'],rtol=0,atol=1e-12)
                assert len(tt)==16 and tt[0]==0 and abs(tt[-1]-3)<1e-9 and all(x>=0 for x in forecast['uncertainty_radius_m'])
            c=corners(o);checks.append({'track_id':o['track_id'],'corners_three':(c[:,[1,2,0]]*np.array([1,1,-1])).tolist()})
        for h in tracking['held_unobserved_tracks']:assert h['measurement_current'] is False and h['used_for_occupancy'] is False
        counts['held']+=len(tracking['held_unobserved_tracks']);times.append(m['timing_ms'])
        geometry.append({'key':index['key'],'file':index['file'],'objects':checks})
        sources[(folder/'motion.json').as_posix()]=sha(motion/folder/'motion.json')
    report=read(ROOT/'artifacts/runs/phase7-evaluation-v1-004/report.json')
    velocity=read(ROOT/'artifacts/runs/phase7-evaluation-v1-004/paired_velocity.json');details=read(ROOT/'artifacts/runs/phase7-evaluation-v1-004/forecast_details.json')
    assert report['tracking']['ID_switches']==report['baseline_tracking_same_detections']['ID_switches']
    assert report['tracking']['IDF1']==report['baseline_tracking_same_detections']['IDF1']
    assert report['velocity']['paired_samples']==len(velocity)
    assert report['velocity']['vector_MAE_mps']==float(np.mean([v['vector_error_mps'] for v in velocity]))
    for h,d in details.items():
        score=report['horizons_s'][h];assert score['covered_GT']==len(d)
        assert score['ADE_m']==float(np.mean([x['ADE'] for x in d])) and score['FDE_m']==float(np.mean([x['FDE'] for x in d]))
        assert score['conditional_misses_max2m']==sum(x['max_error_m']>2 for x in d)
        assert abs(score['all_GT_miss_rate_including_unavailable']-(score['eligible_GT']-len(d)+score['conditional_misses_max2m'])/score['eligible_GT'])<1e-12
    write_json(out,{'status':'passed_causal_replay_prefix_transfer_and_metric_recount','frames':len(geometry),**counts,
        'verified_files':len(bundle.inventory),'bundle_sha256':sha(bundle.root/'bundle.json'),'motion_output_manifest':sources,'geometry_for_node':geometry,
        'motion_compute_ms':{'p50':float(np.quantile(times,.5)),'p95':float(np.quantile(times,.95))},
        'identity_backbone_unchanged':True,'formal_model_quality_passed':False,'limits':'Correct causal engineering and metric arithmetic only. Same known train/validation, not independent quality certification.'})
    print({'status':'passed','frames':len(geometry),**counts,'motion_compute_p50_ms':float(np.quantile(times,.5)),'motion_compute_p95_ms':float(np.quantile(times,.95))})

if __name__=='__main__':main()
