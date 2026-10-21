"""Offline semantic oracle under identical flat-ground projection, never control."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json
from senior_ai.bev_geometry import Grid,cameras_from_calibration
from senior_ai.road_fusion_cache import CachedRoadFusion
from senior_ai.live_perception import LivePerception
from senior_ai.phase10_metrics import RAW_TO_SEG

def main():
    out=ROOT/'artifacts/reports/phase10-road-gate-oracle-final.json'
    if out.exists():raise FileExistsError(out)
    config=read(ROOT/'configs/phase4_bev_v1.json');nav_config=read(ROOT/'configs/phase89_navigation_v1.json');rows=[]
    for folder in sorted((ROOT/'artifacts/runs').glob('phase10-drive-*-audit-v1-001')):
        report=read(folder/'report.json');available=[r for r in report['records'] if (folder/'audit'/f'{r["frame"]:08d}'/'front-instance.png').exists()]
        chosen=[available[0],available[-1]]
        failed=next((r for r in available if r['state']=='FAILSAFE'),None)
        if failed:chosen.append(failed)
        selected={r['frame']:r for r in chosen};grid=Grid(**config['grid']);cameras=cameras_from_calibration(read(folder/'calibration.json'));fuser=CachedRoadFusion(cameras,grid,config)
        stub=object.__new__(LivePerception);stub.grid=grid
        lut=np.zeros(29,np.uint8)
        for key,value in RAW_TO_SEG.items():lut[key]=value
        for entry in selected.values():
            record=read(folder/entry['file']);sample=folder/'audit'/f'{entry["frame"]:08d}';variants={}
            for variant in ['GT_semantic_oracle','RGB_hard_segmentation']:
                observations={}
                for name in cameras:
                    if variant=='GT_semantic_oracle':ids=lut[np.array(Image.open(sample/f'{name}-instance.png'))[:,:,0]]
                    else:ids=np.array(Image.open(sample/f'{name}-predicted_seg.png'))
                    probability=np.stack([np.isin(ids,[1,10]),ids==10],axis=-1).astype(np.float32)
                    observations[name]={'ids':ids,'ground_probabilities':probability}
                fused=fuser(observations)
                variants[variant]=LivePerception.road_fractions(stub,fused,record['navigation']['local_xy_m'],nav_config)
            rows.append({'run_id':folder.name,'frame':entry['frame'],'state':entry['state'],
                'actual_RGB_soft_road_fractions':record['candidate_road_fractions'],**variants,
                'original_route_status':report['status']})
    write_json(out,{'status':'completed_offline_same_geometry_road_evidence_diagnostic','samples':rows,
        'minimum_fraction_unchanged':nav_config['minimum_road_fraction'],'used_for_vehicle_control':False,
        'limits':['GT labels and model hard labels projected under the SAME flat-ground geometry; this is not independent metric BEV/depth accuracy.',
          'If even GT semantic support is low, inspect geometry/map/visibility sampling; hard-vs-soft gaps can implicate probability confidence gates.',
          'First/last sampled GT bundle plus first sampled FAILSAFE in each of six fixed routes; no success/threshold tuning.'],
        'tool_sha256':sha(Path(__file__))})
    print([{'run':r['run_id'],'frame':r['frame'],'actual':max(r['actual_RGB_soft_road_fractions'].values()),'GT':max(r['GT_semantic_oracle'].values()),'hard':max(r['RGB_hard_segmentation'].values())} for r in rows])

if __name__=='__main__':main()
