"""Independent saved-control recomputation, RGB SHA and geometry checks."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.core import write_json

def main():
    results=[];total=0
    for report_path in sorted((ROOT/'artifacts/runs').glob('phase89-drive-*/report.json')):
        folder=report_path.parent;report=read(report_path)
        if not report.get('records'):continue
        contract=read(folder/'contract.json');source=folder/'source/src/senior_ai/navigation.py'
        spec=importlib.util.spec_from_file_location('frozen_nav',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name,digest in contract['sources'].items():assert sha(folder/'source'/name)==digest
        planner=module.Planner(contract['config']);controller=module.Controller();max_cross=0.;times=[];images=0;controls=[]
        for i,row in enumerate(report['records']):
            p=folder/row['file'];assert sha(p)==row['sha256'];r=read(p)
            assert r['other_actor_GT_in_planning'] is False and r['perception']['GT_other_actors_used'] is False
            assert r['image_age_sim_s']<=contract['config']['max_image_age_s']
            for name,item in r['RGB'].items():
                assert item['frame']==r['frame'] and sha(folder/item['file'])==item['sha256']
                with Image.open(folder/item['file']) as image:assert image.size==(640,360)
                images+=1
            ob={'valid':not(contract.get('test_fault')=='sensor_loss' and i>=8),'timestamp_s':r['timestamp_s'],
                'objects':r['perception']['objects'],'signals':r.get('relevant_RGB_signals',r['perception']['signals']),
                'held_objects':r['perception']['dynamics']['held_unobserved_tracks'],
                'candidate_road_fractions':r['candidate_road_fractions']}
            plan=planner.plan(r['navigation'],ob,r['speed_mps'],r['timestamp_s']+r['image_age_sim_s'],.2)
            assert plan==r['plan']
            control=controller.step(plan,r['speed_mps'],.2)
            # Cancel can override the final command with full brake.
            assert control==r['control'] or (report['status']=='cancelled' and r['control']=={'throttle':0.,'brake':1.,'steer':0.})
            controls.append(r['control']);max_cross=max(max_cross,r['navigation']['cross_track_m']);times.append(r['perception']['inference_ms'])
            for o in r['perception']['objects']:
                assert 'actor_id' not in o and np.isfinite(o['center_ego_m']).all()
            total+=1
        assert images==len(report['records'])*8
        if report['engineering_route_passed']:
            assert report['status']=='arrived' and report['records'][-1]['speed_mps']<=.3 and report['records'][-1]['goal_distance_m']<=2.5
            assert not report['events']['collisions'] and max_cross<=1.5
        results.append({'run_id':folder.name,'status':report['status'],'engineering_passed':report['engineering_route_passed'],'frames':len(times),'RGB_images':images,
            'max_cross_track_m':max_cross,'inference_p50_ms':float(np.median(times)),'inference_p95_ms':float(np.quantile(times,.95)),
            'wall_s':report['wall_s'],'simulation_s':report['simulation_s'],'recomputed_plan_and_control':'exact',
            'report_sha256':sha(report_path),'contract_sha256':sha(folder/'contract.json')})
    write_json(ROOT/'artifacts/reports/phase89-integrity-final.json',{'status':'passed_saved_source_RGB_control_and_metrics_integrity','runs':results,'frames':total,
        'formal_quality_passed':False,'limits':'Source/control/receipt consistency, not new perception GT scoring or independent traffic compliance certification.'})
    print({'runs':len(results),'frames':total,'states':[(r['run_id'],r['status']) for r in results]})
if __name__=='__main__':main()
