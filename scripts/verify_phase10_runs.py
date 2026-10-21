"""Replay each saved planner/controller from its own version and verify RGB/GT bytes."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    out=ROOT/'artifacts/reports/phase10-integrity-final.json'
    if out.exists():raise FileExistsError(out)
    results=[];images=0;frames=0;GT_images=0
    for path in sorted((ROOT/'artifacts/runs').glob('phase10-drive-*/report.json')):
        folder=path.parent;report=read(path)
        if not report.get('records'):continue
        contract=read(folder/'contract.json')
        for relative,digest in contract['sources'].items():assert sha(folder/'source'/relative)==digest
        name='senior_ai.navigation_safety';previous=sys.modules.get(name)
        safety=folder/'source/src/senior_ai/navigation_safety.py'
        if safety.exists():sys.modules[name]=load(name,safety)
        try:
            module=load('senior_ai._frozen_phase10_nav',folder/'source/src/senior_ai/navigation.py')
            planner=module.Planner(contract['config']);controller=module.Controller()
            for i,entry in enumerate(report['records']):
                p=folder/entry['file'];assert sha(p)==entry['sha256'];r=read(p)
                assert r['other_actor_GT_in_planning'] is False and r['perception']['GT_other_actors_used'] is False
                assert not any(k in r['plan'] for k in ['actor_id','GT_objects'])
                for camera,item in r['RGB'].items():
                    assert item['frame']==r['frame'] and sha(folder/item['file'])==item['sha256']
                    with Image.open(folder/item['file']) as image:assert image.size==(640,360)
                    images+=1
                ob={'valid':not(contract.get('test_fault')=='sensor_loss' and i>=8),'timestamp_s':r['timestamp_s'],
                    'objects':r['perception']['objects'],'signals':r['relevant_RGB_signals'],
                    'held_objects':r['perception']['dynamics']['held_unobserved_tracks'],
                    'candidate_road_fractions':r['candidate_road_fractions']}
                plan=planner.plan(r['navigation'],ob,r['speed_mps'],r['timestamp_s']+r['image_age_sim_s'],.2)
                assert plan==r['plan'],(folder.name,r['frame'],'plan')
                command=controller.step(plan,r['speed_mps'],.2)
                assert command==r['control'] or (report['status']=='cancelled' and r['control']=={'throttle':0.,'brake':1.,'steer':0.}), (folder.name,r['frame'],'control')
                gt=folder/'audit'/f'{r["frame"]:08d}.json'
                if gt.exists():
                    g=read(gt);assert abs(g['timestamp_s']-r['timestamp_s'])<1e-5
                    if g['sampled_images']:
                        for camera,data in g['cameras'].items():
                            d=folder/'audit'/f'{r["frame"]:08d}'
                            for field,ending in [('instance_sha256','instance'),('predicted_seg_sha256','predicted_seg')]:
                                assert sha(d/f'{camera}-{ending}.png')==data[field];GT_images+=1
                frames+=1
            results.append({'run_id':folder.name,'status':report['status'],'frames':len(report['records']),
                'source_snapshots_verified':len(contract['sources']),'planner_and_control_replay':'exact',
                'contract_sha256':sha(folder/'contract.json'),'report_sha256':sha(path)})
        finally:
            if previous is not None:sys.modules[name]=previous
            else:sys.modules.pop(name,None)
    write_json(out,{'status':'passed_saved_source_RGB_GT_and_frozen_control_integrity','runs':results,
        'frames':frames,'RGB_images':images,'GT_and_predicted_semantic_images':GT_images,'formal_quality_passed':False})
    print({'runs':len(results),'frames':frames,'RGB':images,'GT_and_semantic':GT_images})

if __name__=='__main__':main()
