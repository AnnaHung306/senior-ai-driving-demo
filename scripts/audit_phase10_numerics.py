"""Record malformed-evidence counterexamples before a guarded correction."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation import Planner
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    c=read(ROOT/'configs/phase89_navigation_v1.json')
    route={'local_xy_m':[[x,0] for x in range(1,16)],'remaining_m':30.,'cross_track_m':0.,'goal_distance_m':30.}
    ob={'valid':True,'timestamp_s':10.,'objects':[],'signals':[],'candidate_road_fractions':{'0.0':1.,'-0.45':1.,'0.45':1.}}
    cases=[]
    for field in ['timestamp','road_fraction','cross_track','remaining','goal_distance','signal_score']:
        for value,name in [(float('nan'),'NaN'),(float('inf'),'Infinity')]:
            r,o=copy.deepcopy(route),copy.deepcopy(ob)
            if field=='timestamp':o['timestamp_s']=value
            elif field=='road_fraction':o['candidate_road_fractions']={str(x):value for x in c['lateral_offsets_m']}
            elif field=='signal_score':o['signals']=[{'phase':'red','score':value}]
            else:r[{'cross_track':'cross_track_m','remaining':'remaining_m','goal_distance':'goal_distance_m'}[field]]=value
            try:
                plan=Planner(c).plan(r,o,2.,10.1,.2)
                cases.append({'field':field,'input':name,'state':plan['state'],'target_speed_mps':plan['target_speed_mps'],
                    'safe_rejected_or_stopped':plan['target_speed_mps']==0})
            except (ValueError,TypeError,KeyError) as e:cases.append({'field':field,'input':name,'exception':str(e),'safe_rejected_or_stopped':True})
    curve=copy.deepcopy(route);curve['local_xy_m']=[[0,0],[2,0],[4,0],[4,2],[3,4],[1,6]]
    curved_ob=copy.deepcopy(ob);curved_ob['objects']=[{'center_ego_m':[4,2,1],'size_lwh_m':[1,1,2]}]
    curved=Planner(c).plan(curve,curved_ob,2.,10.1,.2)
    write_json(a.output,{'status':'numerical_evidence_boundary_probe','navigation_source_sha256':sha(ROOT/'src/senior_ai/navigation.py'),
        'cases':cases,'unsafe_acceptances':sum(not x['safe_rejected_or_stopped'] for x in cases),
        'curved_path_counterexample':{'path':curve['local_xy_m'],'obstacle':curved_ob['objects'][0],
            'observed_plan':curved,'object_center_exactly_on_polyline':True,
            'limits':'Numeric supported-path-shape diagnostic, not a CARLA crash reproduction'},
        'actual_sensor_failure_reproduction':False,'scope':'numeric malformed-provider robustness, not evidence of CARLA naturally producing NaN'})
    print(cases)

if __name__=='__main__':main()
