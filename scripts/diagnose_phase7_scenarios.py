"""Fixed synthetic diagnostics, explicitly separate from RGB/AI performance."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.motion_forecast import MotionTracker,swept_contact
from senior_ai.world_display import read,sha
from senior_ai.core import write_json

def scenario(name,t):
    if name=='following':return np.array([14+4*t,0.])
    if name=='turning':return np.array([12+8*np.sin(.25*t),8*(1-np.cos(.25*t))])
    if name=='cut_in':return np.array([18+2*t,max(0,5-1.25*max(0,t-3))])
    if name=='sudden_brake':
        q=min(max(t-3,0),2);return np.array([16+5*min(t,3)+5*q-1.25*q*q,0.])
    raise ValueError(name)

def main():
    config=read(ROOT/'configs/phase7_motion_v1.json');out=ROOT/'artifacts/runs/phase7-synthetic-v1-001'
    if out.exists():raise FileExistsError(out)
    rng=np.random.default_rng(config['seed']);cases=[];diagnostic=[]
    for name in ['following','turning','cut_in','sudden_brake']:
        for repeat in range(4):
            tracker=MotionTracker(config);history=[];ids=[];errors={h:[] for h in [1.,2.,3.]}
            for k in range(51):
                t=k*.2;ego=np.array([4*t,0.]);point=scenario(name,t)-ego
                measurement=point+rng.normal(0,.08,2)
                obj={'class_id':1,'center_ego_m':[*measurement,1.],'size_lwh_m':[4.5,1.9,1.5],'yaw_rad':0.,'score_uncalibrated':.9,'measurement_current':True}
                objects,info=tracker.update([obj],'synthetic-'+name,t,'fixed-rig',{'status':'aligned','previous_to_current':[[1,0,-.8],[0,1,0],[0,0,1]]})
                o=objects[0];ids.append(o['track_id']);history.append({'time_s':t,'measured_xy':measurement.tolist(),'motion':o['motion'],'forecast':o['forecast'],'risk':o['risk'],'ego':info['ego_motion']})
                if o['forecast']['status']!='available' or t<2:continue
                for h in errors:
                    if t+h>10+1e-6:continue
                    tt=np.array(o['forecast']['times_s']);pp=np.array(o['forecast']['positions_ego_xy_m']);take=(tt>0)&(tt<=h+1e-6)
                    target=np.array([scenario(name,t+future)-ego for future in tt[take]])
                    e=np.linalg.norm(pp[take]-target,axis=1);errors[h].append({'ADE':float(e.mean()),'FDE':float(e[-1])})
            result={'scenario':name,'repeat':repeat,'ID_switches':sum(a!=b for a,b in zip(ids,ids[1:])),
                'errors':{str(h):{'samples':len(v),'ADE':float(np.mean([x['ADE'] for x in v])),'FDE':float(np.mean([x['FDE'] for x in v]))} for h,v in errors.items()}}
            cases.append(result)
            if repeat==0:diagnostic.append({'scenario':name,'scope':'synthetic_geometry_not_AI_or_CARLA','frames':history})
    # Unambiguous geometry risks with analytic expected answer, including tunneling.
    risk_cases=[]
    for relative,expected in [([[12,0],[0,0]],True),([[12,12],[0,12]],False),([[10,0],[-10,0]],True),([[12,0],[20,0]],False)]:
        t,c=swept_contact(relative,[0,2],[2.5,2.5]);assert (t is not None)==expected;risk_cases.append({'relative':relative,'expected':expected,'TTC_s':t,'minimum_clearance':c})
    steady=[r for r in cases if r['scenario']=='following'];gate=all(r['errors']['1.0']['ADE']<=.75 and r['errors']['2.0']['FDE']<=1.5 and r['ID_switches']==0 for r in steady)
    write_json(out/'report.json',{'status':'completed_synthetic_diagnostics','scope':'No RGB model, GT actors or CARLA; diagnostic inputs are generated noisy positions',
        'config_sha256':sha(ROOT/'configs/phase7_motion_v1.json'),'tool_sha256':sha(Path(__file__)),'cases':cases,'risk_cases':risk_cases,
        'steady_engineering_gates_passed':gate,'all_scenarios_predictable':False,'limits':'CV cannot anticipate turns/cut-ins/sudden braking; errors intentionally retained, no best-of-K scoring'})
    write_json(out/'traces.json',diagnostic);print(json.dumps({'steady_engineering_gates_passed':gate,'cases':len(cases),'motion_risk_geometry_cases':len(risk_cases)}))

if __name__=='__main__':main()
