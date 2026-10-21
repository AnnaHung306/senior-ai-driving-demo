"""Report final named engineering cases while retaining every earlier attempt."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.core import write_json

def main():
    named=['phase89-drive-unseen-straight1-v1-001','phase89-drive-unseen-straight2-v1-001',
           'phase89-drive-signal-probe-v1-001','phase89-drive-blocker-probe-v1-002','phase89-drive-pedestrian-probe-v1-001']
    results=[];times=[];speeds=[];source_versions={}
    for name in named:
        folder=ROOT/'artifacts/runs'/name;r=read(folder/'report.json');c=read(folder/'contract.json')
        assert r['engineering_route_passed'] and r['status']=='arrived'
        rows=[read(folder/x['file']) for x in r['records']]
        times += [x['perception']['inference_ms'] for x in rows];speeds += [x['speed_mps'] for x in rows]
        max_cross=max(x['navigation']['cross_track_m'] for x in rows)
        assert max_cross<=1.5 and rows[-1]['speed_mps']<=.3 and rows[-1]['navigation']['goal_distance_m']<=2.5
        results.append({'run_id':name,'frames':len(rows),'length_m':r['route']['length_m'],'status':r['status'],
            'collisions':len(r['events']['collisions']),'lane_events':len(r['events']['lane_invasions']),
            'max_cross_track_m':max_cross,'final_goal_distance_m':rows[-1]['navigation']['goal_distance_m'],
            'final_speed_mps':rows[-1]['speed_mps'],'simulation_s':r['simulation_s'],'wall_s':r['wall_s'],
            'report_sha256':sha(folder/'report.json'),'contract_sha256':sha(folder/'contract.json')})
        source_versions[name]=c['sources']
    probes={}
    for key,name in [('signal',named[2]),('blocker',named[3]),('pedestrian',named[4])]:
        p=ROOT/'artifacts/runs'/(name+'-director')/'report.json';probes[key]=read(p);assert probes[key]['passed']
    signal_rows=[read(ROOT/'artifacts/runs'/named[2]/x['file']) for x in read(ROOT/'artifacts/runs'/named[2]/'report.json')['records']]
    release=next(e['frame'] for e in probes['signal']['events'] if e['event']=='environment_green_set_after_RGB_stop')
    red_rows=[r for r in signal_rows if r['frame']<=release]
    assert all(r['map_junction_distance_m'] is None or r['map_junction_distance_m']>=0 for r in red_rows)
    assert sum(r['plan']['state']=='SIGNAL_STOP' and r['speed_mps']<.15 for r in red_rows)>=3
    faults={}
    for name in ['phase89-drive-sensor-fault-v1-001','phase89-drive-camera-timeout-v1-001']:
        r=read(ROOT/'artifacts/runs'/name/'report.json');faults[name]={'status':r['status'],'checks':r['checks'],
            'final_recorded_speed_mps':r['records'][-1]['speed_mps'],'cleanup_final_speed_mps':r.get('cleanup_final_speed_mps'),
            'errors':r['errors'],'report_sha256':sha(ROOT/'artifacts/runs'/name/'report.json')}
    assert faults['phase89-drive-sensor-fault-v1-001']['status']=='fault_stopped'
    timeout=faults['phase89-drive-camera-timeout-v1-001'];assert timeout['status']=='sensor_timeout_stopped' and timeout['checks']['cleanup_stopped'] and timeout['cleanup_final_speed_mps']<=.3
    attempts=[]
    for folder in sorted((ROOT/'artifacts/runs').glob('phase89-drive-*')):
        if not (folder/'contract.json').is_file():continue
        c=read(folder/'contract.json')
        if c.get('task')!='CARLA_map_RGB_hybrid_closed_loop':continue
        report=read(folder/'report.json') if (folder/'report.json').is_file() else {}
        attempts.append({'run_id':folder.name,'status':report.get('status','interrupted_no_final_report'),
             'frames':len(list(folder.glob('frames/*/record.json'))),'engineering_route_passed':report.get('engineering_route_passed',False),
             'contract_sha256':sha(folder/'contract.json'),'errors':report.get('errors',[])})
    write_json(ROOT/'artifacts/reports/phase89-validation-final.json',{'status':'completed_limited_closed_loop_engineering_demo',
        'named_final_route_cases':results,'named_final_route_cases_passed':len(results),'named_final_route_cases_total':len(named),
        'controlled_dynamic_probes':probes,'signal_red_phase_no_junction_entry_observed':True,'fault_tests':faults,
        'inference_ms_named_cases':{'samples':len(times),'p50':float(np.median(times)),'p95':float(np.quantile(times,.95))},
        'maximum_speed_named_cases_mps':max(speeds),'source_versions_per_case':source_versions,'all_attempts_including_failures':attempts,
        'formal_quality_passed':False,'limitations':['Final named versions are development engineering cases, not statistical success rate or heldout model quality.',
          'Three map routes in the same city; original007 training ego traces separated 86-139m, historical/camera/asset overlap not excluded.',
          'Ground-truth map and ego localization are explicit planning priors; other actor GT never enters planner.',
          'Signal/blocker/pedestrian scenes use a declared environment director, no BasicAgent/autopilot for ego.',
          'Object classes/sizes can be wrong and following/yielding shows stop-go jitter; comfort not validated.',
          'No learned intent/planner, dense mixed traffic, arbitrary lane changes, overtake or general route reliability certified.',
          'Camera timeout case intentionally contains a TimeoutError; successful emergency stop is distinct from route completion.']})
    print({'final_named_cases':len(results),'all_attempts':len(attempts),'frames_named':len(times),'inference_p50_ms':float(np.median(times)),'inference_p95_ms':float(np.quantile(times,.95))})
if __name__=='__main__':main()
