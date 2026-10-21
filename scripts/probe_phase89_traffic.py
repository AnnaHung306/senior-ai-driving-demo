"""Declared CARLA test director. Never passes actor/light GT into ego planning."""
import argparse,math,sys,threading,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.core import write_json
from senior_ai.navigation_runtime import drive

def main(a):
    import carla
    route=next(r for r in read(ROOT/'artifacts/runs/phase89-route-catalog-v1-002/catalog.json')['routes'] if r['id']==('turn-1' if a.scenario=='signal' else 'straight-2'))
    probe=Path(str(a.output)+'-director');probe.mkdir(parents=True,exist_ok=False)
    write_json(probe/'contract.json',{'scenario':a.scenario,'source_sha256':sha(Path(__file__)),'route':route,
        'asset':getattr(a,'asset',None),'minimum_stationary_hold_sim_s':getattr(a,'stop_hold_s',.6),
        'lead_brake_recipe':'vehicle throttle.35 for3 simulation seconds, then full brake; resume throttle.4 after stationary ego RGB obstacle stop' if a.scenario=='lead_brake' else None,
        'recipe':'signal: freeze red until 3 RGB SIGNAL_STOP frames at <.15m/s then set green; blocker: parked vehicle at route15m, remove only after 3 RGB OBSTACLE_STOP stationary frames; pedestrian: cross at 12m from +3m lateral at 1m/s',
        'other_actor_GT_given_to_planner':False,'CARLA_tick_owner':'drive only; director never ticks','route_scoring_gate_unchanged':True})
    cancel=threading.Event();done=threading.Event();events=[];errors=[];restored=[];released=False
    def director():
        nonlocal released
        owned=[];saved=[];client=carla.Client('127.0.0.1',2000);client.set_timeout(20);w=client.get_world();last=-1;stopped=0;spawned=False;released=False;spawn_time=None;lead_braked=False
        try:
            while not done.is_set():
                progress=a.output/'progress.json'
                if not progress.exists():time.sleep(.1);continue
                try:p=read(progress)
                except (ValueError,PermissionError):continue
                r=read(a.output/p['latest']['file'])
                if r['frame']==last:time.sleep(.05);continue
                last=r['frame']
                if not spawned:
                    w.wait_for_tick(30)
                    if a.scenario=='signal':
                        lights=list(w.get_actors().filter('traffic.traffic_light'));saved=[(l,l.get_state(),l.is_frozen()) for l in lights]
                        if not saved:raise RuntimeError('No traffic lights')
                        lights[0].freeze(True)
                        for l in lights:l.set_state(carla.TrafficLightState.Red)
                        events.append({'frame':r['frame'],'event':'environment_red_set','lights':len(lights)})
                    else:
                        points=np.array(route['points_xy_m']);i=min(15 if a.scenario in ['blocker','lead_brake'] else 12,len(points)-2)
                        tangent=points[i+1]-points[i];tangent/=np.linalg.norm(tangent);right=np.array([-tangent[1],tangent[0]])
                        xy=points[i]+(right*3 if a.scenario=='pedestrian' else 0)
                        wp=w.get_map().get_waypoint(carla.Location(x=float(xy[0]),y=float(xy[1])))
                        tr=carla.Transform(carla.Location(x=float(xy[0]),y=float(xy[1]),z=wp.transform.location.z+.4),carla.Rotation(yaw=math.degrees(math.atan2(tangent[1],tangent[0]))))
                        bp=w.get_blueprint_library().find((getattr(a,'asset',None) or 'vehicle.audi.a2') if a.scenario in ['blocker','lead_brake'] else 'walker.pedestrian.0001')
                        if a.scenario in ['blocker','lead_brake']:bp.set_attribute('role_name','phase89_test_blocker')
                        actor=w.spawn_actor(bp,tr);owned.append(actor)
                        if a.scenario in ['blocker','lead_brake']:actor.apply_control(carla.VehicleControl(throttle=.35 if a.scenario=='lead_brake' else 0.,brake=0. if a.scenario=='lead_brake' else 1.))
                        else:actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=float(-right[0]),y=float(-right[1])),speed=1.))
                        events.append({'frame':r['frame'],'event':'environment_actor_spawn','scenario':a.scenario,'actor_id_evaluation_only':actor.id,'position_xy_m':xy.tolist()})
                    spawned=True;spawn_time=r['timestamp_s']
                if a.scenario=='lead_brake' and not lead_braked and r['timestamp_s']-spawn_time>=3.:
                    owned[0].apply_control(carla.VehicleControl(brake=1.));lead_braked=True
                    events.append({'frame':r['frame'],'event':'lead_full_brake_after_3s'})
                state='SIGNAL_STOP' if a.scenario=='signal' else 'OBSTACLE_STOP'
                stopped=stopped+1 if r['plan']['state']==state and r['speed_mps']<.15 else 0
                if stopped>=max(3,round(getattr(a,'stop_hold_s',.6)/.2)) and not released and (a.scenario!='lead_brake' or lead_braked):
                    if a.scenario=='signal':
                        for l,_,_ in saved:l.set_state(carla.TrafficLightState.Green)
                        events.append({'frame':r['frame'],'event':'environment_green_set_after_RGB_stop'})
                    elif a.scenario=='blocker':
                        for actor in owned:actor.destroy()
                        owned=[];events.append({'frame':r['frame'],'event':'blocker_removed_after_RGB_stop'})
                    elif a.scenario=='lead_brake' and lead_braked:
                        owned[0].apply_control(carla.VehicleControl(throttle=.4));events.append({'frame':r['frame'],'event':'lead_resumed_after_RGB_stop'})
                    released=True
        except Exception as e:
            errors.append(type(e).__name__+': '+str(e));cancel.set()
        finally:
            for actor in owned:
                try:actor.destroy()
                except Exception as e:errors.append(str(e))
            for light,state,frozen in saved:
                try:light.set_state(state)
                except Exception as e:errors.append(str(e))
            if saved:
                try:saved[0][0].freeze(saved[0][2]);restored.append('original_light_states_and_freeze_restored_elapsed_phase_not_restorable')
                except Exception as e:errors.append(str(e))
    thread=threading.Thread(target=director,daemon=False);thread.start()
    from senior_ai.phase10_observer import EvaluationObserver
    try:report=drive(route,a.output,cancel,audit_sink=EvaluationObserver() if getattr(a,'audit',False) else None)
    finally:done.set();thread.join(40)
    records=[read(a.output/r['file']) for r in report['records']]
    stop_state='SIGNAL_STOP' if a.scenario=='signal' else 'OBSTACLE_STOP'
    stationary_stop=any(r['plan']['state']==stop_state and r['speed_mps']<.15 for r in records)
    reaction=any(r['plan']['state'] in ['FOLLOW','OBSTACLE_STOP'] and any(o['class_id']==2 for o in r['perception']['objects']) for r in records)
    gate=(stationary_stop and released if a.scenario!='pedestrian' else reaction) and report['engineering_route_passed'] and not errors
    write_json(probe/'report.json',{'scenario':a.scenario,'status':'completed_declared_real_RGB_environment_probe','passed':gate,
        'stationary_RGB_stop':stationary_stop,'release_trigger_observed':released,'pedestrian_reaction':reaction,'route_passed':report['engineering_route_passed'],
        'events':events,'errors':errors,'restored':restored,'director_thread_ended':not thread.is_alive(),'formal_quality_passed':False,
        'limits':'Scripted controlled empty-road environment, not varied multi-actor collision/violation-rate certification.'})
    print({'scenario':a.scenario,'passed':gate,'events':events,'errors':errors})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenario',choices=['signal','blocker','pedestrian','lead_brake'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--audit',action='store_true');p.add_argument('--asset');p.add_argument('--stop-hold-s',type=float,default=.6);main(p.parse_args())
