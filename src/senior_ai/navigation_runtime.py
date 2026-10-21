"""Single-owner CARLA adapter. Ego localization is an explicit navigation prior."""
import copy, hashlib, json, math, queue, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
from .core import write_json, intrinsics, receive_frame
from .frames import RgbFrame
from .navigation import RouteCursor, Planner, Controller, validate_route, validate_maneuvers
from .navigation_io import read, sha
from .navigation_lock import TickLease

ROOT=Path(__file__).resolve().parents[2]

def pose_of(t):return [t.location.x,t.location.y,math.radians(t.rotation.yaw)]
def speed_of(v):return float(math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z))


def route_from_request(client, start_xy, goal_xy, config):
    import carla
    sys.path.insert(0,str(ROOT/'artifacts/carla/PythonAPI/carla'))
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    world=client.get_world();m=world.get_map()
    if m.name.split('/')[-1]!=config['map']:raise ValueError('Load declared Town01 before service')
    for xy in [start_xy,goal_xy]:
        if len(xy)!=2 or not np.isfinite(xy).all():raise ValueError('Finite XY endpoints required')
    a=m.get_waypoint(carla.Location(x=float(start_xy[0]),y=float(start_xy[1])),project_to_road=True,lane_type=carla.LaneType.Driving)
    b=m.get_waypoint(carla.Location(x=float(goal_xy[0]),y=float(goal_xy[1])),project_to_road=True,lane_type=carla.LaneType.Driving)
    if not a or not b:raise ValueError('No driving lane near endpoints')
    if any(np.linalg.norm(np.array(x)-[wp.transform.location.x,wp.transform.location.y])>8 for x,wp in [(start_xy,a),(goal_xy,b)]):raise ValueError('Endpoint too far from drivable road')
    trace=GlobalRoutePlanner(m,1.).trace_route(a.transform.location,b.transform.location)
    validate_maneuvers(sorted({o.name for _,o in trace}))
    points=[]
    for wp,_ in trace:
        xy=[wp.transform.location.x,wp.transform.location.y]
        if not points or np.linalg.norm(np.array(xy)-points[-1])>.1:points.append(xy)
    validate_route(points,config['max_route_m'])
    return {'id':'custom','start':{'xyz':[a.transform.location.x,a.transform.location.y,a.transform.location.z+.5],'yaw_deg':a.transform.rotation.yaw},
            'goal_xy_m':points[-1],'points_xy_m':points,'length_m':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
            'turn_options':sorted({o.name for _,o in trace}),'route_prior':'CARLA_OpenDRIVE_not_RGB',
            'training_route_overlap':'not_audited_custom_route_no_guarantee'}


def drive(route, output, cancel, publish=lambda state:None, device='cuda', fault=None):
    import carla
    from .live_perception import LivePerception
    config=read(ROOT/'configs/phase89_navigation_v1.json')
    validate_route(route['points_xy_m'],config['max_route_m'])
    validate_maneuvers(route['turn_options'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'src/senior_ai/navigation.py',ROOT/'src/senior_ai/live_perception.py',ROOT/'src/senior_ai/navigation_lock.py',ROOT/'src/senior_ai/navigation_io.py',ROOT/'configs/phase89_navigation_v1.json']}
    for n in sources:
        p=output/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    write_json(output/'contract.json',{'task':'CARLA_map_RGB_hybrid_closed_loop','sources':sources,'route':route,'config':config,
        'models':read(ROOT/'configs/phase4_bev_v1.json')['models'],'actor_GT_in_planning':False,
        'ego_localization':'simulator_ego_pose_speed_explicit_navigation_prior','test_fault':fault,
        'GT_assets_or_development_lockbox_used':False,'map_route_from_AI':False,'autopilot_or_BasicAgent_used':False})
    client=carla.Client('127.0.0.1',2000);client.set_timeout(30)
    world=client.get_world();actors=[];sensors=[];events={'collisions':[],'lane_invasions':[]};original=None;records=[];engine=None;settings_owned=False;lease=TickLease(ROOT/'artifacts/runs/phase89-controller.lock')
    status='failed';errors=[];started=time.monotonic();cleanup=True;cleanup_speed=None
    state={'status':'starting','run_id':output.name,'route':route,'frame':None,'message':'準備八鏡頭與凍結GPU模型','source_mode':'live_CARLA_map_RGB_hybrid','formal_quality_passed':False}
    publish(copy.deepcopy(state))
    try:
        lease.acquire()
        if client.get_server_version()!='0.9.16' or world.get_map().name.split('/')[-1]!=config['map']:raise ValueError('Wrong CARLA version or map')
        # Do not take over an existing synchronous controller or existing vehicles.
        original=world.get_settings()
        if original.synchronous_mode:raise RuntimeError('World has another synchronous owner')
        world.wait_for_tick(10)
        if world.get_settings().synchronous_mode:raise RuntimeError('World acquired by another synchronous owner')
        if list(world.get_actors().filter('vehicle.*')):raise RuntimeError('Dedicated empty simulation world required')
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=config['dt_s'];settings.no_rendering_mode=False
        settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10
        world.apply_settings(settings);settings_owned=True;world.set_weather(carla.WeatherParameters.ClearNoon)
        t=route['start'];transform=carla.Transform(carla.Location(*t['xyz']),carla.Rotation(yaw=t['yaw_deg']))
        bp=world.get_blueprint_library().find('vehicle.tesla.model3');bp.set_attribute('role_name','senior_AI_RGB_hybrid')
        vehicle=world.spawn_actor(bp,transform);actors.append(vehicle)
        library=world.get_blueprint_library()
        for kind,sink in [('sensor.other.collision',events['collisions']),('sensor.other.lane_invasion',events['lane_invasions'])]:
            sensor=world.spawn_actor(library.find(kind),carla.Transform(),attach_to=vehicle);actors.append(sensor);sensors.append(sensor)
            def record_event(e,target=sink):
                target.append({'frame':e.frame,'markings':[str(m.type) for m in getattr(e,'crossed_lane_markings',[])]})
            sensor.listen(record_event)
        rig=read(ROOT/'configs/camera_rig_v1.json');cal=[];queues={};camera_sensors={}
        for d in rig['cameras']:
            bp=library.find('sensor.camera.rgb')
            for k,v in {'image_size_x':640,'image_size_y':360,'fov':90,'sensor_tick':0}.items():bp.set_attribute(k,str(v))
            tr=carla.Transform(carla.Location(*d['position_m']),carla.Rotation(yaw=d['yaw_deg']))
            sensor=world.spawn_actor(bp,tr,attach_to=vehicle);actors.append(sensor);sensors.append(sensor);sink=queue.Queue(maxsize=8);queues[d['name']]=sink
            camera_sensors[d['name']]=sensor
            def callback(im,q=sink):
                frame=RgbFrame.from_carla(im)
                try:q.put_nowait(frame)
                except queue.Full:
                    try:q.get_nowait()
                    except queue.Empty:pass
                    try:q.put_nowait(frame)
                    except queue.Full:pass
            sensor.listen(callback)
            cal.append({'name':d['name'],'K':intrinsics(640,360,90),'T_ego_from_camera_local':tr.get_matrix()})
        calibration={'optical_from_carla_camera':[[0,1,0],[0,0,-1],[1,0,0]],'cameras':cal}
        write_json(output/'calibration.json',calibration)
        vehicle.apply_control(carla.VehicleControl(brake=1.))
        for _ in range(8):world.tick(30)
        engine=LivePerception(calibration,device)
        cursor=RouteCursor(route['points_xy_m'],config['max_route_m']);planner=Planner(config);controller=Controller();start_sim=world.get_snapshot().timestamp.elapsed_seconds
        # Static junction locations only, never simulator light states. A distant
        # light beyond the requested goal must not stop an unrelated route.
        road_map=world.get_map()
        junction_progress=[float(cursor.distance[i]) for i,p in enumerate(cursor.points)
                           if road_map.get_waypoint(carla.Location(x=float(p[0]),y=float(p[1]))).is_junction]
        blocked_s=0.;stall_s=0.;best_progress=0.
        for step in range(round(config['maximum_simulation_s']/(2*config['dt_s']))):
            if cancel.is_set() or (output/'CANCEL').exists():status='cancelled';break
            if time.monotonic()-started>config['maximum_wall_s']:raise TimeoutError('Wall-time budget reached')
            frame=world.tick(30);snapshot=world.get_snapshot()
            if snapshot.frame!=frame:raise RuntimeError('Another tick owner detected')
            ego=snapshot.find(vehicle.id);capture_pose=pose_of(ego.get_transform());timestamp=snapshot.timestamp.elapsed_seconds
            flush=world.tick(30);current=world.get_snapshot()
            if flush!=frame+1 or current.frame!=flush:raise RuntimeError('Unexpected readback tick lag')
            rgb={};inventory={};folder=output/'frames'/f'{frame:08d}';folder.mkdir(parents=True)
            for n,q in queues.items():
                image=receive_frame(q,frame,15)
                if abs(image.timestamp-timestamp)>1e-5:raise ValueError('RGB timestamp mismatch')
                rgb[n]=np.frombuffer(image.raw_data,np.uint8).reshape(360,640,4)[:,:,2::-1].copy()
                path=folder/(n+'.png');Image.fromarray(rgb[n]).save(path)
                inventory[n]={'file':path.relative_to(output).as_posix(),'sha256':sha(path),'frame':frame}
            prediction,fused=engine.infer(rgb,output.name,frame,timestamp)
            nav=cursor.update(capture_pose);speed=speed_of(current.find(vehicle.id).get_velocity())
            upcoming=[d-nav['progress_m'] for d in junction_progress if -2<=d-nav['progress_m']<=18]
            observation={'valid':True,'timestamp_s':timestamp,'objects':prediction['objects'],'signals':prediction['signals'] if upcoming else [],
                         'held_objects':prediction['dynamics']['held_unobserved_tracks'],
                         'candidate_road_fractions':engine.road_fractions(fused,nav['local_xy_m'],config)}
            if fault=='sensor_loss' and step>=8:observation['valid']=False
            plan=planner.plan(nav,observation,speed,current.timestamp.elapsed_seconds,2*config['dt_s'])
            control=controller.step(plan,speed,2*config['dt_s'])
            if cancel.is_set():control={'throttle':0.,'brake':1.,'steer':0.}
            vehicle.apply_control(carla.VehicleControl(**control))
            record={'schema_version':1,'run_id':output.name,'frame':frame,'timestamp_s':timestamp,'simulation_s':timestamp-start_sim,
                    'source_mode':'live_CARLA_map_RGB_hybrid','image_age_sim_s':current.timestamp.elapsed_seconds-timestamp,
                    'ego_pose_navigation_prior':capture_pose,'speed_mps':speed,'navigation':nav,'plan':plan,'control':control,
                    'perception':prediction,'RGB':inventory,'candidate_road_fractions':observation['candidate_road_fractions'],
                    'map_junction_distance_m':min(upcoming) if upcoming else None,'relevant_RGB_signals':observation['signals'],
                    'other_actor_GT_in_planning':False,'formal_quality_passed':False}
            write_json(folder/'record.json',record)
            records.append({'frame':frame,'file':(folder/'record.json').relative_to(output).as_posix(),'sha256':sha(folder/'record.json'),
                            'state':plan['state'],'speed_mps':speed,'cross_track_m':nav['cross_track_m'],'remaining_m':nav['remaining_m'],
                            'goal_distance_m':nav['goal_distance_m'],'inference_ms':prediction['inference_ms']})
            state.update(status='running',frame=frame,message=plan['reason'],latest=record,completed_frames=len(records))
            publish(copy.deepcopy(state));write_json(output/'progress.json',{'frames':len(records),'latest':records[-1]})
            if fault=='camera_timeout' and step==6:
                camera_sensors['front'].stop()
                write_json(output/'fault-injection.json',{'frame':frame,'mode':'actual_front_RGB_sensor_stopped','planner_validity_flag_modified':False})
            print(json.dumps({'run':output.name,'step':step,'frame':frame,'state':plan['state'],'speed':round(speed,2),'remaining':round(nav['remaining_m'],2),'road':observation['candidate_road_fractions']}),flush=True)
            if events['collisions']:status='collision';break
            if nav['cross_track_m']>config['maximum_cross_track_m']:status='route_deviation';break
            if plan['state']=='ARRIVED':status='arrived';break
            blocked_s=blocked_s+2*config['dt_s'] if speed<.1 and plan['target_speed_mps']==0 else 0.
            if nav['progress_m']>best_progress+.15:best_progress=nav['progress_m'];stall_s=0.
            else:stall_s+=2*config['dt_s']
            if fault and plan['state']=='FAILSAFE' and speed<.1:status='fault_stopped';break
            if max(blocked_s,stall_s)>12 and plan['state']!='SIGNAL_STOP':status='blocked_safe_stop';break
        else:status='simulation_timeout'
    except Exception as e:
        if isinstance(e,TimeoutError) and 'Camera timeout' in str(e):status='sensor_timeout_stopped'
        errors.append(type(e).__name__+': '+str(e));write_json(output/'failure.json',{'error':errors[-1],'traceback':traceback.format_exc()})
    finally:
        if actors:
            try:
                vehicle.apply_control(carla.VehicleControl(brake=1.))
                for _ in range(12):world.tick(10)
                cleanup_speed=speed_of(world.get_snapshot().find(vehicle.id).get_velocity())
            except Exception as e:cleanup=False;errors.append('brake cleanup '+str(e))
        for sensor in reversed(sensors):
            try:
                if sensor.is_listening:sensor.stop()
            except Exception:cleanup=False
        for actor in reversed(actors):
            try:cleanup=actor.destroy() and cleanup
            except Exception:cleanup=False
        if original is not None and settings_owned:
            try:world.apply_settings(original)
            except Exception as e:cleanup=False;errors.append('restore '+str(e))
        if engine is not None:
            del engine
            import gc,torch
            gc.collect();torch.cuda.empty_cache()
        lease.release()
    exact=all(sha(ROOT/n)==h for n,h in sources.items())
    solid=sum(any('Solid' in m for m in e['markings']) for e in events['lane_invasions'])
    checks={'arrived_stopped':status=='arrived','collision_free':not events['collisions'],'no_solid_lane_crossing':solid==0,
            'cross_track_within_1_5m':bool(records) and max(r['cross_track_m'] for r in records)<=config['maximum_cross_track_m'],
            'cleanup':cleanup,'cleanup_stopped':not actors or (cleanup_speed is not None and cleanup_speed<=.3),'source_exact':exact,'no_errors':not errors}
    report={'status':status,'engineering_route_passed':all(checks.values()),'checks':checks,'errors':errors,'events':events,
            'frames':len(records),'records':records,'wall_s':time.monotonic()-started,'route':route,
            'simulation_s':0 if not records else (len(records)*2*config['dt_s']), 'traffic_signal_and_dynamic_actor_GT_compliance':'not_independently_certified',
            'cleanup_final_speed_mps':cleanup_speed,'formal_quality_passed':False,'planning_source':'map_ego_localization_RGB_hybrid_classical','BasicAgent_used':False}
    write_json(output/'report.json',report)
    state.update(status=status,message=errors[-1] if errors else status,report=report)
    publish(copy.deepcopy(state))
    return report
