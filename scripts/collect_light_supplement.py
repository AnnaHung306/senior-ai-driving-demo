"""Fresh controlled phase coverage; frozen lockbox viewpoints, never train data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import queue
import sys
import traceback

from collect_dataset import ROOT, sha, pose, xyz
import numpy as np
from senior_ai.core import receive_frame, write_json, intrinsics
from senior_ai.frames import RgbFrame


def camera_transform(matrix):
    import carla
    m=np.asarray(matrix)
    rotation=carla.Rotation(pitch=math.degrees(math.atan2(m[2,0],math.hypot(m[0,0],m[1,0]))),
        yaw=math.degrees(math.atan2(m[1,0],m[0,0])),roll=math.degrees(math.atan2(-m[2,1],m[2,2])))
    result=carla.Transform(carla.Location(*m[:3,3]),rotation)
    if np.abs(np.asarray(result.get_matrix())-m).max()>.0001:raise ValueError('Pose reconstruction differs')
    return result


def collect(args):
    import carla
    if args.output.exists():raise FileExistsError('Use a new capture version')
    plan=json.loads(args.plan.read_text(encoding='utf-8'));root=args.output;root.mkdir(parents=True)
    if plan['split']!='lockbox' or plan['weather']!='ClearNoon':raise ValueError('Only frozen sunny heldout protocol supported')
    if sha(ROOT/plan['source_root']/'audit.json')!=plan['source_audit_sha256']:raise ValueError('Source audit changed')
    (root/'plan.json').write_bytes(args.plan.read_bytes())
    report={'status':'failed','task':'controlled-phase-supplement-v1','plan_sha256':sha(args.plan),
            'source_sha256':sha(Path(__file__)),'training_approved':False,'records':0,'cleanup':False}
    write_json(root/'report.json',report)
    client=carla.Client('127.0.0.1',2000);client.set_timeout(120)
    if client.get_server_version()!='0.9.16':raise ValueError('Wrong CARLA version')
    world=client.get_world()
    if world.get_map().name.split('/')[-1]!=plan['town']:world=client.load_world(plan['town'])
    if hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest()!=plan['opendrive_sha256']:raise ValueError('Map changed')
    original,weather=world.get_settings(),world.get_weather();sensors=[];queues={};records=[]
    lights=list(world.get_actors().filter('traffic.traffic_light*'))
    saved=[(a,a.get_state(),a.is_frozen()) for a in lights]
    if len({f for _,_,f in saved})>1:raise ValueError('Global freeze restore requires uniform original flags')
    try:
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=False;world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        if lights:lights[0].freeze(True)
        for _ in range(20):world.tick(20)
        lamps=[]
        for light in lights:
            for i,b in enumerate(light.get_light_boxes()):
                lamps.append({'actor_id':light.id,'lamp_index':i,'world_pose':pose(carla.Transform(b.location,b.rotation)),
                    'extent_m':xyz(b.extent),'world_vertices_m':[xyz(v) for v in b.get_world_vertices(carla.Transform())]})
        write_json(root/'lamps.json',lamps)
        rig=plan['rig'];actual={}
        for modality,blueprint in [('rgb','sensor.camera.rgb'),('semantic','sensor.camera.semantic_segmentation'),('depth','sensor.camera.depth')]:
            bp=world.get_blueprint_library().find(blueprint)
            for key,value in {'image_size_x':rig['width'],'image_size_y':rig['height'],'fov':rig['horizontal_fov_deg'],'sensor_tick':0.0}.items():bp.set_attribute(key,str(value))
            sensor=world.spawn_actor(bp,camera_transform(plan['poses'][0]['T_world_from_camera']));sensors.append(sensor);actual[modality]=dict(sensor.attributes)
            q=queue.Queue(maxsize=8);queues[modality]=q
            def callback(image,target=q):
                owned=RgbFrame.from_carla(image)
                if target.full():
                    try:target.get_nowait()
                    except queue.Empty:pass
                target.put_nowait(owned)
            sensor.listen(callback)
        write_json(root/'calibration.json',{'K':intrinsics(rig['width'],rig['height'],rig['horizontal_fov_deg']),
            'sensor_attributes':actual,'server':client.get_server_version(),'weather':str(world.get_weather())})
        for item in plan['poses']:
            for sensor in sensors:sensor.set_transform(camera_transform(item['T_world_from_camera']))
            for _ in range(20):world.tick(20)
            for state in plan['phases']:
                for light in lights:light.set_state(getattr(carla.TrafficLightState,state))
                for _ in range(4):world.tick(20)
                frame=world.tick(20);snapshot=world.get_snapshot();observed={str(a.id):str(a.get_state()).split('.')[-1] for a in lights}
                if any(v!=state for v in observed.values()) or any(not a.is_frozen() for a in lights):raise ValueError('Phase program did not persist')
                world.tick(20)  # GPU camera F readback occurs after the next tick.
                row={**item,'phase':state,'frame':frame,'timestamp_s':snapshot.timestamp.elapsed_seconds,'states':observed,
                     'all_lights_frozen':all(a.is_frozen() for a in lights),'images':{}}
                for modality,q in queues.items():
                    image=receive_frame(q,frame,30)
                    if abs(image.timestamp-row['timestamp_s'])>.0001:raise ValueError('Timestamp mismatch')
                    p=root/f'raw/{item["id"]}/{state}/{modality}.png';p.parent.mkdir(parents=True,exist_ok=True);image.save_to_disk(p)
                    row['images'][modality]={'path':p.relative_to(root).as_posix(),'frame':image.frame,'timestamp_s':image.timestamp,'T_world_from_camera':image.world_from_camera}
                records.append(row);write_json(root/'records.json',records)
            print(json.dumps({'pose':item['id'],'records':len(records)}),flush=True)
        report.update(status='passed',records=len(records))
    except Exception as exc:
        report.update(error=str(exc),traceback=traceback.format_exc());raise
    finally:
        for sensor in sensors:
            sensor.stop();sensor.destroy()
        for light,state,frozen in saved:light.set_state(state)
        if lights:lights[0].freeze(saved[0][2])
        world.set_weather(weather);world.apply_settings(original);report['cleanup']=True;write_json(root/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);collect(p.parse_args())
