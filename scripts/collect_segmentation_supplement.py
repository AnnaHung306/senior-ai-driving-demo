"""Capture one frozen training town's static RGB and three GT modalities."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import traceback
import time
import numpy as np
from collect_light_supplement import camera_transform,aligned_triplet
from collect_dataset import ROOT,sha
from senior_ai.frames import RgbFrame
from senior_ai.core import write_json,intrinsics


def collect(args):
    import carla
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    if args.town not in {'Town01','Town03'} or plan['split']!='train' or plan['task']!='segmentation_only' or plan['weather']!='ClearNoon':raise ValueError('Train-only sunny scenery required')
    if sha(ROOT/'data/sunny-cross-town-v3-007/audit.json')!=plan['source_audit_sha256']:raise ValueError('Main audit changed')
    root=args.output;town_root=root/args.town
    if town_root.exists():raise FileExistsError('Use a new capture version after failure')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists() and sha(root/'plan.json')!=sha(args.plan):raise ValueError('Different plan')
    (root/'plan.json').write_bytes(args.plan.read_bytes());town_root.mkdir()
    report={'status':'failed','town':args.town,'plan_sha256':sha(args.plan),'collector_sha256':sha(Path(__file__)),'records':0,'cleanup':False,'training_approved':False}
    report['probe_only']=args.probe_anchors is not None
    report['wall_delay_after_tick_s']=args.tick_wall_delay
    report['max_exact_frame_attempts_per_view']=24
    if not .05<=args.tick_wall_delay<=.2:raise ValueError('Static capture wall pacing must be .05..0.2s')
    source_files=sorted((ROOT/'src').rglob('*.py'))+[ROOT/'scripts'/name for name in
        ['collect_segmentation_supplement.py','collect_light_supplement.py','collect_dataset.py','rig_capture.py']]
    report['source_files']={p.relative_to(ROOT).as_posix():sha(p) for p in source_files}
    for source in source_files:
        destination=town_root/'source_snapshot'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.read_bytes())
    write_json(town_root/'report.json',report)
    client=carla.Client('127.0.0.1',2000);client.set_timeout(60);world=client.get_world();road_map=world.get_map()
    if road_map.name.split('/')[-1]!=args.town:raise ValueError('Prepare map in a separate client')
    if hashlib.sha256(road_map.to_opendrive().encode()).hexdigest()!=plan['towns'][args.town]['opendrive_sha256']:raise ValueError('Map changed')
    if list(world.get_actors().filter('vehicle.*')) or list(world.get_actors().filter('walker.*')):raise ValueError('Static scenery capture requires an empty world')
    original,weather=world.get_settings(),world.get_weather();sensors=[];queues={};records=[]
    rig=plan['rig'];actual={}
    try:
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=False;world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        for modality,bpname in [('rgb','rgb'),('semantic','semantic_segmentation'),('depth','depth'),('instance','instance_segmentation')]:
            bp=world.get_blueprint_library().find('sensor.camera.'+bpname)
            for k,v in {'image_size_x':rig['width'],'image_size_y':rig['height'],'fov':rig['horizontal_fov_deg'],'sensor_tick':0.0}.items():bp.set_attribute(k,str(v))
            anchor=camera_transform(plan['towns'][args.town]['anchors'][0]['native_transform_matrix'])
            sensor=world.spawn_actor(bp,anchor);sensors.append(sensor);actual[modality]=dict(sensor.attributes)
            q=queue.Queue(maxsize=8);queues[modality]=q
            def callback(image,target=q):
                owned=RgbFrame.from_carla(image)
                if target.full():
                    try:target.get_nowait()
                    except queue.Empty:pass
                target.put_nowait(owned)
            sensor.listen(callback)
        write_json(town_root/'calibration.json',{'K':intrinsics(rig['width'],rig['height'],rig['horizontal_fov_deg']),
            'sensor_attributes':actual,'weather':str(world.get_weather()),'map':road_map.name,'opendrive_sha256':plan['towns'][args.town]['opendrive_sha256'],
            'coordinate_scope':'Virtual rig anchor at native road spawn; no vehicle or route movement','server':client.get_server_version()})
        anchors=plan['towns'][args.town]['anchors']
        if args.probe_anchors is not None:
            if not 1<=args.probe_anchors<len(anchors):raise ValueError('Probe must be a strict subset of frozen anchors')
            anchors=anchors[:args.probe_anchors]
        for a in anchors:
            anchor=camera_transform(a['native_transform_matrix']);group=f'{args.town.lower()}-static-anchor-{a["spawn_index"]:03}'
            for camera in rig['cameras']:
                local=carla.Transform(carla.Location(*camera['position_m']),carla.Rotation(yaw=camera['yaw_deg']))
                expected=np.asarray(anchor.get_matrix())@np.asarray(local.get_matrix())
                for sensor in sensors:sensor.set_transform(camera_transform(expected))
                for _ in range(12):world.tick(20);time.sleep(args.tick_wall_delay)
                frame,snapshot,images,rejected=aligned_triplet(world,queues,max_attempts=24,wall_delay_s=args.tick_wall_delay)
                row={'split':'train','task':'segmentation_only','town':args.town,'group_id':group,'scene_id':group,'camera':camera['name'],
                    'anchor_spawn_index':a['spawn_index'],'frame':frame,'timestamp_s':snapshot.timestamp.elapsed_seconds,
                    'T_world_from_camera':expected.tolist(),'verified_obstacle_actor_ids':[],'images':{},'rejected_capture_attempts':rejected}
                for modality,image in images.items():
                    p=root/f'raw/{group}/{camera["name"]}/{modality}.png';p.parent.mkdir(parents=True,exist_ok=True);image.save_to_disk(p)
                    row['images'][modality]={'path':p.relative_to(root).as_posix(),'frame':image.frame,'timestamp_s':image.timestamp,'T_world_from_camera':image.world_from_camera}
                records.append(row);write_json(town_root/'records.json',records)
            report['records']=len(records);write_json(town_root/'report.json',report)
            print(json.dumps({'town':args.town,'anchor':a['spawn_index'],'views':len(records)}),flush=True)
        report['status']='passed'
    except Exception as exc:
        report.update(error=str(exc),traceback=traceback.format_exc());raise
    finally:
        for sensor in sensors:sensor.stop();sensor.destroy()
        # Keep the user-requested ideal sunny weather; restore time/render settings.
        world.set_weather(carla.WeatherParameters.ClearNoon);world.apply_settings(original)
        report.update(cleanup=True,records=len(records));write_json(town_root/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--town',choices=['Town01','Town03'],required=True)
    p.add_argument('--probe-anchors',type=int)
    p.add_argument('--tick-wall-delay',type=float,default=.05)
    p.add_argument('--output',type=Path,required=True);collect(p.parse_args())
