"""Cross-town sunny 32-sensor feasibility and blueprint taxonomy evidence.

Static interventions are engineering fixtures, explicitly never training data.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import queue
import sys
import time

import numpy as np

from collect_dataset import ROOT, pose, xyz, sha
from senior_ai.core import write_json, intrinsics, receive_frame
from senior_ai.frames import RgbFrame

ASSETS = [('vehicle.carlamotors.european_hgv','truck',[14,0,.5]),
          ('vehicle.mitsubishi.fusorosa','bus',[-16,0,.5]),
          ('static.prop.trafficcone01','temporary_obstacle',[7,2,.2])]


def probe(client, town, root):
    import carla
    root.mkdir(parents=True, exist_ok=False)
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != town:
        world = client.load_world(town)
    original = world.get_settings()
    actors, sensors, queues = [], [], {}
    report = {'town':town,'status':'failed','training_approved':False, 'errors':[]}
    write_json(root/'report.json',report)
    start = time.monotonic()
    try:
        settings=world.get_settings(); settings.synchronous_mode=True; settings.fixed_delta_seconds=.05; settings.no_rendering_mode=False
        world.apply_settings(settings); world.set_weather(carla.WeatherParameters.ClearNoon)
        for _ in range(10): world.tick(30)
        road_map=world.get_map(); spawns=road_map.get_spawn_points(); library=world.get_blueprint_library()
        origin=spawns[0]
        ego=world.spawn_actor(library.find('vehicle.tesla.model3'),origin); actors.append(ego); ego.set_simulate_physics(False)
        targets=[]
        for blueprint,kind,offset in ASSETS:
            t=carla.Transform(origin.transform(carla.Location(*offset)),origin.rotation)
            # This static taxonomy fixture may overlap map furniture. Spawn in
            # free air first, disable physics, then place at the recorded pose.
            # This is not a motion or collision-free scenario acceptance.
            air=carla.Transform(carla.Location(t.location.x,t.location.y,t.location.z+20),t.rotation)
            a=world.spawn_actor(library.find(blueprint),air); actors.append(a); a.set_simulate_physics(False); a.set_transform(t)
            targets.append((a,kind))
        rig=json.loads((ROOT/'configs/camera_rig_v1.json').read_text(encoding='utf-8'))
        for cam in rig['cameras']:
            local=carla.Transform(carla.Location(*cam['position_m']),carla.Rotation(yaw=cam['yaw_deg']))
            for modality in ['rgb','semantic_segmentation','depth','instance_segmentation']:
                bp=library.find('sensor.camera.'+modality)
                for k,v in {'image_size_x':rig['width'],'image_size_y':rig['height'],'fov':rig['horizontal_fov_deg'],'sensor_tick':0}.items(): bp.set_attribute(k,str(v))
                a=world.spawn_actor(bp,local,attach_to=ego); actors.append(a); sensors.append(a)
                sink=queue.Queue(); queues[(cam['name'],modality)]=sink
                a.listen(lambda image, sink=sink: sink.put(RgbFrame.from_carla(image)))
        for _ in range(30): world.tick(30)
        frame=world.tick(30); snapshot=world.get_snapshot()
        target_meta=[]
        for a,kind in targets:
            t=snapshot.find(a.id).get_transform(); b=a.bounding_box
            target_meta.append({'actor_id':a.id,'blueprint':a.type_id,'kind':kind,'semantic_tags':list(a.semantic_tags),
                'world_pose':pose(t),'extent_m':xyz(b.extent),'T_world_from_box':(np.array(t.get_matrix()) @ np.array(carla.Transform(b.location,b.rotation).get_matrix())).tolist(),
                'world_vertices_m':[xyz(v) for v in b.get_world_vertices(t)]})
        world.tick(30)
        hist={str(a.id):Counter() for a,_ in targets}; views={}; inventory=[]
        for (name,modality),sink in queues.items():
            image=receive_frame(sink,frame,30)
            if abs(image.timestamp-snapshot.timestamp.elapsed_seconds)>1e-4: raise ValueError('Timestamp mismatch')
            relative=f'{modality}/{name}.png'; path=root/relative; path.parent.mkdir(parents=True,exist_ok=True); image.save_to_disk(path)
            inventory.append({'path':relative,'sha256':sha(path),'bytes':path.stat().st_size})
            views.setdefault(name,{'K':intrinsics(rig['width'],rig['height'],rig['horizontal_fov_deg']),
                                  'T_world_from_camera':image.world_from_camera})[modality]=relative
            if modality=='instance_segmentation':
                raw=np.frombuffer(image.raw_data,dtype=np.uint8).reshape(image.height,image.width,4)
                ids=raw[...,1].astype(np.uint16)+256*raw[...,0].astype(np.uint16)
                for a,_ in targets:
                    tags,counts=np.unique(raw[...,2][ids==a.id],return_counts=True)
                    hist[str(a.id)].update({int(t):int(n) for t,n in zip(tags,counts)})
        lights=[]
        for a in world.get_actors().filter('traffic.traffic_light'):
            lights.append({'id':a.id,'pose':pose(a.get_transform()),'state':str(a.get_state()),
                'boxes':[{'pose':pose(carla.Transform(b.location,b.rotation)), 'extent_m':xyz(b.extent),
                          'world_vertices_m':[xyz(v) for v in b.get_world_vertices(carla.Transform())]} for b in a.get_light_boxes()]})
        report.update({'map_name':road_map.name,'spawn_count':len(spawns),'waypoint_count':len(road_map.generate_waypoints(5)),
            'opendrive_sha256':__import__('hashlib').sha256(road_map.to_opendrive().encode()).hexdigest(),
            'targets':target_meta,'instance_pixels':{k:dict(v) for k,v in hist.items()},'views':views,'lights':lights,
            'image_inventory':inventory,'sensors':len(sensors),'frame':frame,'weather':'ClearNoon',
            'gpu_mib_end_sample':__import__('subprocess').check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()})
        if len(sensors)!=32 or any(not v for v in hist.values()): raise ValueError('Missing asset visible evidence')
        report['status']='passed'
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        for a in reversed(sensors): a.stop()
        for a in reversed(actors): a.destroy()
        world.apply_settings(original)
        report['wall_s']=time.monotonic()-start; report['source_sha256']=sha(Path(__file__))
        write_json(root/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['views','lights','image_inventory']}),flush=True)
    return report['status']=='passed'


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--towns',nargs='+',default=['Town03','Town04','Town05','Town02','Town01']); args=p.parse_args()
    import carla
    c=carla.Client('127.0.0.1',2000); c.set_timeout(120)
    for town in args.towns:
        if not probe(c,town,args.output/town): sys.exit(1)
