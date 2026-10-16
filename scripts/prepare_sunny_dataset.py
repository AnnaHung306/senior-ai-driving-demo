"""Freeze cross-town routes and actor poses using geometry only, before capture."""
import argparse
import hashlib
import json
from pathlib import Path

from collect_dataset import ROOT, pose, sha
from prepare_motion_dataset import forward_path
from senior_ai.core import write_json

ALLOCATION=[('Town01','train',4),('Town03','train',4),('Town04','validation',3),('Town05','development',3),('Town02','lockbox',4)]


def main(args):
    import carla
    if args.output.exists(): raise FileExistsError('Frozen plan exists')
    c=carla.Client('127.0.0.1',2000); c.set_timeout(120)
    plan={'version':'sunny-cross-town-v3.1','scope':'ClearNoon only by user instruction; whole-town splits; 30-second moving episodes',
          'weather':'ClearNoon','rig_sha256':sha(ROOT/'configs/camera_rig_v1.json'), 'maps':{},'scenes':[],
          'selection':'Geometry only; longest non-junction candidates first then farthest point sampling; no model scoring or capture-based reselection',
          'criteria':{'bundles_per_scene':150,'fixed_delta_s':.05,'capture_stride_ticks':4,'sensor_warmup_ticks':30,
              'readback_lag_ticks':1,'max_timestamp_error_s':.0001,'max_scene_wall_s':1800,'target_speed_kph':12,
              'min_ego_displacement_m':5,'min_label_duration_s':29.5,'min_participant_displacement_m':1,
              'max_ego_collisions':0,'modalities':['rgb','semantic_segmentation','depth','instance_segmentation'],
              'static_controller_motion_exempt':True},
          'coverage_quotas':{'per_object_class_positive_views':10,'per_object_class_positive_groups':2,
                             'per_semantic_focus_class_pixels':1000,'per_semantic_focus_class_positive_views':10},
          'quality_gates':{'detection_mAP50':.5,'detection_per_class_recall_at_score_0_5':.5,
                           'segmentation_focus_mIoU':.5,'segmentation_per_focus_class_IoU':.2},
          'quality_gate_note':'Frozen before new acquisition and heldout scoring. Failure is reported, never relabeled a pass. Ideal simulated daylight only.'}
    for town,split,count in ALLOCATION:
        w=c.get_world()
        if w.get_map().name.split('/')[-1]!=town: w=c.load_world(town)
        m=w.get_map(); lib=w.get_blueprint_library(); candidates=[]
        driving=m.generate_waypoints(2)
        for index,t in enumerate(m.get_spawn_points()):
            wp=m.get_waypoint(t.location)
            if wp.is_junction: continue
            try:
                route=forward_path(wp,65,2)
                # Geometry feasibility for all frozen opposing-lane slots.
                opposite=[]
                for distance in [16,32,52,72]:
                    ref=route[distance//2].transform
                    choices=[v for v in driving if 2.8<=v.transform.location.distance(ref.location)<=12
                        and abs((v.transform.rotation.yaw-ref.rotation.yaw+180)%360-180)>120]
                    # One-way roads are valid too. Use a spaced forward convoy
                    # if no opposing driving lane exists; record the exact pose.
                    opposite.append(min(choices,key=lambda v:(v.transform.location.distance(ref.location),v.road_id,v.s))
                                    if choices else route[distance//2])
                anchor=route[5].transform; right=anchor.get_right_vector(); sidewalks=[]
                for side in [-1,1]:
                    query=carla.Location(anchor.location.x+side*6*right.x,anchor.location.y+side*6*right.y,anchor.location.z)
                    sw=m.get_waypoint(query,lane_type=carla.LaneType.Sidewalk)
                    if sw is None or sw.transform.location.distance(query)>6: continue
                    pedestrian_path=[sw]
                    for _ in range(35):
                        options=pedestrian_path[-1].next(1)
                        if not options: break
                        pedestrian_path.append(min(options,key=lambda v:(v.road_id,v.lane_id,v.s)))
                    if len(pedestrian_path)<10: continue
                    sidewalks.append(pedestrian_path)
                candidates.append((index,t,route,opposite,sidewalks))
            except ValueError: continue
        if not candidates: raise ValueError(f'No geometry candidates {town}')
        chosen=[max(candidates,key=lambda v:(len(v[4]),-v[0]))]
        while len(chosen)<count:
            remaining=[v for v in candidates if v[0] not in {p[0] for p in chosen}]
            best=max(remaining,key=lambda v:min(v[1].location.distance(p[1].location) for p in chosen))
            if min(best[1].location.distance(p[1].location) for p in chosen)<70: raise ValueError('Insufficient separated origins')
            chosen.append(best)
        plan['maps'][town]={'opendrive_sha256':hashlib.sha256(m.to_opendrive().encode()).hexdigest(),
            'spawn_count':len(m.get_spawn_points()),'geometry_candidate_count':len(candidates)}
        walkers=sorted(lib.filter('walker.pedestrian.*'),key=lambda b:b.id)
        for j,(index,t,route,opposite,sidewalks) in enumerate(chosen):
            scene={'id':f'{town.lower()}-{j:02d}','town':town,'group_id':f'{town.lower()}-route-{j:02d}',
                   'split':split,'weather':'ClearNoon','seed':20261016+len(plan['scenes'])*101,'spawn_index':index,
                   'start':pose(t),'ego_route_m':[xyz_waypoint(p) for p in route], 'goal_m':xyz_waypoint(route[-1]),'participants':[]}
            definitions=[('vehicle.bh.crossbike','bicycle',route[4]),
                         ('vehicle.kawasaki.ninja','motorcycle',opposite[0]),
                         ('vehicle.lincoln.mkz_2020','car',opposite[1]),
                         ('vehicle.carlamotors.european_hgv','truck',opposite[2]),
                         ('vehicle.mitsubishi.fusorosa','bus',opposite[3])]
            for bp,kind,wp in definitions:
                lib.find(bp); spawn=wp.transform; spawn.location.z+=.4
                scene['participants'].append({'blueprint':bp,'kind':kind,'start':pose(spawn),'controller':'traffic_manager','speed_difference_percent':70})
            for k,path in enumerate(sidewalks):
                spawn=path[0].transform; spawn.location.z+=1
                scene['participants'].append({'blueprint':walkers[(j*2+k+len(plan['scenes'])*2)%len(walkers)].id,
                    'kind':'pedestrian','start':pose(spawn),'controller':'walker_path','speed_mps':.8,'path_m':[xyz_waypoint(v) for v in path]})
            # Verified temporary-obstacle instance; broad Dynamic is not enough.
            sw=(sidewalks[0][2] if sidewalks else route[6]).transform; sw.location.z+=.05
            right=sw.get_right_vector(); offset=.8 if sidewalks else 2.5
            sw.location.x+=right.x*offset; sw.location.y+=right.y*offset
            scene['participants'].append({'blueprint':'static.prop.trafficcone01','kind':'temporary_obstacle',
                'start':pose(sw),'controller':'static'})
            scene['requested_npcs']=len(scene['participants']); plan['scenes'].append(scene)
        print(json.dumps({'town':town,'frozen_groups':count,'candidates':len(candidates)}),flush=True)
    write_json(args.output,plan)
    print(json.dumps({'status':'frozen','sha256':sha(args.output),'groups':len(plan['scenes'])}),flush=True)


def xyz_waypoint(w):
    return [w.transform.location.x,w.transform.location.y,w.transform.location.z]


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); main(p.parse_args())
