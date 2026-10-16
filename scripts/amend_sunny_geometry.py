"""Pre-training geometry amendment: add a second pedestrian-capable Town04 group.

No simulator ticks, model scores, or capture successes are read for this choice.
The original frozen plan remains intact and the amendment has a new version.
"""
import json
from pathlib import Path
import carla
from collect_dataset import ROOT, pose, sha
from prepare_motion_dataset import forward_path
from senior_ai.core import write_json


def main():
    source=ROOT/'configs/dataset_sunny_v3_1.json'; output=ROOT/'configs/dataset_sunny_v3_2.json'
    if output.exists(): raise FileExistsError(output)
    plan=json.loads(source.read_text(encoding='utf-8')); scene=next(s for s in plan['scenes'] if s['id']=='town04-01')
    m=carla.Map('Town04',(ROOT/'artifacts/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town04.xodr').read_text(encoding='utf-8'))
    others=[carla.Location(*s['start']['position_m']) for s in plan['scenes'] if s['town']=='Town04' and s['id']!=scene['id']]
    candidates=[]
    driving=m.generate_waypoints(5)
    for wp in driving:
        if wp.lane_type!=carla.LaneType.Driving or wp.is_junction: continue
        if min(wp.transform.location.distance(v) for v in others)<70: continue
        sw=m.get_waypoint(wp.transform.location,lane_type=carla.LaneType.Sidewalk)
        if sw is None or sw.transform.location.distance(wp.transform.location)>12: continue
        try:
            route=forward_path(wp,65,2); path=forward_path(sw,12,1)
        except ValueError: continue
        candidates.append((wp,route,path))
    if not candidates: raise ValueError('No independently separated pedestrian-capable route')
    wp,route,path=max(candidates,key=lambda v:(min(v[0].transform.location.distance(o) for o in others),-v[0].road_id,-v[0].s))
    t=wp.transform; t.location.z+=.3
    scene.update({'group_id':'town04-pedestrian-route-01','start':pose(t),'spawn_index':None,
                  'ego_route_m':[[v.transform.location.x,v.transform.location.y,v.transform.location.z] for v in route],
                  'selection':'Driving waypoint geometry near real sidewalk; farthest from other Town04 frozen origins >=70m, before acquisition/scoring'})
    scene['goal_m']=scene['ego_route_m'][-1]
    parts=[]
    for bp,kind,distance in [('vehicle.bh.crossbike','bicycle',8),('vehicle.kawasaki.ninja','motorcycle',16),
            ('vehicle.lincoln.mkz_2020','car',32),('vehicle.carlamotors.european_hgv','truck',52),('vehicle.mitsubishi.fusorosa','bus',72)]:
        ref=route[distance//2].transform
        choices=[v for v in driving if v.lane_type==carla.LaneType.Driving and 2.8<=v.transform.location.distance(ref.location)<=12
            and abs((v.transform.rotation.yaw-ref.rotation.yaw+180)%360-180)>120]
        selected=min(choices,key=lambda v:v.transform.location.distance(ref.location)) if choices else route[distance//2]
        spawn=selected.transform; spawn.location.z+=.4
        parts.append({'blueprint':bp,'kind':kind,'start':pose(spawn),'controller':'traffic_manager','speed_difference_percent':70})
    spawn=path[0].transform; spawn.location.z+=1
    parts.append({'blueprint':'walker.pedestrian.0015','kind':'pedestrian','start':pose(spawn),'controller':'walker_path','speed_mps':.8,
                  'path_m':[[v.transform.location.x,v.transform.location.y,v.transform.location.z] for v in path]})
    cone=path[2].transform; right=cone.get_right_vector(); cone.location.x+=right.x*.8; cone.location.y+=right.y*.8; cone.location.z+=.05
    parts.append({'blueprint':'static.prop.trafficcone01','kind':'temporary_obstacle','start':pose(cone),'controller':'static'})
    scene['participants']=parts;scene['requested_npcs']=len(parts)
    plan.update({'version':'sunny-cross-town-v3.2','parent_plan_sha256':sha(source),
                 'amendment_reason':'Geometry-only inventory revealed original Town04 had pedestrians in just one group, insufficient for the pre-frozen two-group quota. Replace only uncollected Town04-01 geometry; unchanged groups may be reused with explicit lineage.'})
    write_json(output,plan)
    print(json.dumps({'status':'frozen','new_plan':str(output),'sha256':sha(output),'candidate_count':len(candidates),
                      'scene':scene['id'],'origin':scene['start'],'min_origin_distance_m':min(wp.transform.location.distance(o) for o in others)}))


if __name__=='__main__': main()
