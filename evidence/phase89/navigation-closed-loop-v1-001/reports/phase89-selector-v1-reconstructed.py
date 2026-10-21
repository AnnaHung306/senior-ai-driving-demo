"""Freeze map geometry and new demonstration routes before any driving result."""
import sys, json, hashlib, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/carla/PythonAPI/carla'))
sys.path.insert(0,str(ROOT/'src'))
from senior_ai.core import write_json
from senior_ai.world_display import sha, read

def main():
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    out=ROOT/'artifacts/runs/phase89-route-catalog-v1-001'
    if out.exists():raise FileExistsError(out)
    c=carla.Client('127.0.0.1',2000);c.set_timeout(60)
    w=c.get_world()
    if w.get_map().name.split('/')[-1]!='Town01':w=c.load_world('Town01')
    m=w.get_map();grp=GlobalRoutePlanner(m,1.)
    spawns=m.get_spawn_points();nodes=[]
    for i,t in enumerate(spawns):
        wp=m.get_waypoint(t.location)
        nodes.append({'id':i,'xy_m':[t.location.x,t.location.y],'z_m':t.location.z,'yaw_deg':t.rotation.yaw,'road_id':wp.road_id,'lane_id':wp.lane_id})
    def pack(start,goal,tag):
        trace=grp.trace_route(start.location,goal)
        points=[]
        for wp,option in trace:
            xy=[wp.transform.location.x,wp.transform.location.y]
            if not points or np.linalg.norm(np.array(xy)-points[-1])>.1:points.append(xy)
        d=np.linalg.norm(np.diff(points,axis=0),axis=1)
        if (d>4).any() or not 20<=d.sum()<=100:return None
        return {'id':tag,'start':{'xyz':[start.location.x,start.location.y,start.location.z], 'yaw_deg':start.rotation.yaw},
                'goal_xy_m':points[-1],'points_xy_m':points,'length_m':float(d.sum()),
                'turn_options':sorted({o.name for _,o in trace}), 'route_prior':'CARLA_OpenDRIVE_not_RGB',
                'training_route_overlap':'not_yet_audited_no_unseen_claim'}
    oldroads={r['road_id'] for r in read(ROOT/'configs/phase1_scenarios_v1.json')['routes']}
    straights=[];used=set()
    for i,t in enumerate(spawns):
        wp=m.get_waypoint(t.location)
        if wp.is_junction or wp.road_id in oldroads or wp.road_id in used:continue
        path=[wp]
        for _ in range(35):
            following=path[-1].next(1.)
            if len(following)!=1 or following[0].is_junction or following[0].road_id!=wp.road_id:break
            path.append(following[0])
        if len(path)<31:continue
        r=pack(t,path[30].transform.location,f'straight-{len(straights)+1}')
        if r:
            r['spawn_index']=i;straights.append(r);used.add(wp.road_id)
        if len(straights)==2:break
    turning=None
    for i,t in enumerate(spawns):
        if m.get_waypoint(t.location).is_junction:continue
        for j,g in enumerate(spawns):
            if t.location.distance(g.location)<25 or t.location.distance(g.location)>65:continue
            r=pack(t,g.location,'turn-1')
            if r and any(o in r['turn_options'] for o in ['LEFT','RIGHT']) and 35<r['length_m']<80:
                r.update(spawn_index=i,goal_spawn_index=j);turning=r;break
        if turning:break
    if len(straights)!=2 or turning is None:raise RuntimeError('Required map-only route cases unavailable')
    roads=[]
    for a,b in m.get_topology():
        points=[[a.transform.location.x,a.transform.location.y]];current=a
        for _ in range(300):
            nxt=current.next(3.)
            same=[p for p in nxt if p.road_id==a.road_id and p.lane_id==a.lane_id]
            if not same:break
            current=same[0];points.append([current.transform.location.x,current.transform.location.y])
            if current.transform.location.distance(b.transform.location)<4:break
        if len(points)>1:roads.append(points)
    out.mkdir(parents=True)
    write_json(out/'catalog.json',{'map':m.name,'opendrive_sha256':hashlib.sha256(m.to_opendrive().encode()).hexdigest(),
        'config_sha256':sha(ROOT/'configs/phase89_navigation_v1.json'),'source_sha256':sha(Path(__file__)),
        'selection':'First two qualifying non-Phase1 straight roads and first qualifying left/right map route, before driving; no perception scores used.',
        'spawn_points':nodes,'roads_xy_m':roads,'routes':straights+[turning],'GT_actors_used':False,
        'new_unseen_route_quality_passed':False})
    print({'status':'frozen_before_driving','routes':[(r['id'],r['length_m'],r['spawn_index']) for r in straights+[turning]],'spawns':len(nodes)})
if __name__=='__main__':main()
