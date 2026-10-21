"""Select six map-only audit routes before observing any Phase10 prediction."""
import hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'artifacts/carla/PythonAPI/carla'))
from senior_ai.core import write_json
from senior_ai.navigation_io import read,sha
from senior_ai.navigation import validate_route,validate_maneuvers


def main():
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    out=ROOT/'artifacts/runs/phase10-route-catalog-v1-001'
    if out.exists():raise FileExistsError(out)
    config=read(ROOT/'configs/phase10_audit_v1.json');rng=np.random.default_rng(config['seed'])
    client=carla.Client('127.0.0.1',2000);client.set_timeout(60);m=client.get_world().get_map()
    assert m.name.split('/')[-1]=='Town01'
    grp=GlobalRoutePlanner(m,1.);spawns=m.get_spawn_points()
    old=read(ROOT/'configs/phase89_routes_v1.json');old_points=np.array([p for r in old['routes'] for p in r['points_xy_m']])
    old_starts={r['spawn_index'] for r in old['routes']}
    train=[];train_sha={};base=ROOT/'data/sunny-cross-town-v3-007'
    for scene in sorted((base/'scenes').glob('town01-*')):
        rows=read(scene/'index.json')
        if rows and rows[0]['split']=='train':
            for row in rows:
                train.append(read(base/row['label'])['ego_world_pose']['position_m'][:2]);train_sha[row['label']]=sha(base/row['label'])
    train=np.array(train);routes=[];used=set(old_starts)
    def pack(i,goal,kind):
        trace=grp.trace_route(spawns[i].location,goal);options=sorted({o.name for _,o in trace})
        try:validate_maneuvers(options)
        except ValueError:return None
        if kind=='straight' and any(x in options for x in ['LEFT','RIGHT']):return None
        if kind!='straight' and (kind not in options or ({'LEFT','RIGHT'}-{kind})&set(options)):return None
        p=[]
        for wp,_ in trace:
            xy=[wp.transform.location.x,wp.transform.location.y]
            if not p or np.linalg.norm(np.array(xy)-p[-1])>.1:p.append(xy)
        try:points=validate_route(p,config['route_length_m'][1])
        except ValueError:return None
        length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        if length<config['route_length_m'][0]:return None
        t=spawns[i]
        return {'id':f'phase10-{kind.lower()}-{1+sum(r["kind"]==kind for r in routes)}','kind':kind,
            'spawn_index':int(i),'start':{'xyz':[t.location.x,t.location.y,t.location.z],'yaw_deg':t.rotation.yaw},
            'goal_xy_m':p[-1],'points_xy_m':p,'length_m':length,'turn_options':options,'route_prior':'CARLA_OpenDRIVE_not_RGB',
            'minimum_distance_to_007_train_ego_m':float(np.linalg.norm(points[:,None,:]-train[None,:,:],axis=-1).min()),
            'minimum_distance_to_phase89_test_route_m':float(np.linalg.norm(points[:,None,:]-old_points[None,:,:],axis=-1).min()),
            'heldout_generalization_claim':False}
    order=rng.permutation(len(spawns)).tolist()
    for kind,count in config['route_counts'].items():
        for i in order:
            if i in used or m.get_waypoint(spawns[i].location).is_junction:continue
            if kind=='straight':
                wp=m.get_waypoint(spawns[i].location);path=[wp]
                for _ in range(55):
                    nxt=path[-1].next(1.)
                    if len(nxt)!=1 or nxt[0].is_junction or nxt[0].road_id!=wp.road_id:break
                    path.append(nxt[0])
                candidate=pack(i,path[-1].transform.location,kind) if len(path)>=39 else None
            else:
                candidate=None
                for j in order:
                    if 25<=spawns[i].location.distance(spawns[j].location)<=65:
                        candidate=pack(i,spawns[j].location,kind)
                        if candidate: candidate['goal_spawn_index']=int(j);break
            if candidate:routes.append(candidate);used.add(i)
            if sum(r['kind']==kind for r in routes)==count:break
        if sum(r['kind']==kind for r in routes)!=count:raise RuntimeError('Map selection unavailable: '+kind)
    out.mkdir(parents=True)
    write_json(out/'catalog.json',{'map':m.name,'opendrive_sha256':hashlib.sha256(m.to_opendrive().encode()).hexdigest(),
        'routes':routes,'selection':'seeded map-only first qualifying distinct non-Phase89 starts; 35-90m; no prediction/success selection',
        'config_sha256':sha(ROOT/'configs/phase10_audit_v1.json'),'selector_sha256':sha(Path(__file__)),
        'training_metadata_selection_only':train_sha,'other_actor_GT_in_planning':False,'frozen_before_driving':True})
    print([(r['id'],r['spawn_index'],round(r['length_m'],2)) for r in routes])

if __name__=='__main__':main()
