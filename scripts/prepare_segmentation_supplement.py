"""Freeze train-town static scenery views from audited native inventories."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.sunny_data import sha
from senior_ai.core import write_json


def select_anchors(spawns,objects,count=64):
    positions=np.array([s['native_transform_matrix'] for s in spawns])[:,:2,3]
    selected=[];reasons={}
    def add(i,reason):
        if i in selected or selected and np.linalg.norm(positions[selected]-positions[i],axis=1).min()<12:return False
        selected.append(i);reasons[i]=reason;return True
    # Names only prioritize candidates; labels and material claims require RGB/GT.
    for hint,quota in [('picket',20),('wall',40)]:
        for obj in sorted(objects,key=lambda r:r['native_name']):
            if hint not in obj['native_name'].casefold():continue
            # Native object transform is an unambiguous world origin. No bbox-space assumption.
            center=np.asarray(obj['native_transform_matrix'])[:2,3]
            distances=np.linalg.norm(positions-center,axis=1);i=int(distances.argmin())
            if distances[i]<=40:add(i,{'name_hint_not_material_GT':hint,'native_id':obj['native_id'],'distance_to_native_origin_m':float(distances[i])})
            if len(selected)>=quota:break
    if not selected:add(0,{'geometry_only':True})
    while len(selected)<count:
        distances=np.linalg.norm(positions[:,None]-positions[selected][None],axis=2).min(1)
        i=int(distances.argmax())
        if not add(i,{'farthest_geometry_only':True}):raise ValueError('Not enough separated road anchors')
    return [{**spawns[i],'selection':reasons[i]} for i in selected]


def main(args):
    if args.output.exists():raise FileExistsError('Freeze a new plan version')
    main=json.loads((args.root/'plan.json').read_text(encoding='utf-8'))
    rig=json.loads((args.root/'rig.json').read_text(encoding='utf-8'))
    original_rig=ROOT/'configs/camera_rig_v1.json'
    if sha(original_rig)!=main['rig_sha256'] or rig!=json.loads(original_rig.read_text(encoding='utf-8')):
        raise ValueError('Main rig differs from frozen rig definition')
    towns={};inventories=[]
    for scout in args.scout:
        report=json.loads((scout/'report.json').read_text(encoding='utf-8'));town=report['town']
        if town not in {'Town01','Town03'} or town in towns:raise ValueError('Distinct original train towns only')
        if report['status']!='completed_metadata_only' or report['source_audit_sha256']!=sha(args.root/'audit.json'):raise ValueError('Unbound inventory')
        if report['opendrive_sha256']!=main['maps'][town]['opendrive_sha256']:raise ValueError('Map mismatch')
        objects=[]
        for name in ['fences.json','walls.json','road_spawn_points.json']:
            if sha(scout/name)!=report['files'][name]['sha256']:raise ValueError('Inventory changed')
            if name!='road_spawn_points.json':objects.extend(json.loads((scout/name).read_text(encoding='utf-8')))
        anchors=select_anchors(json.loads((scout/'road_spawn_points.json').read_text(encoding='utf-8')),objects,args.anchors)
        towns[town]={'opendrive_sha256':report['opendrive_sha256'],'anchors':anchors}
        inventories.append({'root':scout.as_posix(),'report_sha256':sha(scout/'report.json'),'files':report['files']})
    if set(towns)!={'Town01','Town03'}:raise ValueError('Both original training towns required')
    write_json(args.output,{'version':'sunny-static-segmentation-supplement-v1','split':'train','task':'segmentation_only',
        'source_audit_sha256':sha(args.root/'audit.json'),'source_plan_sha256':sha(args.root/'plan.json'),
        'weather':'ClearNoon','fixed_delta_s':.05,'rig':rig,'towns':towns,'inventories':inventories,
        'selection':'Native name hints then separated farthest road spawn geometry; all8 cameras per anchor, no model score selection.',
        'scope':'Stationary virtual rig anchors with no ego/NPC. Not dynamic episodes, collision-free routes, object detection or phase training. Nearby anchors/views are correlated.',
        'minimum_anchor_separation_m':12,'minimum_pixels_per_target_class':100000,'minimum_views_per_target_class':16,
        'target_classes':[12,13,16],'preparer_sha256':sha(Path(__file__))})
    print(json.dumps({'plan_sha256':sha(args.output),'anchors':{k:len(v['anchors']) for k,v in towns.items()},'views':sum(len(v['anchors']) for v in towns.values())*8}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scout',type=Path,nargs='+',required=True)
    p.add_argument('--anchors',type=int,default=64);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
