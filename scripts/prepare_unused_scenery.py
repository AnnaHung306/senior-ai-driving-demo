"""Freeze unused-town static segmentation candidates before any image capture.

Native names/heights only prioritize preview positions; never material GT.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.core import write_json
from senior_ai.sunny_data import sha
from scout_unused_training_town import validate_unused

VERSION='sunny-static-segmentation-expansion-v2'
POLICY=ROOT/'configs/segmentation_training_expansion_scout_v1.json'
MAIN=ROOT/'data/sunny-cross-town-v3-007'
INVENTORY_FILES=['report.json','policy.json','walls.json','fences.json','road_spawn_points.json','available_maps.json']
RUNTIME_FILES=['state.json','contract.json','runtime_inventory.json']


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def select_anchors(spawns,walls,count,maximum_hint_distance_m=40):
    if maximum_hint_distance_m not in [40,80]:raise ValueError('Explicit40m or80m preview distance required')
    if not 4<=count<=96 or len(spawns)<count:raise ValueError('4..96 available native poses required')
    positions=np.asarray([s['native_transform_matrix'] for s in spawns],dtype=float)[:,:2,3]
    if not np.isfinite(positions).all():raise ValueError('Nonfinite native poses')
    selected=[];reasons={}
    def add(index,reason):
        if index in selected or selected and np.linalg.norm(positions[selected]-positions[index],axis=1).min()<8:return False
        selected.append(index);reasons[index]=reason;return True
    buckets={key:[] for key in [(height,hint) for height in ['low','tall'] for hint in ['wood','concrete','stone','other']]}
    for obj in sorted(walls,key=lambda r:(r['native_name'],r['native_id'])):
        height=2*float(obj['bounding_box_as_returned']['extent_m'][2])
        if not .2<=height<=6:continue
        origin=np.asarray(obj['native_transform_matrix'])[:2,3];dist=np.linalg.norm(positions-origin,axis=1)
        valid=np.flatnonzero((dist>=3)&(dist<=maximum_hint_distance_m))
        if not len(valid):continue
        index=min(valid,key=lambda i:(dist[i],int(i)));hint=next((k for k in ['wood','concrete','stone'] if k in obj['native_name'].casefold()),'other')
        buckets[('low' if height<=1.6 else 'tall',hint)].append((int(index),{'native_wall_id':obj['native_id'],'name_hint_not_material_GT':hint,
            'native_origin_distance_m':float(dist[index]),'native_bbox_height_hint_m':height}))
    # Interleave height/name strata and maximize separation within each stratum.
    # Native hints only select positions; RGB/raw labels decide actual material.
    while any(buckets.values()) and len(selected)<max(4,count*3//4):
        for entries in buckets.values():
            if entries:
                if selected:
                    pick=max(range(len(entries)),key=lambda j:(float(np.linalg.norm(positions[selected]-positions[entries[j][0]],axis=1).min()),-j))
                else:pick=0
                index,reason=entries.pop(pick);add(index,reason)
                if len(selected)>=max(4,count*3//4):break
    if not selected:add(0,{'geometry_only':True})
    while len(selected)<count:
        distances=np.linalg.norm(positions[:,None]-positions[selected][None],axis=2).min(1)
        index=int(distances.argmax())
        if not add(index,{'farthest_geometry_only':True}):raise ValueError('Not enough8m-separated road poses')
    return [{**spawns[i],'selection':reasons[i]} for i in selected]


def validate_expansion_plan(plan,archive_root=None,check_runtime=True):
    main=read(MAIN/'plan.json');policy=read(POLICY);audit=read(MAIN/'audit.json');review=read(MAIN/'visual_review.json')
    if plan.get('version')!=VERSION or plan.get('split')!='train' or plan.get('task')!='segmentation_only' or plan.get('weather')!='ClearNoon':
        raise ValueError('Explicit sunny train-only expansion-v2 required')
    if plan.get('training_approved') is not False or plan.get('fixed_delta_s')!=.05 or plan.get('target_classes')!=[12,16] or plan.get('minimum_pixels_per_target_class')!=100000 or plan.get('minimum_views_per_target_class')!=16 or plan.get('minimum_anchor_separation_m')!=8:
        raise ValueError('Frozen acquisition schema/support differs; plan is not approval')
    if audit['status']!='passed' or review['status']!='passed' or not review['training_approved'] or review['audit_sha256']!=sha(MAIN/'audit.json'):
        raise ValueError('Bound original data approval required')
    if plan['source_audit_sha256']!=sha(MAIN/'audit.json') or plan['source_plan_sha256']!=sha(MAIN/'plan.json'):
        raise ValueError('Main source changed')
    if plan['policy_sha256']!=sha(POLICY) or policy['source_audit_sha256']!=plan['source_audit_sha256'] or policy['source_plan_sha256']!=plan['source_plan_sha256']:
        raise ValueError('Training-only reservation differs')
    if plan['rig']!=read(MAIN/'rig.json') or plan['rig']!=read(ROOT/'configs/camera_rig_v1.json') or sha(ROOT/'configs/camera_rig_v1.json')!=main['rig_sha256']:
        raise ValueError('Original eight-camera rig required')
    if not plan['towns'] or not set(plan['towns']).issubset({'Town06','Town07'}):raise ValueError('Reserved full unused towns only')
    for town,entry in plan['towns'].items():
        validate_unused(town,main,policy,entry['opendrive_sha256'])
        if set(entry['inventory_files'])!=set(INVENTORY_FILES):raise ValueError('Full native metadata binding required')
        scout=(archive_root/'native_inventories'/town) if archive_root else (ROOT/entry['inventory_root']).resolve()
        if not archive_root and not scout.is_relative_to(ROOT/'artifacts/runs'):raise ValueError('Project native inventory required')
        for name,digest in entry['inventory_files'].items():
            if name not in INVENTORY_FILES or sha(scout/name)!=digest:raise ValueError('Native inventory bytes changed')
        report=read(scout/'report.json')
        if report['status']!='completed_metadata_only' or report['town']!=town or report['source_audit_sha256']!=plan['source_audit_sha256'] or report['policy_sha256']!=plan['policy_sha256'] or report['opendrive_sha256']!=entry['opendrive_sha256']:
            raise ValueError('Inventory not bound to unused training candidate')
        for name,value in report['files'].items():
            if sha(scout/name)!=value['sha256']:raise ValueError('Native inventory internal digest differs')
        spawns={a['spawn_index']:a['native_transform_matrix'] for a in read(scout/'road_spawn_points.json')}
        ids=[a['spawn_index'] for a in entry['anchors']]
        if len(ids)!=len(set(ids)) or not ids:raise ValueError('Unique native anchors required')
        xy=np.asarray([a['native_transform_matrix'] for a in entry['anchors']])[:,:2,3]
        if len(xy)>1 and min(np.linalg.norm(xy[i+1:]-xy[i],axis=1).min() for i in range(len(xy)-1))<8:
            raise ValueError('Anchors too close to claim the planned separation')
        for a in entry['anchors']:
            matrix=np.asarray(a['native_transform_matrix'],dtype=float)
            if a['spawn_index'] not in spawns or matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.array_equal(matrix,np.asarray(spawns[a['spawn_index']])):
                raise ValueError('Anchor differs from native road pose')
            if not np.allclose(matrix[3],[0,0,0,1],atol=1e-8) or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-5):raise ValueError('Nonrigid anchor')
    origin=plan['runtime'];receipt=(archive_root/'runtime_provenance') if archive_root else (ROOT/origin['receipt_root']).resolve()
    if set(origin['files'])!=set(RUNTIME_FILES):raise ValueError('Full prepared runtime provenance binding required')
    for name,digest in origin['files'].items():
        if name not in RUNTIME_FILES or sha(receipt/name)!=digest:raise ValueError('Runtime provenance changed')
    state=read(receipt/'state.json')
    if state['status']!='completed_content_not_native_validated' or not state['baseline_unchanged']:raise ValueError('Complete isolated content preparation required')
    if state['runtime_inventory_sha256']!=sha(receipt/'runtime_inventory.json') or state['contract_sha256']!=sha(receipt/'contract.json'):
        raise ValueError('Runtime manifest receipt differs')
    if check_runtime:
        runtime=Path(state['runtime']).resolve()
        if runtime.parent!=ROOT/'artifacts' or not runtime.name.startswith('carla-phase3-expansion-'):raise ValueError('Isolated runtime required')
        manifest=read(receipt/'runtime_inventory.json')
        for name in ['CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe']+[f'CarlaUE4/Content/Carla/Maps/{town}.umap' for town in plan['towns']]:
            if sha(runtime/name)!=manifest[name]['sha256']:raise ValueError('Live binary/main map differs from prepared assets')
    return main


def archive_inputs(plan,root):
    """Preserve exact metadata so later training needs no live CARLA runtime."""
    for town,entry in plan['towns'].items():
        source=ROOT/entry['inventory_root'];target=root/'native_inventories'/town;target.mkdir(parents=True,exist_ok=True)
        for name,digest in entry['inventory_files'].items():
            dest=target/name
            if dest.exists() and sha(dest)!=digest:raise ValueError('Archived input differs')
            if not dest.exists():dest.write_bytes((source/name).read_bytes())
    target=root/'runtime_provenance';target.mkdir(exist_ok=True)
    for name,digest in plan['runtime']['files'].items():
        dest=target/name
        if dest.exists() and sha(dest)!=digest:raise ValueError('Archived runtime input differs')
        if not dest.exists():dest.write_bytes((ROOT/plan['runtime']['receipt_root']/name).read_bytes())


def main(args):
    if args.output.exists():raise FileExistsError('New frozen plan path required')
    state=read(args.runtime_receipt/'state.json');policy=read(POLICY);towns={}
    for scout in args.scout:
        report=read(scout/'report.json');town=report['town']
        if town in towns:raise ValueError('Duplicate native town inventory')
        towns[town]={'opendrive_sha256':report['opendrive_sha256'],'inventory_root':scout.resolve().relative_to(ROOT).as_posix(),
            'inventory_files':{name:sha(scout/name) for name in INVENTORY_FILES},
            'anchors':select_anchors(read(scout/'road_spawn_points.json'),[o for o in read(scout/'walls.json') if not args.low_only or .2<=2*float(o['bounding_box_as_returned']['extent_m'][2])<=1.6],args.anchors,args.maximum_hint_distance_m)}
    plan={'version':VERSION,'split':'train','task':'segmentation_only','training_approved':False,
        'source_audit_sha256':sha(MAIN/'audit.json'),'source_plan_sha256':sha(MAIN/'plan.json'),'policy_sha256':sha(POLICY),
        'weather':'ClearNoon','fixed_delta_s':.05,'rig':read(MAIN/'rig.json'),'towns':towns,
        'runtime':{'receipt_root':args.runtime_receipt.resolve().relative_to(ROOT).as_posix(),'files':{name:sha(args.runtime_receipt/name) for name in RUNTIME_FILES},
            'capture_integrity_scope':'Full distribution verified at preparation. Capture rechecks manifest, binary and main maps; no claim of continuously rehashing every asset.'},
        'minimum_anchor_separation_m':8,'minimum_pixels_per_target_class':100000,'minimum_views_per_target_class':16,'target_classes':[12,16],
        'scope':'Previously unused Town06/07 static virtual road rig, sunny train-only segmentation. No ego/NPC, dynamics, new route independence, or detection/phase training. Preview probe cannot pass training audit.',
        'selection':f'Native Walls origins within{args.maximum_hint_distance_m}m; low-only={args.low_only}; height/name strata with farthest separation, then farthest road geometry. Before RGB; hints are not material GT.',
        'preparer_sha256':sha(Path(__file__))}
    validate_expansion_plan(plan);write_json(args.output,plan)
    print(json.dumps({'plan_sha256':sha(args.output),'towns':list(towns),'training_approved':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scout',nargs='+',type=Path,required=True);p.add_argument('--runtime-receipt',type=Path,required=True)
    p.add_argument('--maximum-hint-distance-m',type=int,choices=[40,80],default=40);p.add_argument('--low-only',action='store_true')
    p.add_argument('--anchors',type=int,default=64);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
