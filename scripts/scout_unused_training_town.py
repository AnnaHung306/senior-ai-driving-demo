"""Metadata-only scout for explicitly reserved, previously unused train towns.

Does not start CARLA, load maps, tick, alter weather or create actors/sensors.
Main split towns and OpenDrive aliases remain excluded even if policy is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scout_training_scenery import object_record
from senior_ai.core import write_json
from senior_ai.sunny_data import sha


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def town_family(value):
    leaf=value.replace('\\','/').rstrip('/').split('/')[-1]
    match=re.fullmatch(r'town(\d+)(hd)?(?:_opt)?',leaf,re.IGNORECASE)
    if not match:raise ValueError('Unrecognized native town name')
    return f'Town{int(match[1]):02d}'+('HD' if match[2] else '')


def validate_unused(town,plan,policy,opendrive_sha256=None):
    family=town_family(town)
    if town not in {'Town06','Town07'} or family!=town:
        raise ValueError('Only explicit full-map Town06/Town07 candidates; no variants')
    used={town_family(scene['town']) for scene in plan['scenes']}
    if family in used:raise ValueError('Candidate already belongs to an original data split')
    if family not in policy['candidate_training_towns']:raise ValueError('Candidate not reserved in frozen policy')
    if policy['task']!='segmentation_training_expansion_scout_only' or policy['training_approved'] is not False:
        raise ValueError('Metadata scout is not a training approval')
    if opendrive_sha256 is not None and opendrive_sha256 in {entry['opendrive_sha256'] for entry in plan['maps'].values()}:
        raise ValueError('Native map is an alias of an already used map')
    return family


def main(args):
    root=args.root.resolve();plan=read(root/'plan.json');policy=read(args.policy)
    audit=read(root/'audit.json');review=read(root/'visual_review.json')
    if audit['status']!='passed' or review['status']!='passed' or not review['training_approved'] or review['audit_sha256']!=sha(root/'audit.json'):
        raise ValueError('Bound original data approval required')
    if sha(root/'metadata_inventory.json')!=audit['metadata_inventory_sha256']:raise ValueError('Main metadata inventory changed')
    metadata={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
    if sha(root/'plan.json')!=metadata['plan.json'] or sha(root/'audit.json')!=policy['source_audit_sha256'] or sha(root/'plan.json')!=policy['source_plan_sha256']:
        raise ValueError('Policy/main plan binding differs')
    validate_unused(args.town,plan,policy)
    if args.dry_run:
        print(json.dumps({'status':'policy_validated_without_RPC','town':args.town,'training_approved':False,'policy_sha256':sha(args.policy)}));return
    if args.output is None or args.output.exists() or args.output.resolve().is_relative_to(root):
        raise ValueError('New explicit inventory output outside immutable main data required')
    args.output.mkdir(parents=True)
    report={'status':'failed','town':args.town,'rendered_images':0,'training_approved':False,
        'scope':'Previously unused train-candidate metadata only; no data/model inputs or new route approval',
        'source_audit_sha256':sha(root/'audit.json'),'source_plan_sha256':sha(root/'plan.json'),
        'policy_sha256':sha(args.policy),'tool_sha256':sha(Path(__file__)),'native_record_helper_sha256':sha(ROOT/'scripts/scout_training_scenery.py')}
    (args.output/'policy.json').write_bytes(args.policy.read_bytes())
    for source in [Path(__file__),ROOT/'scripts/scout_training_scenery.py']:
        target=args.output/'source_snapshot'/source.name;target.parent.mkdir(exist_ok=True);target.write_bytes(source.read_bytes())
    write_json(args.output/'report.json',report)
    try:
        import carla
        client=carla.Client('127.0.0.1',args.port);client.set_timeout(60)
        if client.get_server_version()!='0.9.16':raise ValueError('CARLA0.9.16 required')
        world=client.get_world();native=world.get_map()
        if native.name.replace('\\','/').split('/')[-1]!=args.town:raise ValueError('Prepare requested full map separately; scout never loads it')
        opendrive=native.to_opendrive()
        if not opendrive.strip():raise ValueError('Native OpenDrive is empty')
        digest=hashlib.sha256(opendrive.encode()).hexdigest();validate_unused(args.town,plan,policy,digest)
        report.update(opendrive_sha256=digest,server_version=client.get_server_version(),weather_observed=str(world.get_weather()))
        files={}
        for name,label in [('fences',carla.CityObjectLabel.Fences),('walls',carla.CityObjectLabel.Walls)]:
            rows=sorted((object_record(o) for o in world.get_environment_objects(label)),key=lambda r:(r['native_name'],r['native_id']))
            path=args.output/(name+'.json');write_json(path,rows);files[path.name]={'objects':len(rows),'sha256':sha(path)}
            report['files']=files;write_json(args.output/'report.json',report)
        spawns=[{'spawn_index':i,'native_transform_matrix':pose.get_matrix()} for i,pose in enumerate(native.get_spawn_points())]
        path=args.output/'road_spawn_points.json';write_json(path,spawns);files[path.name]={'poses':len(spawns),'sha256':sha(path)}
        write_json(args.output/'available_maps.json',client.get_available_maps());files['available_maps.json']={'sha256':sha(args.output/'available_maps.json')}
        report.update(status='completed_metadata_only',files=files,
            limitations='Native names/box heights are appearance hints only. Must inspect actual RGB/GT and freeze a separate new data plan/audit/review before training. Original train-only v1 loader remains unchanged.')
        write_json(args.output/'report.json',report);print(json.dumps(report))
    except Exception as exc:
        report.update(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());write_json(args.output/'report.json',report);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--town',choices=['Town06','Town07'],required=True);p.add_argument('--port',type=int,default=2000);p.add_argument('--output',type=Path);p.add_argument('--dry-run',action='store_true');main(p.parse_args())
