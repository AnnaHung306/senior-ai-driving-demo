"""Read-only native scenery inventory for train-town supplement planning.

Never starts a simulator, loads a map, ticks, changes settings/weather, spawns
actors, or renders sensors. An inventory is not training data or approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.core import write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def training_town(town,plan):
    allowed={scene['town'] for scene in plan['scenes'] if scene['split']=='train'}
    if town not in {'Town01','Town03'} or town not in allowed:
        raise ValueError('Only original training towns Town01/Town03 may be scouted')


def xyz(value):return [float(value.x),float(value.y),float(value.z)]


def rotation(value):return {'pitch':float(value.pitch),'yaw':float(value.yaw),'roll':float(value.roll)}


def object_record(obj):
    # Keep native values explicit. No speculative camera poses or bbox-space
    # conversion is derived from these records before a separate native check.
    name=str(obj.name);box=obj.bounding_box
    return {'native_id':int(obj.id),'native_name':name,'native_semantic_type':str(obj.type),
            'native_transform_matrix':obj.transform.get_matrix(),
            'bounding_box_as_returned':{'location':xyz(box.location),'rotation_deg':rotation(box.rotation),'extent_m':xyz(box.extent)},
            'name_hints_not_material_GT':[token for token in ['wood','picket','brick','stone','concrete','chain','metal','fence'] if token in name.casefold()]}


def main(args):
    root=args.root.resolve();plan=read(root/'plan.json');training_town(args.town,plan)
    audit=read(root/'audit.json');review=read(root/'visual_review.json')
    if audit['status']!='passed' or review['status']!='passed' or not review['training_approved'] or review['audit_sha256']!=sha(root/'audit.json'):
        raise ValueError('Bound original data approval required')
    if args.output.exists():raise FileExistsError('Use a new scout output version')
    args.output.mkdir(parents=True)
    report={'status':'failed','scope':'Read-only native training-town scenery metadata; no rendered images or model inputs',
            'town':args.town,'source_audit_sha256':sha(root/'audit.json'),'source_plan_sha256':sha(root/'plan.json'),
            'tool_sha256':sha(Path(__file__)),'rendered_images':0,'training_approved':False}
    write_json(args.output/'report.json',report)
    try:
        import carla
        client=carla.Client('127.0.0.1',args.port);client.set_timeout(60)
        if client.get_server_version()!='0.9.16':raise ValueError('Expected CARLA0.9.16')
        world=client.get_world();native_map=world.get_map()
        if native_map.name.split('/')[-1]!=args.town:
            raise ValueError('Prepare the requested training map in a separate client; scout will not load it')
        digest=hashlib.sha256(native_map.to_opendrive().encode()).hexdigest()
        if digest!=plan['maps'][args.town]['opendrive_sha256']:raise ValueError('Map differs from frozen training plan')
        report.update(opendrive_sha256=digest,server_version=client.get_server_version(),weather_observed=str(world.get_weather()))
        files={}
        for name,label in [('fences',carla.CityObjectLabel.Fences),('walls',carla.CityObjectLabel.Walls)]:
            records=sorted((object_record(obj) for obj in world.get_environment_objects(label)),key=lambda row:(row['native_name'],row['native_id']))
            path=args.output/(name+'.json');write_json(path,records)
            files[path.name]={'objects':len(records),'sha256':sha(path)}
            report['files']=files;write_json(args.output/'report.json',report)
        spawns=[{'spawn_index':i,'native_transform_matrix':pose.get_matrix()} for i,pose in enumerate(native_map.get_spawn_points())]
        path=args.output/'road_spawn_points.json';write_json(path,spawns)
        files[path.name]={'poses':len(spawns),'sha256':sha(path)}
        report.update(status='completed_metadata_only',files=files,
                      limitations='Native object names are hints, not visual material GT; bbox values are preserved as returned, not converted into verified camera geometry. No new independent route or training approval.')
        write_json(args.output/'report.json',report);print(json.dumps(report))
    except Exception as exc:
        report.update(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        write_json(args.output/'report.json',report);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--town',choices=['Town01','Town03'],required=True);p.add_argument('--port',type=int,default=2000)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
