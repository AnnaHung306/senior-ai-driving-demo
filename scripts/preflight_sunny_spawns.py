"""Native feasibility only; never training data or driving acceptance."""
import argparse
import json
from pathlib import Path
import sys
from collect_dataset import transform,ROOT,sha
from senior_ai.core import write_json


def main(args):
    import carla
    plan=json.loads(args.plan.read_text(encoding='utf-8'));c=carla.Client('127.0.0.1',2000);c.set_timeout(120)
    report={'task':'native-spawn-preflight','training_approved':False,'plan_sha256':sha(args.plan),'scenes':[]}
    for scene in plan['scenes']:
        if args.scenes and scene['id'] not in args.scenes: continue
        w=c.get_world()
        if w.get_map().name.split('/')[-1]!=scene['town']:w=c.load_world(scene['town'])
        original=w.get_settings();settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;w.apply_settings(settings)
        actors=[];rows=[]
        try:
            for _ in range(5):w.tick(20)
            definitions=[{'blueprint':'vehicle.tesla.model3','kind':'ego','start':scene['start']}]+scene['participants']
            for definition in definitions:
                a=w.try_spawn_actor(w.get_blueprint_library().find(definition['blueprint']),transform(definition['start']))
                rows.append({'blueprint':definition['blueprint'],'kind':definition['kind'],'success':a is not None})
                if a is not None:actors.append(a)
            for _ in range(5):w.tick(20)
        finally:
            for a in reversed(actors):a.destroy()
            w.apply_settings(original)
        result={'scene':scene['id'],'status':'passed' if all(r['success'] for r in rows) else 'failed','assets':rows}
        report['scenes'].append(result);write_json(args.output,report);print(json.dumps(result),flush=True)
    report['status']='passed' if all(s['status']=='passed' for s in report['scenes']) else 'failed';report['source_sha256']=sha(Path(__file__));write_json(args.output,report)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scenes',nargs='*');sys.exit(main(p.parse_args()))
