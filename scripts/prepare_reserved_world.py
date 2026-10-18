"""Load one explicitly unused map once, then verify it with a fresh client.

A load RPC exception is preserved; never blindly retry a map already loaded.
No sensors, actors or training. Sunny weather is applied after observation.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.core import write_json
from senior_ai.sunny_data import sha
from scout_unused_training_town import validate_unused


def main(args):
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT/'artifacts/runs'):raise ValueError('New project receipt required')
    plan=json.loads((args.root/'plan.json').read_text());policy=json.loads(args.policy.read_text());validate_unused(args.town,plan,policy)
    if sha(args.root/'audit.json')!=policy['source_audit_sha256'] or sha(args.root/'plan.json')!=policy['source_plan_sha256']:raise ValueError('Bound main/policy required')
    import carla
    result={'status':'failed','requested_map':args.town,'tool_sha256':sha(Path(__file__)),'started_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Native map preparation only; no image capture or training approval','load_RPC_called':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,result)
    try:
        c=carla.Client('127.0.0.1',args.port);c.set_timeout(120)
        if c.get_server_version()!='0.9.16':raise ValueError('CARLA0.9.16 required')
        initial=c.get_world().get_map().name;result['initial_map']=initial
        if initial.replace('\\','/').split('/')[-1]!=args.town:
            result['load_RPC_called']=True;write_json(args.output,result)
            try:c.load_world(args.town)
            except RuntimeError as exc:result['load_RPC_exception']=str(exc);write_json(args.output,result)
        # A new client observes actual state after success or exception.
        observed=carla.Client('127.0.0.1',args.port);observed.set_timeout(60);world=observed.get_world();native=world.get_map()
        result['observed_map']=native.name
        if native.name.replace('\\','/').split('/')[-1]!=args.town:raise ValueError('Requested full map not actually loaded')
        digest=hashlib.sha256(native.to_opendrive().encode()).hexdigest();validate_unused(args.town,plan,policy,digest)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        result.update(status='native_requested_full_map_observed',opendrive_sha256=digest,road_spawns=len(native.get_spawn_points()),
            weather=str(world.get_weather()),server_version=observed.get_server_version(),finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        result.update(error=type(exc).__name__,message=str(exc));raise
    finally:write_json(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--town',choices=['Town06','Town07'],required=True);p.add_argument('--port',type=int,default=2000)
    p.add_argument('--root',type=Path,default=ROOT/'data/sunny-cross-town-v3-007')
    p.add_argument('--policy',type=Path,default=ROOT/'configs/segmentation_training_expansion_scout_v1.json')
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
