"""Read-only signal ground truth for diagnosis; does not tick or control actors."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read

def main():
    import carla
    out=ROOT/'artifacts/reports/phase10-signal-observer-v1-001.jsonl'
    if out.exists():raise FileExistsError(out)
    client=carla.Client('127.0.0.1',2000);client.set_timeout(10);world=client.get_world();last=None;start=time.monotonic()
    with out.open('w',encoding='utf8') as f:
        while time.monotonic()-start<2400:
            result=ROOT/'artifacts/reports/phase10-suite-audit.json'
            if result.exists() and len(read(result))==6:break
            snapshot=world.get_snapshot()
            if snapshot.frame!=last:
                vehicles=list(world.get_actors().filter('vehicle.*'));lights=list(world.get_actors().filter('traffic.traffic_light'))
                ego=next((v for v in vehicles if v.attributes.get('role_name')=='senior_AI_RGB_hybrid'),None)
                current=ego.get_traffic_light() if ego else None
                row={'readback_frame':snapshot.frame,'timestamp_s':snapshot.timestamp.elapsed_seconds,
                    'ego_at_signal':ego.is_at_traffic_light() if ego else None,'ego_traffic_light_id':current.id if current else None,
                    'lights':[{'id':l.id,'state':str(l.get_state()),'frozen':l.is_frozen(),
                        'position_m':[l.get_location().x,l.get_location().y,l.get_location().z]} for l in lights],
                    'GT_evaluation_only_not_sent_to_planner':True}
                f.write(json.dumps(row)+'\n');f.flush();last=snapshot.frame
            time.sleep(.5)
    print({'status':'observer_ended','file':str(out)})

if __name__=='__main__':main()
