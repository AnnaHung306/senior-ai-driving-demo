"""Real HTTP-started driving without any viewer heartbeat; wait via disk only."""
import argparse,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8769/api/navigation/'

def post(action,body):
    request=urllib.request.Request(BASE+action,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as r:return json.loads(r.read())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--while-moving',action='store_true');parser.add_argument('--output',type=Path);a=parser.parse_args()
    out=a.output or ROOT/('artifacts/reports/phase10-real-viewer-watchdog-moving-final.json' if a.while_moving else 'artifacts/reports/phase10-real-viewer-watchdog-final.json')
    if out.exists():raise FileExistsError(out)
    preview=post('preview',{'case_id':'straight-1'});accepted=post('start',{'token':preview['token']})
    run=ROOT/'artifacts/runs'/accepted['run_id'];start=time.monotonic()
    if a.while_moving:
        while True:
            with urllib.request.urlopen(BASE+'state',timeout=10) as response:state=json.loads(response.read())
            if (state.get('latest') or {}).get('speed_mps',0)>.5:break
            if time.monotonic()-start>45:post('stop',{});raise TimeoutError('Did not start moving')
            time.sleep(.5)
    heartbeat_lost=time.monotonic()
    while not (run/'report.json').exists():
        if time.monotonic()-start>100:post('stop',{});raise TimeoutError('Viewer watchdog failed to finish')
        time.sleep(.15)
    r=json.loads((run/'report.json').read_text(encoding='utf8'))
    assert r['status']=='cancelled' and r['checks']['cleanup'] and r['checks']['cleanup_stopped'] and r['checks']['source_exact']
    max_speed=max((x['speed_mps'] for x in r['records']),default=0.)
    receipt={'status':'passed_real_HTTP_worker_viewer_watchdog','run_id':run.name,
        'no_GET_state_or_heartbeat_after_start':not a.while_moving,'no_GET_or_heartbeat_after_loss':True,
        'loss_triggered_after_verified_motion':a.while_moving,'waited_via_filesystem_only_after_loss':True,
        'time_from_loss_to_stopped_report_s':time.monotonic()-heartbeat_lost,
        'elapsed_wall_s':time.monotonic()-start,'max_recorded_speed_mps':max_speed,
        'cleanup_final_speed_mps':r['cleanup_final_speed_mps'],'records':r['frames'],'formal_quality_passed':False}
    out.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(receipt)

if __name__=='__main__':main()
