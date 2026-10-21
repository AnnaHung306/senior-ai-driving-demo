"""Real loopback acceptance against replay-only service, with CARLA stopped."""
import argparse,json,socket,sys,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.core import write_json
from senior_ai.world_display import sha
BASE='http://127.0.0.1:8768';rows=[]
def request(path,expected,body=None,headers=None,raw=None):
    h=headers or {};data=raw if raw is not None else json.dumps(body).encode() if body is not None else None
    if data is not None:h={'Content-Type':'application/json',**h}
    req=urllib.request.Request(BASE+path,data=data,headers=h)
    try:
        with urllib.request.urlopen(req,timeout=10) as r:code=r.status;payload=r.read();response_headers=dict(r.headers)
    except urllib.error.HTTPError as e:code=e.code;payload=e.read();response_headers=dict(e.headers)
    assert code==expected,(path,code,payload[:200]);rows.append({'path':path,'status':code,'method':'GET' if data is None else 'POST'})
    return json.loads(payload) if response_headers.get('Content-Type','').startswith('application/json') else payload
def main(output):
    output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts/reports') or output.exists():raise ValueError('New report path under artifacts/reports required')
    with socket.socket() as s:s.settimeout(.5);assert s.connect_ex(('127.0.0.1',2000))!=0
    for p in ['/','/navigation.js','/navigation.css','/navigation-contract.js','/vendor/three.module.js','/vendor/three.core.js']:request(p,200)
    catalog=request('/api/navigation/catalog',200);assert len(catalog['spawn_points'])==255 and 'training_ego_metadata_used_for_selection_only' not in catalog
    state=request('/api/navigation/state',200);assert state['replay_only'] and not state['worker_active']
    assert request('/api/navigation/runs',200)
    for p in ['/PHASE89.md','/data/labels/a.json','/../configs/phase89_navigation_v1.json','/vendor/provenance.json']:request(p,404)
    request('/api/navigation/state',403,headers={'Host':'attacker.example'})
    request('/api/navigation/start',403,{},headers={'Origin':'https://attacker.example'})
    request('/api/navigation/start',415,{},headers={'Content-Type':'text/plain'})
    request('/api/navigation/start',413,raw=b'x'*5000)
    request('/api/navigation/start',422,raw=b'{broken')
    request('/api/navigation/start',409,{'token':'anything'})
    request('/api/navigation/preview',409,{'start_xy_m':[0,0],'goal_xy_m':[10,0]})
    request('/api/navigation/preview',200,{'case_id':'straight-1'})
    request('/api/navigation/replay',422,{'run_id':'../secrets','index':0})
    request('/api/navigation/rgb?camera=front&frame=0',422)
    request('/api/navigation/rgb?camera=front&frame=0&run_id=../secret',422)
    request('/api/navigation/rgb?camera=front&camera=rear&frame=0',422)
    request('/artifacts/runs/phase10-drive-a/audit/00000001.json',404)
    request('/vendor/OrbitControls.js',200)
    write_json(output,{'status':'passed_real_loopback_replay_only_HTTP','checks':rows,
        'CARLA_RPC_not_listening':True,'replay_only_prevents_start':True,'public_catalog_omits_training_GT_metadata':True,
        'source_sha256':sha(Path(__file__)),'formal_quality_passed':False})
    print({'checks':len(rows),'CARLA_stopped':True,'status':'passed'})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);main(parser.parse_args().output)
