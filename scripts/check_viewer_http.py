"""Real loopback HTTP integration using a separate disposable display fixture."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import sha
from senior_ai.world_viewer_service import ViewerBundle,create_server,publish_existing_frame


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='viewer-http-') as temp:
        root=Path(temp);(root/'frames').mkdir();data=root/'frames/a.json';data.write_text('{}')
        m={'schema_version':1,'source_mode':'recorded_inference_replay','control_issued':False,'GT_geometry_exported':False,'formal_model_quality_passed':False,
           'run_contract_sha256':'a'*64,'file_inventory':{'frames/a.json':sha(data)},
           'frames':[{'key':'s/1','scene_id':'s','frame':1,'timestamp_s':1,'file':'frames/a.json','sha256':sha(data)}], 'scenes':[{'id':'s','frames':['s/1']}]}
        (root/'bundle.json').write_text(json.dumps(m));bundle=ViewerBundle(root);feed=root/'feed';server=create_server(ROOT/'web/world-viewer',bundle,0,feed)
        thread=Thread(target=server.serve_forever,daemon=True);thread.start();base='http://127.0.0.1:'+str(server.server_address[1])
        def request(path,expected=200,method='GET'):
            try:
                with urlopen(Request(base+path,method=method),timeout=10) as r:code=r.status;body=r.read();headers=dict(r.headers)
            except HTTPError as e:code=e.code;body=e.read();headers=dict(e.headers)
            assert code==expected,(path,code,expected);checks.append({'path':path,'method':method,'status':code});return body,headers
        try:
            body,h=request('/');assert b'<canvas' not in body and b'viewport' in body;assert "default-src 'self'" in h['Content-Security-Policy']
            for p in ['/app.js','/vendor/three.module.js','/vendor/three.core.js','/vendor/OrbitControls.js','/api/health','/api/manifest','/data/frames/a.json']:request(p)
            request('/api/health',method='HEAD');request('/api/health',501,method='POST')
            for p in ['/PHASE5.md','/data/bundle.json','/data/labels/a.json','/vendor/provenance.json']:request(p,404)
            for p in ['/data/%2e%2e/PHASE5.md','/data/frames/%252e%252e/a.json','/data/C:/secret']:request(p,422)
            body,_=request('/api/feed');assert json.loads(body)['status']=='waiting'
            publish_existing_frame(bundle,feed,'s/1');body,_=request('/api/feed');assert json.loads(body)['status']=='ready'
            v=json.loads((feed/'latest.json').read_text());v['published_utc']=(datetime.now(timezone.utc)-timedelta(seconds=4)).isoformat();(feed/'latest.json').write_text(json.dumps(v))
            body,_=request('/api/feed');assert json.loads(body)['status']=='stale'
            v['actor_id']=123;(feed/'latest.json').write_text(json.dumps(v));request('/api/feed',422)
            data.write_text('{"tampered":true}');request('/data/frames/a.json',422)
        finally:server.shutdown();server.server_close();thread.join(timeout=5)
    print(json.dumps({'status':'passed_real_loopback_HTTP','checks':checks,'server_stopped':not thread.is_alive()}))


if __name__=='__main__':main()
