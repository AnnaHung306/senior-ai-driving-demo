"""Loopback-only navigation commands and immutable captured-run replay."""
import base64, copy, hashlib, json, re, threading, time, uuid
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .navigation_io import read, sha
from .navigation_runtime import drive, route_from_request
from .core import write_json

ROOT=Path(__file__).resolve().parents[2]
WEB=ROOT/'web/world-viewer'
RUN_PATTERN=r'phase(?:89|10)-drive-[a-zA-Z0-9-]+'
IMPORT_MAP='{"imports":{"three":"/vendor/three.module.js"}}'
IMPORT_MAP_CSP=base64.b64encode(hashlib.sha256(IMPORT_MAP.encode()).digest()).decode()

class NavigationService:
    def __init__(self,replay_only=False):
        preferred=ROOT/'artifacts/runs/phase89-route-catalog-v1-002/catalog.json'
        self.catalog=read(preferred if preferred.is_file() else ROOT/'configs/phase89_routes_v1.json')
        self.replay_only=replay_only
        self.revision=0;self.instance_id=uuid.uuid4().hex
        self.last_viewer_contact=time.monotonic();self.viewer_timeout_s=3.
        self.config=read(ROOT/'configs/phase89_navigation_v1.json');self.lock=threading.RLock();self.cancel=threading.Event();self.worker=None
        self.preview=None;self.state={'status':'idle','source_mode':'no_live_job','latest':None,'message':'選擇起點與終點，先預覽路線','formal_quality_passed':False}

    def publish(self,state):
        with self.lock:self.revision+=1;self.state=state

    def snapshot(self):
        with self.lock:
            state=copy.deepcopy(self.state)
            if 'report' in state:
                state['report']={k:v for k,v in state['report'].items() if k not in ['records','route']}
            state['worker_active']=self.worker is not None and self.worker.is_alive()
            state['cancel_requested']=self.cancel.is_set()
            state['replay_only']=self.replay_only
            state['revision']=self.revision;state['instance_id']=self.instance_id
            state['viewer_lease_timeout_s']=self.viewer_timeout_s
            return state

    def viewer_contact(self):
        with self.lock:self.last_viewer_contact=time.monotonic()

    def viewer_expired(self,now=None):
        with self.lock:
            expired=(time.monotonic() if now is None else now)-self.last_viewer_contact>self.viewer_timeout_s
            if expired:self.cancel.set()
            return expired

    def command(self,action,body):
        with self.lock:
            active=self.worker is not None and self.worker.is_alive()
            if action=='stop':
                self.cancel.set();self.revision+=1;return {'status':'stop_requested','active':active}
            if active:raise RuntimeError('駕駛尚在執行，請先停止並等候清理')
            if action=='preview':
                if set(body)=={'case_id'}:
                    route=next((r for r in self.catalog['routes'] if r['id']==body['case_id']),None)
                    if route is None:raise ValueError('Unknown fixed route')
                elif set(body)=={'start_xy_m','goal_xy_m'}:
                    if self.replay_only:raise RuntimeError('純回放模式不連接CARLA；可選固定已保存路線預覽')
                    import carla
                    client=carla.Client('127.0.0.1',2000);client.set_timeout(15)
                    route=route_from_request(client,body['start_xy_m'],body['goal_xy_m'],self.config)
                else:raise ValueError('Expected endpoint coordinates or fixed case')
                token=hashlib.sha256(json.dumps(route,sort_keys=True).encode()).hexdigest()
                self.preview={'token':token,'route':route};return copy.deepcopy(self.preview)
            if action=='start':
                if self.replay_only:raise RuntimeError('純回放模式禁止啟動車輛')
                if set(body)!={'token'} or not self.preview or body['token']!=self.preview['token']:raise ValueError('先預覽目前路線')
                route=copy.deepcopy(self.preview['route']);self.cancel.clear()
                self.last_viewer_contact=time.monotonic()
                run_id='phase10-drive-web-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                output=ROOT/'artifacts/runs'/run_id
                self.state={'status':'starting','source_mode':'live_CARLA_map_RGB_hybrid','run_id':run_id,'route':route,'latest':None,'message':'準備感測與模型','formal_quality_passed':False}
                self.revision+=1
                def work():
                    finished=threading.Event()
                    def watchdog():
                        while not finished.wait(.25):
                            if self.viewer_expired():return
                    monitor=threading.Thread(target=watchdog,name='navigation-viewer-lease',daemon=True);monitor.start()
                    try:drive(route,output,self.cancel,self.publish)
                    except Exception as e:
                        output.mkdir(parents=True,exist_ok=True)
                        write_json(output/'service-failure.json',{'status':'failed_before_orchestration_completed','error':type(e).__name__+': '+str(e),'run_id':run_id})
                        self.publish({'status':'failed','run_id':run_id,'source_mode':'live_CARLA_map_RGB_hybrid','latest':None,'message':str(e),'formal_quality_passed':False})
                    finally:finished.set();monitor.join(1)
                self.worker=threading.Thread(target=work,name='single-CARLA-tick-owner',daemon=False);self.worker.start()
                return {'status':'accepted','run_id':run_id}
            if action=='replay':
                if set(body)!={'run_id','index'}:raise ValueError('Expected run_id and integer frame index')
                run_id=body['run_id'];index=body['index']
                if not isinstance(run_id,str) or not re.fullmatch(RUN_PATTERN,run_id) or type(index) is not int:raise ValueError('Invalid replay identity')
                folder=ROOT/'artifacts/runs'/run_id;report=read(folder/'report.json')
                if not 0<=index<len(report['records']):raise ValueError('Replay index outside run')
                item=report['records'][index];path=folder/item['file']
                if not isinstance(item['file'],str) or not re.fullmatch(r'frames/[0-9]{8,12}/record\.json',item['file']):raise ValueError('Replay record path outside captured frames')
                if sha(path)!=item['sha256']:raise ValueError('Captured run changed')
                self.state={'status':'replay','source_mode':'recorded_closed_loop_replay','run_id':run_id,'route':report['route'],'latest':read(path),
                            'message':'已完成駕駛的錄製回放，未控制車輛','replay_index':index,'replay_frames':len(report['records']),'report':report,'formal_quality_passed':False}
                self.revision+=1
                return {'status':'replay_loaded','frames':len(report['records'])}
            raise ValueError('Unknown command')

    def runs(self):
        rows=[]
        for p in sorted((ROOT/'artifacts/runs').glob('phase*-drive-*/report.json')):
            if not re.fullmatch(RUN_PATTERN,p.parent.name):continue
            r=read(p)
            if r.get('records'):rows.append({'id':p.parent.name,'status':r['status'],'frames':len(r['records']),'passed':r['engineering_route_passed']})
        return rows


def create_server(service,port=8768):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,data,mime='application/json; charset=utf-8'):
            payload=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf8') if isinstance(data,(dict,list)) else data
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',f"default-src 'self'; script-src 'self' 'sha256-{IMPORT_MAP_CSP}'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            try:
                if self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}':return self.reply(403,{'error':'Host denied'})
                parsed=urlparse(self.path);path=parsed.path
                if path=='/api/navigation/catalog':return self.reply(200,{k:service.catalog[k] for k in ['map','opendrive_sha256','spawn_points','roads_xy_m','routes']})
                if path=='/api/navigation/state':service.viewer_contact();return self.reply(200,service.snapshot())
                if path=='/api/navigation/runs':return self.reply(200,service.runs())
                if path=='/api/navigation/rgb':
                    q=parse_qs(parsed.query)
                    if set(q) not in [{'camera','frame'},{'camera','frame','run_id'}] or any(len(v)!=1 for v in q.values()):raise ValueError('Camera and frame identity required')
                    state=service.snapshot();r=state.get('latest');n=q['camera'][0]
                    if 'run_id' in q:
                        run=q['run_id'][0];f=q['frame'][0]
                        if not re.fullmatch(RUN_PATTERN,run) or not re.fullmatch(r'[0-9]{1,12}',f):raise ValueError('Invalid captured identity')
                        report=read(ROOT/'artifacts/runs'/run/'report.json')
                        item=next((x for x in report['records'] if str(x['frame'])==f),None)
                        if item is None or not re.fullmatch(r'frames/[0-9]{8,12}/record\.json',item['file']):raise ValueError('Frame not in captured report')
                        record_path=ROOT/'artifacts/runs'/run/item['file']
                        if sha(record_path)!=item['sha256']:raise ValueError('Captured frame changed')
                        r=read(record_path);state={'run_id':run}
                    if not r or str(r['frame'])!=q['frame'][0] or n not in r['RGB']:raise ValueError('Image frame no longer current')
                    entry=r['RGB'][n];p=ROOT/'artifacts/runs'/state['run_id']/entry['file']
                    if n not in {'front','front_right','right','rear_right','rear','rear_left','left','front_left'} or not re.fullmatch(r'frames/[0-9]{8,12}/'+re.escape(n)+r'\.png',entry['file']):raise ValueError('RGB path outside captured frame')
                    if sha(p)!=entry['sha256']:raise ValueError('RGB source changed')
                    return self.reply(200,p.read_bytes(),'image/png')
                allowed={'/':'navigation.html','/navigation.html':'navigation.html','/navigation.js':'navigation.js','/navigation.css':'navigation.css',
                         '/navigation-contract.js':'navigation-contract.js','/vendor/three.module.js':'vendor/three.module.js','/vendor/three.core.js':'vendor/three.core.js','/vendor/OrbitControls.js':'vendor/OrbitControls.js'}
                if path not in allowed:return self.reply(404,{'error':'Not found'})
                p=WEB/allowed[path];mime='text/html; charset=utf-8' if p.suffix=='.html' else 'text/css; charset=utf-8' if p.suffix=='.css' else 'text/javascript; charset=utf-8'
                return self.reply(200,p.read_bytes(),mime)
            except (ValueError,KeyError,FileNotFoundError) as e:return self.reply(422,{'error':str(e)})
        def do_POST(self):
            try:
                if self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}':return self.reply(403,{'error':'Host denied'})
                expected=f'http://127.0.0.1:{self.server.server_port}'
                origin=self.headers.get('Origin')
                if origin is not None and origin!=expected:return self.reply(403,{'error':'Origin denied'})
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.reply(415,{'error':'JSON required'})
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:return self.reply(413,{'error':'Command length outside limit'})
                body=json.loads(self.rfile.read(length),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
                if not isinstance(body,dict):raise ValueError('JSON object required')
                path=urlparse(self.path).path
                if not path.startswith('/api/navigation/'):return self.reply(404,{'error':'Not found'})
                return self.reply(200,service.command(path.rsplit('/',1)[1],body))
            except RuntimeError as e:return self.reply(409,{'error':str(e)})
            except (ValueError,KeyError,FileNotFoundError,TypeError,StopIteration) as e:return self.reply(422,{'error':str(e)})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)
