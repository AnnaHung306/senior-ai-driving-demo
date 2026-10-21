import json,sys,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_service import NavigationService,create_server

class NavigationHTTPTests(unittest.TestCase):
    def setUp(self):
        self.service=NavigationService();self.server=create_server(self.service,0);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.base=f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):self.service.cancel.set();self.server.shutdown();self.server.server_close();self.thread.join()
    def request(self,path,body=None,origin=None,ctype='application/json'):
        data=None if body is None else json.dumps(body).encode();headers={}
        if data is not None:headers['Content-Type']=ctype
        if origin is not None:headers['Origin']=origin
        req=urllib.request.Request(self.base+path,data=data,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=10) as r:return r.status,r.read()
        except urllib.error.HTTPError as e:return e.code,e.read()
    def test_static_allowlist_and_catalog(self):
        for p in ['/','/navigation.js','/navigation.css','/navigation-contract.js','/api/navigation/catalog','/api/navigation/state']:
            self.assertEqual(self.request(p)[0],200)
        for p in ['/PHASE89.md','/../PHASE89.md','/vendor/provenance.json','/data/labels/a.json']:self.assertEqual(self.request(p)[0],404)
    def test_origin_and_content_type(self):
        self.assertEqual(self.request('/api/navigation/start',{},'https://attacker.example')[0],403)
        self.assertEqual(self.request('/api/navigation/start',{},ctype='text/plain')[0],415)
    def test_forged_preview_and_bad_case_rejected(self):
        self.assertEqual(self.request('/api/navigation/start',{'token':'forged'})[0],422)
        self.assertEqual(self.request('/api/navigation/preview',{'case_id':'../other'})[0],422)
    def test_preview_token_current_route_and_single_worker_stop(self):
        began=threading.Event()
        def fake_drive(route,out,cancel,publish):began.set();cancel.wait(5)
        code,data=self.request('/api/navigation/preview',{'case_id':'straight-1'});self.assertEqual(code,200);token=json.loads(data)['token']
        with patch('senior_ai.navigation_service.drive',side_effect=fake_drive):
            self.assertEqual(self.request('/api/navigation/start',{'token':token})[0],200);self.assertTrue(began.wait(1))
            self.assertEqual(self.request('/api/navigation/start',{'token':token})[0],409)
            self.assertEqual(self.request('/api/navigation/preview',{'case_id':'straight-2'})[0],409)
            self.assertEqual(self.request('/api/navigation/stop',{})[0],200);self.service.worker.join(2);self.assertFalse(self.service.worker.is_alive())
    def test_path_injection_and_float_replay_index(self):
        for body in [{'run_id':'../secrets','index':0},{'run_id':'phase89-drive-a','index':1.2},{'run_id':'phase89-drive-a','index':True}]:
            self.assertEqual(self.request('/api/navigation/replay',body)[0],422)
    def test_tampered_report_cannot_escape_record_folder(self):
        report={'records':[{'file':'../../secret.json','sha256':'0'*64}]}
        with patch('senior_ai.navigation_service.read',return_value=report),patch('senior_ai.navigation_service.sha',side_effect=AssertionError('Must reject before file access')):
            self.assertEqual(self.request('/api/navigation/replay',{'run_id':'phase89-drive-any','index':0})[0],422)
    def test_RGB_requires_current_source(self):
        self.assertEqual(self.request('/api/navigation/rgb?camera=front&frame=999')[0],422)
    def test_empty_stop_is_idempotent(self):
        for _ in range(2):self.assertEqual(self.request('/api/navigation/stop',{})[0],200)
    def test_replay_only_forbids_start_and_dynamic_route(self):
        self.service.replay_only=True
        self.assertEqual(self.request('/api/navigation/start',{'token':'anything'})[0],409)
        self.assertEqual(self.request('/api/navigation/preview',{'start_xy_m':[0,0],'goal_xy_m':[10,0]})[0],409)
    def test_server_snapshot_revision_orders_worker_updates(self):
        before=self.service.snapshot();self.service.publish({'status':'running','latest':None});after=self.service.snapshot()
        self.assertEqual(before['instance_id'],after['instance_id']);self.assertGreater(after['revision'],before['revision'])
        self.request('/api/navigation/stop',{});self.assertGreater(self.service.snapshot()['revision'],after['revision'])
if __name__=='__main__':unittest.main()
