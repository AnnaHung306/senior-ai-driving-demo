import hashlib,json,sys,tempfile,threading,unittest,urllib.error,urllib.request
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_service import NavigationService,create_server,IMPORT_MAP,IMPORT_MAP_CSP


class Phase10ServiceTests(unittest.TestCase):
    def setUp(self):
        self.service=NavigationService(replay_only=True);self.server=create_server(self.service,0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.service.cancel.set();self.server.shutdown();self.server.server_close();self.thread.join()
    def get(self,path,body=None):
        req=urllib.request.Request(self.base+path,data=None if body is None else json.dumps(body).encode(),headers={} if body is None else {'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req) as r:return r.status,r.read()
        except urllib.error.HTTPError as e:return e.code,e.read()
    def test_immutable_RGB_survives_current_seek_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'artifacts/runs/phase10-drive-fixture';f=run/'frames/00000024';f.mkdir(parents=True)
            image=b'PNG-fixture';(f/'front.png').write_bytes(image)
            record={'frame':24,'RGB':{'front':{'file':'frames/00000024/front.png','sha256':hashlib.sha256(image).hexdigest()}}}
            (f/'record.json').write_text(json.dumps(record));digest=hashlib.sha256((f/'record.json').read_bytes()).hexdigest()
            (run/'report.json').write_text(json.dumps({'records':[{'frame':24,'file':'frames/00000024/record.json','sha256':digest}]}))
            self.service.publish({'status':'replay','run_id':'phase10-drive-other','latest':{'frame':99}})
            with patch('senior_ai.navigation_service.ROOT',root):
                path='/api/navigation/rgb?run_id=phase10-drive-fixture&frame=24&camera=front'
                self.assertEqual(self.get(path),(200,image))
                (f/'front.png').write_bytes(b'changed');self.assertEqual(self.get(path)[0],422)
                self.assertEqual(self.get(path.replace('frame=24','frame=23'))[0],422)
    def test_GT_paths_and_bad_new_run_prefix_rejected(self):
        for path in ['/artifacts/runs/phase10-drive-fixture/audit/1.json','/api/navigation/rgb?run_id=phase100-drive-a&frame=24&camera=front']:
            self.assertIn(self.get(path)[0],[404,422])
    def test_viewer_timeout_cancels_even_without_pagehide(self):
        self.service.replay_only=False;self.service.viewer_timeout_s=.05
        self.service.command('preview',{'case_id':'straight-1'})
        began=threading.Event()
        def worker(route,out,cancel,publish):began.set();cancel.wait(3)
        with patch('senior_ai.navigation_service.drive',side_effect=worker):
            self.service.command('start',{'token':self.service.preview['token']});self.assertTrue(began.wait(1))
            self.service.worker.join(2);self.assertFalse(self.service.worker.is_alive());self.assertTrue(self.service.cancel.is_set())
    def test_read_state_renews_viewer_but_publish_does_not(self):
        self.service.last_viewer_contact=0;self.service.publish({'status':'running'})
        self.assertEqual(self.service.last_viewer_contact,0);self.get('/api/navigation/state')
        self.assertGreater(self.service.last_viewer_contact,0)
    def test_import_map_is_hash_bound_and_vendor_explicitly_served(self):
        import base64
        self.assertEqual(base64.b64encode(hashlib.sha256(IMPORT_MAP.encode()).digest()).decode(),IMPORT_MAP_CSP)
        self.assertEqual(self.get('/vendor/OrbitControls.js')[0],200)

if __name__=='__main__':unittest.main()
