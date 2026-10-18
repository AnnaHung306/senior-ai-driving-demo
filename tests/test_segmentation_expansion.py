import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
ROOT=Path(__file__).resolve().parents[1]
TEMP=ROOT/'artifacts/runs/test-temp'
TEMP.mkdir(parents=True,exist_ok=True)
from senior_ai.sunny_data import sha
from senior_ai.segmentation_expansion import verify_expansion,append_expansion


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


class SegmentationExpansionTests(unittest.TestCase):
    def fixture(self,path):
        root=path/'expansion';main=path/'main';rig={'cameras':[{'name':str(i)} for i in range(8)]}
        write(main/'rig.json',rig);write(main/'plan.json',{'scenes':[{'town':'Town04_Opt'}],'maps':{'Town04':{'opendrive_sha256':'original'}}});write(main/'audit.json',{'status':'passed'})
        policy={'candidate_training_towns':['Town06','Town07'],'source_audit_sha256':sha(main/'audit.json'),'source_plan_sha256':sha(main/'plan.json')}
        write(root/'native_inventories/Town06/policy.json',policy)
        plan={'version':'sunny-static-segmentation-expansion-v2','task':'segmentation_only','split':'train','weather':'ClearNoon','training_approved':False,
            **{k:policy[k] for k in ['source_audit_sha256','source_plan_sha256']},'rig':rig,'policy_sha256':sha(root/'native_inventories/Town06/policy.json'),
            'towns':{'Town06':{'opendrive_sha256':'new','anchors':[{'spawn_index':0}]}},'scope':'fixture static training only'}
        write(root/'plan.json',plan);rows=[]
        for i in range(8):
            row={'town':'Town06','group_id':'town06-static-anchor-000','scene_id':'town06-static-anchor-000','camera':str(i),'split':'train','verified_obstacle_actor_ids':[]}
            for m in ['rgb','semantic','instance','depth']:
                p=root/f'raw/{i}/{m}.png';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture-not-rendered');row[m]=p.relative_to(root).as_posix()
            rows.append(row)
        (root/'training').mkdir();(root/'training/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        write(root/'Town06/report.json',{'status':'passed','cleanup':True,'probe_only':False,'explicit_success_readback_frames':True,'plan_sha256':sha(root/'plan.json'),'records':8})
        write(root/'Town06/records.json',[{'frame':i,'readback_frame':i+1} for i in range(8)]);write(root/'qa/catalog.json',[])
        (root/'qa/review.png').write_bytes(b'fixture-only')
        files=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in root.rglob('*') if p.is_file()]
        write(root/'audit.json',{'status':'passed','checks':{'complete':True},'source_audit_sha256':plan['source_audit_sha256'],'plan_sha256':sha(root/'plan.json'),
            'mapping_source_sha256':sha(Path(__file__).resolve().parents[1]/'src/senior_ai/sunny_data.py'),'files':files,'views':8,'groups':1,'segmentation_pixels':[0]*17})
        write(root/'visual_review.json',{'status':'passed','training_approved':True,'audit_sha256':sha(root/'audit.json'),'reviewed_images':[{'path':'qa/review.png','sha256':sha(root/'qa/review.png')}]})
        return root,main

    def verify(self,root,main):return verify_expansion(root,main,sha(root/'plan.json'),sha(root/'audit.json'),sha(root/'visual_review.json'))

    def test_complete_provenance_then_changed_image_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            root,main=self.fixture(Path(folder));self.assertEqual(self.verify(root,main)['views'],8)
            (root/'raw/0/rgb.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes changed'):self.verify(root,main)

    def test_probe_and_unreviewed_acquisition_cannot_train(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            root,main=self.fixture(Path(folder));p=root/'Town06/report.json';r=json.loads(p.read_text());r['probe_only']=True;write(p,r)
            with self.assertRaisesRegex(ValueError,'probes excluded'):self.verify(root,main)
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            root,main=self.fixture(Path(folder));p=root/'visual_review.json';r=json.loads(p.read_text());r['training_approved']=False;write(p,r)
            with self.assertRaisesRegex(ValueError,'visual review'):self.verify(root,main)

    def test_original_map_alias_and_training_boundary_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            root,main=self.fixture(Path(folder));p=root/'plan.json';r=json.loads(p.read_text());r['towns']['Town06']['opendrive_sha256']='original';write(p,r)
            p=root/'audit.json';a=json.loads(p.read_text());a['plan_sha256']=sha(root/'plan.json');write(p,a)
            p=root/'visual_review.json';v=json.loads(p.read_text());v['audit_sha256']=sha(root/'audit.json');write(p,v)
            with self.assertRaisesRegex(ValueError,'aliases excluded'):self.verify(root,main)
        for train in [SimpleNamespace(task='detection',samples=[]),SimpleNamespace(task='segmentation',samples=[{'split':'validation'}])]:
            with self.assertRaisesRegex(ValueError,'segmentation training only'):append_expansion(train,Path('.'),1,16)


if __name__=='__main__':unittest.main()
