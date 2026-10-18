import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from senior_ai.segmentation_supplement import verify_scenery,append_scenery,resolved_class_alignment
from prepare_segmentation_supplement import select_anchors
from audit_segmentation_supplement import local_camera
from train_sunny import build_model
from predict_sunny import load_model
from senior_ai.perception_quality import selection_key
from run_phase3_recovery_batch import acquire_batch_lock


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


class SceneryTrainingTests(unittest.TestCase):
    def test_unknown_instance_pixels_are_reported_not_compared_as_a_class(self):
        semantic=np.array([[9,4,5,0]],dtype=np.uint8);instance=np.array([[0,4,4,5]],dtype=np.uint8)
        result=resolved_class_alignment(semantic,instance)
        self.assertEqual(result['raw_disagreement'],.75)
        self.assertEqual(result['resolved_disagreement'],.5)
        self.assertEqual(result['compared_pixels'],2)
        self.assertEqual(result['unlabelled_instance_pixels'],1)
        self.assertEqual(result['unlabelled_semantic_pixels'],1)
    def test_batch_lock_rejects_duplicate_runner_and_releases_on_close(self):
        (ROOT/'artifacts/runs/test-temp').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs/test-temp') as folder:
            path=Path(folder)/'batch.lock';first=acquire_batch_lock(path)
            try:
                with self.assertRaisesRegex(RuntimeError,'Another process'):acquire_batch_lock(path)
            finally:first.close()
            acquired=acquire_batch_lock(path);acquired.close()
    def test_selection_never_discards_a_qualified_model_for_higher_failing_mean(self):
        policy='qualified_first_then_score'
        self.assertGreater(selection_key(.51,True,policy),selection_key(.56,False,policy))
        self.assertGreater(selection_key(.52,True,policy),selection_key(.51,True,policy))
        self.assertEqual(selection_key(.56,False,'score_only'),(0,.56))
        with self.assertRaises(ValueError):selection_key(.51,True,'test-set-score')
    def test_geometry_selection_preserves_train_anchor_separation(self):
        spawns=[]
        for i in range(100):
            m=np.eye(4);m[:2,3]=[(i%10)*15,(i//10)*15];spawns.append({'spawn_index':i,'native_transform_matrix':m.tolist()})
        anchors=select_anchors(spawns,[],64);p=np.array([a['native_transform_matrix'] for a in anchors])[:,:2,3]
        d=np.linalg.norm(p[:,None]-p[None],axis=2);np.fill_diagonal(d,np.inf)
        self.assertGreaterEqual(d.min(),12);self.assertEqual(len({a['spawn_index'] for a in anchors}),64)
        m=local_camera({'yaw_deg':90,'position_m':[1.6,0,1.7]})
        np.testing.assert_allclose(m@np.array([1,0,0,1]),[1.6,1,1.7,1],atol=1e-7)

    def test_scenery_requires_audit_actual_review_and_exact_bytes(self):
        (ROOT/'artifacts/runs/test-temp').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs/test-temp') as folder:
            root=Path(folder);write(root/'plan.json',{'task':'segmentation_only','split':'train','towns':{'Town01':{},'Town03':{}},'weather':'ClearNoon','scope':'static'})
            rows=[{'split':'train','town':'Town01','group_id':'a','camera':'front','verified_obstacle_actor_ids':[]}]
            write(root/'training/train.jsonl',rows[0]);(root/'qa.png').write_bytes(b'reviewed-image-fixture')
            plan_sha=digest(root/'plan.json');audit={'status':'passed','checks':{'ok':True},'source_audit_sha256':'main','plan_sha256':plan_sha,'views':1,'segmentation_pixels':[0]*17,
                'files':[{'path':'training/train.jsonl','sha256':digest(root/'training/train.jsonl')}]}
            write(root/'audit.json',audit)
            review={'status':'passed','training_approved':True,'audit_sha256':digest(root/'audit.json'),'reviewed_images':[{'path':'qa.png','sha256':digest(root/'qa.png')}]};write(root/'visual_review.json',review)
            self.assertEqual(verify_scenery(root,'main',plan_sha)['views'],1)
            with self.assertRaisesRegex(ValueError,'main audit'):verify_scenery(root,'different',plan_sha)
            review['reviewed_images']=[];write(root/'visual_review.json',review)
            with self.assertRaisesRegex(ValueError,'visual review'):verify_scenery(root,'main',plan_sha)
            review['reviewed_images']=[{'path':'qa.png','sha256':digest(root/'qa.png')}];write(root/'visual_review.json',review)
            (root/'training/train.jsonl').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'bytes changed'):verify_scenery(root,'main',plan_sha)
        for train in [SimpleNamespace(task='detection',samples=[]),SimpleNamespace(task='segmentation',samples=[{'split':'validation'}])]:
            with self.assertRaisesRegex(ValueError,'training segmentation only'):append_scenery(train,ROOT,1,16)

    def test_project_initialization_binds_checkpoint_task_audit_and_inference_resize(self):
        (ROOT/'artifacts/runs/test-temp').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs/test-temp') as folder:
            path=Path(folder)/'initial.pt';model=torch.nn.Linear(2,3)
            config={'detection':{'pretrained_path':'unused.pth','initial_checkpoint_path':str(path),'initial_data_audit_sha256':'audit','min_size':720,'max_size':1280}}
            saved={'task':'detection','model':model.state_dict(),'provenance':{'data':{'audit_sha256':'audit'}},'config':config}
            torch.save(saved,path);config['detection']['initial_checkpoint_sha256']=digest(path)
            with patch('train_sunny.make_detector',side_effect=lambda **kw:torch.nn.Linear(2,3)):
                initialized=build_model('detection',config)
            self.assertTrue(all(torch.equal(value,initialized.state_dict()[key]) for key,value in model.state_dict().items()))
            with patch('predict_sunny.make_detector',return_value=torch.nn.Linear(2,3)) as factory:
                loaded,_,_=load_model(path,'cpu');factory.assert_called_once_with(min_size=720,max_size=1280)
            config['detection']['initial_data_audit_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'task or training audit'):build_model('detection',config)
            config['detection']['initial_checkpoint_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):build_model('detection',config)


if __name__=='__main__':unittest.main()
