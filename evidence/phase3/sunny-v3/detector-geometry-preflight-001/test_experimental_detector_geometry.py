from pathlib import Path
import random
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experimental_detector_geometry import transform_one,paired_batch


class DetectorGeometryTests(unittest.TestCase):
    def target(self):
        return {'boxes':torch.tensor([[0.,2.,8.,8.]]),'labels':torch.tensor([3]),
            'ignore_boxes':torch.tensor([[8.,0.,16.,8.]]),'visible_pixels':torch.tensor([64])}

    def test_exact_boxes_flip_and_source_immutable(self):
        rgb=torch.arange(3*8*16).reshape(3,8,16).float()/1000;t=self.target()
        image,truth,ledger=transform_one(rgb,t,scale=.5,flip=True,offset=(1,2))
        self.assertTrue(torch.equal(truth['boxes'],torch.tensor([[10.,2.,14.,5.]])))
        self.assertTrue(torch.equal(truth['ignore_boxes'][0],torch.tensor([6.,1.,10.,5.])))
        self.assertTrue(torch.equal(t['boxes'],torch.tensor([[0.,2.,8.,8.]])))
        self.assertEqual(set(truth),{'boxes','labels','ignore_boxes'})
        self.assertTrue(ledger['padding_ignored']);self.assertEqual(image.shape,rgb.shape)

    def test_padding_exactly_covers_canvas_outside_paste(self):
        rgb=torch.ones(3,8,16);_,truth,_=transform_one(rgb,self.target(),.5,offset=(1,2))
        actual=torch.zeros(8,16,dtype=torch.bool)
        for box in truth['ignore_boxes'][1:].int():
            x1,y1,x2,y2=box.tolist();actual[y1:y2,x1:x2]=True
        expected=torch.ones(8,16,dtype=torch.bool);expected[1:5,2:10]=False
        self.assertTrue(torch.equal(actual,expected))

    def test_unit_geometry_is_exact_and_empty_targets_work(self):
        rgb=torch.rand(3,8,16);target={'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long),'ignore_boxes':torch.empty(0,4)}
        out,t,_=transform_one(rgb,target,1.,False,(0,0))
        self.assertTrue(torch.equal(rgb,out));self.assertEqual(t['boxes'].shape,(0,4));self.assertEqual(t['ignore_boxes'].shape,(0,4))

    def test_rng_resume_reproduces_all_training_geometry(self):
        rgb=torch.rand(2,3,8,16);targets=[self.target(),self.target()]
        random.seed(16);state=random.getstate();a,t,record=paired_batch(rgb,targets)
        random.setstate(state);b,u,again=paired_batch(rgb,targets)
        self.assertTrue(torch.equal(a,b));self.assertEqual(record,again)
        for left,right in zip(t,u):
            for key in left:self.assertTrue(torch.equal(left[key],right[key]))

    def test_asymmetric_pixels_follow_horizontal_flip(self):
        rgb=torch.zeros(3,8,16);rgb[:,2:8,0:8]=1
        out,truth,_=transform_one(rgb,self.target(),1.,True,(0,0))
        self.assertTrue(torch.equal(out,rgb.flip(-1)))
        self.assertTrue(torch.equal(truth['boxes'],torch.tensor([[8.,2.,16.,8.]])))

    def test_invalid_geometry_refused(self):
        for options in [{'scale':1.1},{'scale':.1},{'scale':.5,'offset':(9,0)}]:
            with self.subTest(options=options),self.assertRaises(ValueError):transform_one(torch.rand(3,8,16),self.target(),**options)


if __name__=='__main__':unittest.main()
