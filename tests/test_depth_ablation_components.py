"""Focused paired-target/RNG and loss checks before the GPU research run."""
from pathlib import Path
import random,sys,unittest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from depth_ablation_components import paired_depth_geometry,nearest_depth_loss,nonwall_iou
from senior_ai.sunny_augmentation import paired_geometry
from run_depth_ablation import decide


class DepthAblationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_RGB_semantic_and_post_RNG_exact_production_for_all_scales(self):
        image=torch.arange(2*3*8*12).reshape(2,3,8,12).float();target=torch.arange(2*8*12).reshape(2,8,12)%17
        depth=target.float()/20;valid=target!=3
        for scale in [.75,1.,1.25]:
            for seed in range(4):
                random.seed(seed);a,b=paired_geometry(image,target,scales=[scale],flip_probability=.5);expected=random.getstate()
                random.seed(seed);x,y,z,m,_=paired_depth_geometry(image,target,depth,valid,scales=[scale],flip_probability=.5)
                self.assertTrue(torch.equal(a,x));self.assertTrue(torch.equal(b,y));self.assertEqual(expected,random.getstate())
                self.assertTrue(torch.equal(z[y!=255],y[y!=255].float()/20));self.assertTrue(torch.equal(m[y!=255],y[y!=255]!=3))
                self.assertTrue(torch.equal(z[y==255],torch.zeros_like(z[y==255])));self.assertFalse(bool(m[y==255].any()))

    def test_scale_keeps_metric_values_and_ignore_pixels_do_not_backpropagate(self):
        image=torch.zeros(1,3,8,12);target=torch.ones(1,8,12,dtype=torch.long);z=torch.full((1,8,12),.75);valid=torch.ones_like(z,dtype=torch.bool)
        random.seed(1);_,ids,depth,mask,_=paired_depth_geometry(image,target,z,valid,scales=[.75],flip_probability=1)
        self.assertTrue(torch.equal(depth[mask],torch.full_like(depth[mask],.75)));self.assertFalse(bool(mask[ids==255].any()))
        pred=torch.full((1,1,2,3),.5,requires_grad=True);loss=nearest_depth_loss(pred,depth,torch.zeros_like(mask));loss.backward()
        self.assertEqual(float(loss.detach()),0.);self.assertTrue(torch.equal(pred.grad,torch.zeros_like(pred)))

    def test_nonwall_summary_is_diagnostic_and_requires_all15_classes(self):
        rows=[{'name':'other_annotated','iou':1.,'support_pixels':1},{'name':'wall','iou':0.,'support_pixels':1}]
        rows.extend({'name':str(i),'iou':.6,'support_pixels':1} for i in range(15))
        self.assertAlmostEqual(nonwall_iou({'classes':rows}),.6)
        with self.assertRaises(ValueError):nonwall_iou({'classes':rows[:-1]})

    def test_screening_keeps_wall_gate_separate_and_never_automatically_expands(self):
        def result(score,margin=.6):
            rows=[{'name':'other_annotated','iou':.9,'support_pixels':1},{'name':'wall','iou':.1,'support_pixels':1}]
            rows.extend({'name':str(i),'iou':score,'support_pixels':1} for i in range(15))
            return {'classes':rows,'visible_margin':{'f1':margin}}
        self.assertFalse(decide(result(.6),result(.599))['positive_signal_for_assistant_review'])
        positive=decide(result(.6),result(.603));self.assertTrue(positive['positive_signal_for_assistant_review']);self.assertFalse(positive['automatic_extra_epochs'])
        self.assertFalse(decide(result(.6),result(.603,.58))['positive_signal_for_assistant_review'])


if __name__=='__main__':unittest.main()
