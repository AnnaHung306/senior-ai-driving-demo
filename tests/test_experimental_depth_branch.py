"""Focused CPU mechanism tests; no model quality/production acceptance claim."""
from pathlib import Path
import sys,unittest
import numpy as np
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from experimental_depth_branch import ExperimentalDepthBranch,decode_depth_rgb,depth_target,masked_depth_loss
from senior_ai.mobile_segmentation import MobileUNet


class DepthBranchPrototypeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(20261018)

    def test_packed_RGB_order_metric_scale_and_invalid_masks(self):
        packed=np.array([[[1,0,0],[0,1,0],[0,0,1],[255,255,255]]],dtype=np.uint8)
        np.testing.assert_allclose(decode_depth_rgb(packed),np.array([[1,256,65536,16777215]])*1000/16777215)
        depth=torch.tensor([[[5.,10.,20.,float('nan'),0.,120.,2.]]]);raw=torch.tensor([[[4,11,0,4,4,4,255]]])
        target,valid=depth_target(depth,raw)
        self.assertEqual(valid.tolist(),[[[True,False,False,False,False,False,False]]]);self.assertTrue(torch.isfinite(target).all())

    def test_ignored_pixels_and_empty_masks_have_no_loss_or_gradient(self):
        prediction=torch.tensor([[[[.1,.9]]]],requires_grad=True);target=torch.tensor([[[.1,float('nan')]]]);valid=torch.tensor([[[True,False]]])
        loss=masked_depth_loss(prediction,target,valid);loss.backward()
        self.assertEqual(float(loss),0.);self.assertEqual(float(prediction.grad[0,0,0,1]),0.)
        empty=prediction.detach().clone().requires_grad_();zero=masked_depth_loss(empty,target,torch.zeros_like(valid));zero.backward()
        self.assertEqual(float(zero),0.);self.assertTrue(torch.equal(empty.grad,torch.zeros_like(empty)))

    def test_RGB_identity_is_RNG_neutral_and_depth_reaches_shared_encoder(self):
        base=MobileUNet(precision='float32').eval();rgb=torch.rand(1,3,64,96)
        with torch.inference_mode():original=base(rgb)
        rng=torch.get_rng_state().clone();wrapper=ExperimentalDepthBranch(base)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with torch.inference_mode():logits,depth=wrapper(rgb)
        self.assertTrue(torch.equal(original,logits));self.assertEqual(tuple(depth.shape),(1,1,16,24))
        self.assertEqual(len(base.dec4._forward_hooks),0)
        wrapper.train();_,depth=wrapper(rgb);loss=masked_depth_loss(depth,torch.full((1,64,96),.2),torch.ones(1,64,96,dtype=torch.bool))
        loss.backward();shared=base.encoder[0][0].weight.grad
        self.assertIsNotNone(shared);self.assertGreater(float(shared.abs().sum()),0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None))
        wrapper.eval()
        with torch.inference_mode():self.assertTrue(torch.equal(wrapper(rgb,with_depth=False),base(rgb)))


if __name__=='__main__':unittest.main()
