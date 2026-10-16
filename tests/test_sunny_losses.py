import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.sunny_losses import foreground_dice,segmentation_loss


class ForegroundDiceTests(unittest.TestCase):
    def test_ignored_pixels_have_zero_gradient(self):
        logits=torch.randn(1,16,2,3,requires_grad=True);target=torch.tensor([[[255,1,2],[3,4,0]]])
        segmentation_loss(logits,target,torch.ones(16)).backward()
        self.assertTrue(torch.equal(logits.grad[0,:,0,0],torch.zeros(16)))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_correct_prediction_and_focus_empty_negative(self):
        target=torch.ones(1,2,2,dtype=torch.long);good=torch.full((1,16,2,2),-15.);good[:,1]=15
        bad=good.clone();bad[:,1]=-15;bad[:,2]=15
        self.assertLess(float(foreground_dice(good,target)),.001)
        self.assertGreater(float(foreground_dice(bad,target)),.5)
        negative=torch.randn(1,16,2,2,requires_grad=True)
        loss=foreground_dice(negative,torch.zeros_like(target));self.assertEqual(float(loss.detach()),0.);loss.backward()
        self.assertTrue(torch.equal(negative.grad,torch.zeros_like(negative)))

if __name__=='__main__':unittest.main()
