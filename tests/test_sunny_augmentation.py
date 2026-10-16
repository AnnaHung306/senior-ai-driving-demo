import random
import sys
from pathlib import Path
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.sunny_augmentation import paired_geometry
from senior_ai.mobile_segmentation import MobileUNet


class AugmentationTests(unittest.TestCase):
    def test_flip_preserves_exact_rgb_label_correspondence(self):
        ids=torch.arange(12).reshape(1,3,4).long();image=ids[:,None].float().repeat(1,3,1,1)/16
        out,target=paired_geometry(image,ids,scales=(1.,),flip_probability=1.)
        self.assertTrue(torch.equal(target,ids.flip(-1)))
        self.assertTrue(torch.equal(out[:,0]*16,target.float()))

    def test_downscale_padding_is_ignored_and_reproducible(self):
        image=torch.ones(2,3,40,64);target=torch.full((2,40,64),7,dtype=torch.long)
        random.seed(87);rgb,ids=paired_geometry(image,target,scales=(.75,))
        random.seed(87);again,labels=paired_geometry(image,target,scales=(.75,))
        self.assertTrue(torch.equal(rgb,again));self.assertTrue(torch.equal(ids,labels))
        self.assertTrue(torch.equal(rgb.sum(1)==0,ids==255));self.assertEqual(set(ids.unique().tolist()),{7,255})

    def test_pretrained_encoder_decoder_support_odd_rgb_and_both_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(7);model=MobileUNet(precision='bfloat16')
        rgb=torch.rand(1,3,61,95);logits=model(rgb)
        self.assertEqual(tuple(logits.shape),(1,17,61,95));self.assertEqual(logits.dtype,torch.float32)
        logits.square().mean().backward()
        for weight in [model.encoder[0][0].weight,model.stem[0].weight,model.head.weight]:
            self.assertTrue(torch.isfinite(weight.grad).all());self.assertGreater(float(weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
