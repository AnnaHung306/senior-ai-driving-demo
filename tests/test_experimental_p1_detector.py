"""CPU architectural/transfer preflight, not CUDA capacity or model quality."""
from pathlib import Path
import sys
import unittest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from experimental_p1_detector import add_p1_cpu_prototype
from senior_ai.detection_model import make_detector


class P1PrototypeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(20261018)

    def test_every_existing_parameter_preserved_with_explicit_fpn_index_shift(self):
        model=make_detector(min_size=64,max_size=96)
        old={k:v.detach().clone() for k,v in model.state_dict().items()}
        model=add_p1_cpu_prototype(model);new=model.state_dict()
        for name,value in old.items():
            target=name
            if name.startswith('backbone.fpn.'):
                parts=name.split('.');parts[3]=str(int(parts[3])+1);target='.'.join(parts)
            self.assertTrue(torch.equal(value,new[target]),name)
        self.assertEqual(model.rpn.anchor_generator.num_anchors_per_location(),[3]*6)
        self.assertEqual(model.roi_heads.box_roi_pool.featmap_names,['0','1','2','3','4'])

    def test_existing_coarse_features_exact_and_new_stride2_shape(self):
        model=make_detector(min_size=128,max_size=192).eval();x=torch.rand(1,3,128,192)
        with torch.inference_mode():before=model.backbone(x)
        model=add_p1_cpu_prototype(model).eval()
        with torch.inference_mode():after=model.backbone(x)
        self.assertEqual(after['0'].shape,(1,256,64,96))
        for index in range(4):self.assertTrue(torch.equal(before[str(index)],after[str(index+1)]))
        self.assertTrue(torch.equal(before['pool'],after['pool']))

    def test_new_projection_participates_in_finite_cpu_training_and_eval(self):
        model=add_p1_cpu_prototype(make_detector(min_size=64,max_size=96)).train()
        images=[torch.rand(3,64,96)]
        targets=[{'boxes':torch.tensor([[10.,10.,20.,40.],[30.,10.,35.,20.]]),'labels':torch.tensor([3,7]),'ignore_boxes':torch.tensor([[0.,40.,15.,50.]])}]
        losses=model(images,targets);loss=sum(losses.values());loss.backward()
        gradient=model.backbone.fpn.inner_blocks[0][0].weight.grad
        self.assertTrue(torch.isfinite(loss));self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.eval()
        with torch.inference_mode():prediction=model(images)[0]
        self.assertTrue(torch.isfinite(prediction['boxes']).all());self.assertLessEqual(len(prediction['boxes']),100)

    def test_cannot_apply_twice_or_silently_use_incompatible_pyramid(self):
        model=add_p1_cpu_prototype(make_detector(min_size=64,max_size=96))
        with self.assertRaises(ValueError):add_p1_cpu_prototype(model)


if __name__=='__main__':unittest.main()
