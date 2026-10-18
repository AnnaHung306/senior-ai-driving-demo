"""CPU-only preflight, no model quality or formal integration claims."""
from pathlib import Path
import sys
import unittest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from experimental_feature_style import ExperimentalFeatureStyle,attach_prototype_hooks
from senior_ai.mobile_segmentation import MobileUNet


class FeatureStylePrototypeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261018)

    def test_eval_probability_zero_and_single_sample_are_rng_neutral_identity(self):
        x=torch.randn(2,3,5,7)
        cases=[(ExperimentalFeatureStyle().eval(),x),(ExperimentalFeatureStyle(probability=0).train(),x),
               (ExperimentalFeatureStyle(probability=1).train(),x[:1])]
        for operation,value in cases:
            before=torch.get_rng_state().clone();result=operation(value)
            self.assertIs(result,value);self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_epoch_cpu_rng_restore_reproduces_augmentation_exactly(self):
        x=torch.randn(8,3,5,7);operation=ExperimentalFeatureStyle(probability=1)
        before=torch.get_rng_state().clone();a=operation(x);torch.set_rng_state(before);b=operation(x)
        self.assertTrue(torch.equal(a,b));self.assertFalse(torch.equal(a,x))

    def test_preserves_within_channel_spatial_order_without_new_parameters(self):
        x=torch.randn(8,3,5,7);operation=ExperimentalFeatureStyle(probability=1)
        y=operation(x)
        self.assertEqual(y.shape,x.shape);self.assertEqual(list(operation.parameters()),[]);self.assertEqual(len(operation.state_dict()),0)
        self.assertTrue(torch.equal(torch.argsort(x.flatten(2),dim=2),torch.argsort(y.flatten(2),dim=2)))

    def test_constant_bfloat16_features_and_backward_stay_finite(self):
        x=torch.zeros(4,3,5,7,dtype=torch.bfloat16,requires_grad=True);y=ExperimentalFeatureStyle(probability=1)(x)
        y.float().square().mean().backward()
        self.assertEqual(y.dtype,x.dtype);self.assertTrue(torch.isfinite(y).all());self.assertTrue(torch.isfinite(x.grad).all())

    def test_mobile_prototype_eval_exact_identity_and_training_gradient(self):
        model=MobileUNet(precision='float32').eval();rgb=torch.rand(2,3,64,96)
        with torch.inference_mode():before=model(rgb)
        state_keys=list(model.state_dict());handles=attach_prototype_hooks(model,{'probability':1})
        try:
            with torch.inference_mode():after=model(rgb)
            self.assertTrue(torch.equal(before,after));self.assertEqual(state_keys,list(model.state_dict()))
            model.train();loss=model(rgb).square().mean();loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        finally:
            for handle in handles:handle.remove()

    def test_invalid_settings_and_feature_schema_rejected(self):
        for settings in [{'probability':-1},{'probability':2},{'alpha':0},{'epsilon':0},{'alpha':float('nan')},{'probability':True}]:
            with self.subTest(settings=settings),self.assertRaises(ValueError):ExperimentalFeatureStyle(**settings)
        operation=ExperimentalFeatureStyle()
        for x in [torch.zeros(3,4,5),torch.zeros(0,3,4,5),torch.zeros(2,3,4,5,dtype=torch.uint8)]:
            with self.assertRaises(ValueError):operation(x)


if __name__=='__main__':unittest.main()
