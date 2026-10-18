import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from senior_ai.detection_model import make_detector, ignore_background,EagerDeterministicRoIAlign
from torchvision.ops import MultiScaleRoIAlign
from senior_ai.visibility_policy import ignore_background_proposals


class DetectorIgnoreTests(unittest.TestCase):
    def test_high_resolution_transform_preserves_raw_box_coordinates(self):
        model=make_detector(min_size=720,max_size=1280)
        target={'boxes':torch.tensor([[12.,20,18,34]]),'labels':torch.tensor([7]),
                'ignore_boxes':torch.tensor([[40.,60,44,70]])}
        image,out=model.transform.resize(torch.zeros(3,360,640),target)
        self.assertEqual(tuple(image.shape),(3,720,1280))
        self.assertTrue(torch.equal(out['boxes'],torch.tensor([[24.,40,36,68]])))
        self.assertTrue(torch.equal(out['ignore_boxes'],torch.tensor([[80.,120,88,140]])))
        model.eval()
        restored=model.transform.postprocess([{'boxes':out['boxes']}],[(720,1280)],[(360,640)])[0]['boxes']
        self.assertTrue(torch.equal(restored,torch.tensor([[12.,20,18,34]])))
        for dimensions in [(0,640),(720,640),(360.,640)]:
            with self.assertRaises(ValueError):make_detector(min_size=dimensions[0],max_size=dimensions[1])
    def test_eager_roi_matches_upstream_and_backpropagates(self):
        reference=MultiScaleRoIAlign(['0','1'],7,2)
        eager=EagerDeterministicRoIAlign(['0','1'],7,2)
        torch.manual_seed(12)
        features={k:torch.rand(1,4,12,20,requires_grad=True) for k in ['0','1']}
        boxes=[torch.tensor([[20.,20,80,100],[100.,100,300,300]])]
        expected=reference(features,boxes,[(360,640)])
        actual=eager(features,boxes,[(360,640)])
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6,rtol=1e-5))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(features['0'].grad).all())
    def test_matches_numpy_contract(self):
        b = np.array([[0,0,10,10], [20,20,30,30], [0,0,10,10], [8,8,12,12]], dtype=np.float32)
        l = np.array([0,0,2,-1])
        ignored = np.array([[0,0,10,10]], dtype=np.float32)
        np.testing.assert_array_equal(ignore_background(torch.tensor(b), torch.tensor(l), torch.tensor(ignored)).numpy(),
                                      ignore_background_proposals(b,l,ignored))

    def test_real_rpn_and_roi_samplers(self):
        model = make_detector()
        boxes = torch.tensor([[0.,0,10,10], [20.,20,30,30], [40.,40,50,50]])
        target = {'boxes': boxes[1:2], 'labels': torch.tensor([1]), 'ignore_boxes': boxes[:1]}
        labels, _ = model.rpn.assign_targets_to_anchors([boxes], [target])
        self.assertEqual(labels[0].tolist(), [-1., 1., 0.])
        _, _, selected_labels, _ = model.roi_heads.select_training_samples([boxes], [target])
        self.assertEqual(sorted(selected_labels[0].tolist()), [0,1,1])
        self.assertEqual(model.rpn.ignore_assignment_calls, 1)
        self.assertEqual(model.roi_heads.ignore_assignment_calls, 1)
        self.assertIsNone(model.roi_heads._training_ignore)
        # Ignored regions must not suppress a real GT positive.
        target['ignore_boxes'] = boxes
        labels, _ = model.rpn.assign_targets_to_anchors([boxes], [target])
        self.assertEqual(labels[0].tolist(), [-1.,1.,-1.])
        _, _, selected_labels, _ = model.roi_heads.select_training_samples([boxes], [target])
        self.assertEqual(selected_labels[0].tolist(), [1,1])

    def test_ignore_boxes_resize_and_empty_gt(self):
        model = make_detector()
        target = {'boxes': torch.empty(0,4), 'labels': torch.empty(0,dtype=torch.long),
                  'ignore_boxes': torch.tensor([[10.,10,20,20]])}
        image, out = model.transform.resize(torch.zeros(3,180,320), target)
        self.assertEqual(tuple(image.shape), (3,360,640))
        self.assertTrue(torch.equal(out['ignore_boxes'], torch.tensor([[20.,20,40,40]])))
        boxes = torch.tensor([[20.,20,40,40], [80.,80,100,100]])
        _, _, labels, _ = model.roi_heads.select_training_samples([boxes], [out])
        self.assertEqual(labels[0].tolist(), [0])

    def test_all_ignored_loss_is_zero_and_differentiable(self):
        model=make_detector()
        labels=[torch.full((5,),-1.)]
        objectness=torch.randn(5,requires_grad=True);deltas=torch.randn(5,4,requires_grad=True)
        a,b=model.rpn.compute_loss(objectness,deltas,labels,[torch.zeros(5,4)])
        self.assertEqual(float((a+b).detach()),0.);(a+b).backward()
        self.assertTrue(torch.equal(objectness.grad,torch.zeros_like(objectness)))
        # Real ROI forward with no selected proposals must not call empty CE.
        model.train();features={str(k):torch.rand(1,256,h,w,requires_grad=True) for k,(h,w) in enumerate([(16,24),(8,12),(4,6),(2,3)])}
        proposals=[torch.tensor([[0.,0,10,10]])];target={'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long),'ignore_boxes':torch.tensor([[0.,0,96,64]])}
        _,losses=model.roi_heads(features,proposals,[(64,96)],[target])
        total=sum(losses.values());self.assertEqual(float(total.detach()),0.)
        total.backward();self.assertEqual(model.roi_heads.last_supervised_samples,0)


if __name__ == '__main__':
    unittest.main()
