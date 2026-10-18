from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.detection_augmentation import bicycle_scale_batch


class DetectionScaleTrainingTests(unittest.TestCase):
    def target(self,label=3):
        return {'boxes':torch.tensor([[0.,2,8,8],[8,0,16,8]]),'labels':torch.tensor([label,7]),
            'ignore_boxes':torch.empty(0,4),'visible_pixels':torch.tensor([48,64])}

    def test_only_bicycle_positive_views_selected_all_boxes_follow(self):
        rgb=torch.ones(2,3,8,16);targets=[self.target(),self.target(2)]
        with patch('senior_ai.detection_augmentation.random.random',return_value=0),patch('senior_ai.detection_augmentation.random.randint',return_value=0):
            out,truth,ledger=bicycle_scale_batch(rgb,targets)
        self.assertEqual([r['zoomed'] for r in ledger],[True,False])
        self.assertTrue(torch.equal(truth[0]['boxes'],targets[0]['boxes']*.5))
        self.assertTrue(torch.equal(truth[1]['boxes'],targets[1]['boxes']))
        self.assertEqual(set(truth[0]),{'boxes','labels','ignore_boxes'})
        self.assertTrue(torch.equal(out[1],rgb[1]))
        self.assertTrue(torch.equal(targets[0]['boxes'],self.target()['boxes']))
        self.assertEqual(truth[0]['ignore_boxes'].tolist(),[[0,4,16,8],[8,0,16,4]])

    def test_rng_state_restores_geometry_and_targets(self):
        rgb=torch.rand(8,3,8,16);targets=[self.target() for _ in rgb]
        random.seed(42);state=random.getstate();out,truth,ledger=bicycle_scale_batch(rgb,targets)
        random.setstate(state);again,labels,record=bicycle_scale_batch(rgb,targets)
        self.assertTrue(torch.equal(out,again));self.assertEqual(ledger,record)
        for a,b in zip(truth,labels):
            for key in a:self.assertTrue(torch.equal(a[key],b[key]))

    def test_policy_and_matrix_mismatch_refused(self):
        for kw in [{'scale':.75},{'probability':.5}]:
            with self.assertRaises(ValueError):bicycle_scale_batch(torch.ones(1,3,8,16),[self.target()],**kw)
        with self.assertRaises(ValueError):bicycle_scale_batch(torch.ones(2,3,8,16),[self.target()])


if __name__=='__main__':unittest.main()
