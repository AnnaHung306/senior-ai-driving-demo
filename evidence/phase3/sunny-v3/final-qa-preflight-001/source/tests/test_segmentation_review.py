from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from review_sunny_segmentation import batch_indices,selected_cases,supervised_margins
from senior_ai.sunny_data import SEGMENTATION_NAMES
from senior_ai.segmentation_metrics import confusion,metrics
from senior_ai.road_boundaries import boundary_counts


class SegmentationReviewTests(unittest.TestCase):
    def test_original_batch_boundaries_and_short_last_batch(self):
        self.assertEqual(batch_indices(7,10,8),list(range(8)))
        self.assertEqual(batch_indices(8,10,8),[8,9])
        for index,size in [(-1,8),(10,8),(1,0)]:
            with self.assertRaises(ValueError):batch_indices(index,10,size)

    def test_display_margin_uses_same_ignore_neighborhood_as_metric(self):
        truth=np.array([[255,1,2,2],[255,1,2,2],[0,1,2,2]],dtype=np.uint8)
        a,b=supervised_margins(truth,truth);counts=boundary_counts(truth,truth)
        self.assertEqual(int(a.sum()),counts['gt_pixels']);self.assertEqual(int(b.sum()),counts['predicted_pixels'])
        self.assertFalse(a[0,1]);self.assertFalse(a[1,1]);self.assertTrue(a[2,1])

    def test_all_focus_classes_and_other_only_wall_false_positive_included(self):
        frames=[]
        for i in range(1,17):
            target=np.full((4,4),i,dtype=np.uint8);prediction=np.zeros_like(target)
            frames.append({'index':i-1,'scene_id':'scene','split':'validation','visible_margin':{'gt_pixels':0,'f1':None},
                **metrics(confusion(target,prediction,17),SEGMENTATION_NAMES)})
        frames.append({'index':16,'scene_id':'scene','split':'validation','visible_margin':{'gt_pixels':0,'f1':None},
            **metrics(confusion(np.zeros((4,4),dtype=np.uint8),np.full((4,4),12,dtype=np.uint8),17),SEGMENTATION_NAMES)})
        self.assertEqual(selected_cases(frames),[])
        cases=selected_cases(frames,True)
        self.assertEqual({c['sample']['index'] for c in cases},set(range(17)))
        self.assertEqual(cases[-1]['reason'],'most_Wall_false_positive_pixels')


if __name__=='__main__':unittest.main()
