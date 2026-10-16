import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.road_boundaries import visible_margin,boundary_counts,boundary_metrics


class VisibleMarginTests(unittest.TestCase):
    def test_road_ground_interface_excludes_vehicle_occlusion(self):
        classes=np.ones((10,10),dtype=np.uint8);classes[:,7:]=16;classes[3:5,3:5]=5
        border=visible_margin(classes)
        self.assertEqual(int(border.sum()),10);self.assertTrue(border[:,6].all())
        self.assertFalse(border[2:6,2:6].any())
        self.assertEqual(boundary_metrics(boundary_counts(classes,classes))['f1'],1.)

    def test_empty_negative_and_ignored_boundary(self):
        target=np.ones((8,8),dtype=np.uint8);pred=target.copy();pred[:,4:]=2
        r=boundary_metrics(boundary_counts(target,pred));self.assertEqual(r['f1'],0.)
        target[:,4:]=255
        r=boundary_metrics(boundary_counts(target,pred));self.assertIsNone(r['f1'])

if __name__=='__main__':unittest.main()
