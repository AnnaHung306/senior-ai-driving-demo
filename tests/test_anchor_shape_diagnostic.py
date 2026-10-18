"""Official rounded anchor shapes, with no grid-coverage or AP claims."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_sunny_box_sizes import centered_anchor_iou


class AnchorShapeDiagnosticTests(unittest.TestCase):
    def test_adding_small_shape_preserves_original_matches_and_improves_tiny_box_bound(self):
        wh=[[16,16],[5,9],[32,64],[200,300]]
        current=centered_anchor_iou(wh,[16,32,64,128,256])
        smaller=centered_anchor_iou(wh,[8,16,32,64,128,256])
        self.assertEqual(current[0],1.)
        self.assertTrue(np.all(smaller>=current))
        self.assertLess(current[1],.25);self.assertGreater(smaller[1],.5)


if __name__=='__main__':unittest.main()
