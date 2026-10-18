from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_sunny_detection_artifacts import expected_targets,validate_predictions


class DetectionArtifactValidationTests(unittest.TestCase):
    def test_canonical_clipping_phase_and_float32_ignore_preserved(self):
        labels={'positive_objects':[{'visible_bbox_xyxy':[0,1,640,8],'category_id':7,'visible_pixels':42,'light_state_class':1},
            {'visible_bbox_xyxy':[4,5,12,15],'category_id':3,'visible_pixels':64}],
            'ignore_regions':[{'bbox_xyxy':[1/3,0,8,15]}]}
        t,ignored=expected_targets(labels)
        self.assertTrue(t[0]['truncated']);self.assertFalse(t[1]['truncated'])
        self.assertEqual(t[0]['light_state_class'],1);self.assertEqual(t[1]['light_state_class'],-1)
        self.assertEqual(ignored[0][0],float(np.float32(1/3)))
        self.assertNotEqual(ignored[0][0],1/3)

    def test_invalid_predictions_rejected(self):
        for p in [{'box':[0,0,1,1],'label':0,'score':.9},{'box':[0,0,1,1],'label':3,'score':float('nan')},
            {'box':[0,0,641,1],'label':3,'score':.9},{'box':[2,0,1,1],'label':3,'score':.9}]:
            with self.subTest(p=p),self.assertRaises(ValueError):validate_predictions([p])

    def test_original_coordinate_predictions_and_empty_targets_valid(self):
        validate_predictions([{'box':[0.,0.,640.,360.],'label':3,'score':.5}])
        self.assertEqual(expected_targets({'positive_objects':[],'ignore_regions':[]}),([],[]))


if __name__=='__main__':unittest.main()
