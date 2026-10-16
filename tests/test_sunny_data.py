import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.sunny_data import segmentation_target,SunnySamples


class SunnyLabelBoundaryTests(unittest.TestCase):
    def test_static_labels_use_actual_carla_enum_not_legacy_guess(self):
        import carla
        raw=np.array([[int(getattr(carla.CityObjectLabel,name)) for name in ['Walls','Fences','Poles','Sky']]],dtype=np.uint8)
        instance=np.zeros((1,4,3),dtype=np.uint8);instance[...,0]=raw
        self.assertEqual(segmentation_target(raw,instance,[]).tolist(),[[12,13,14,0]])

    def test_dynamic_tag_requires_verified_instance_id(self):
        raw=np.array([[21,21,13,18,0]],dtype=np.uint8)
        instance=np.zeros((1,5,3),dtype=np.uint8);instance[...,0]=raw;instance[0,:,1]=[123,124,3,3,0]
        target=segmentation_target(raw,instance,[123])
        self.assertEqual(target.tolist(),[[15,255,4,8,255]])
        instance[0,0,0]=20
        self.assertEqual(int(segmentation_target(raw,instance,[123])[0,0]),255)

    def test_lockbox_blocked_before_any_file_access(self):
        with self.assertRaisesRegex(ValueError,'Lockbox'):
            SunnySamples('does-not-exist','lockbox','detection')


if __name__=='__main__':unittest.main()
