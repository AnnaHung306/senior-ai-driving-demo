import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.light_labels import phase_target


class ObservableLightLabelTests(unittest.TestCase):
    def test_hidden_phase_not_assigned_without_visible_lens(self):
        rgb=np.zeros((4,4,3),dtype=np.uint8)+60;mask=np.ones((4,4),dtype=bool)
        self.assertEqual(phase_target(rgb,mask,'Red')['light_state_class'],0)
        rgb[0,:2]=[240,20,10]
        self.assertEqual(phase_target(rgb,mask,'Red')['light_state_class'],1)
        self.assertEqual(phase_target(rgb,mask,'Green')['light_state_class'],0)
        rgb[0,:2]=[10,240,30]
        self.assertEqual(phase_target(rgb,mask,'Green')['light_state_class'],3)
        rgb[0,:2]=[240,220,10]
        self.assertEqual(phase_target(rgb,mask,'Yellow')['light_state_class'],2)
        mask[0,:]=False
        self.assertEqual(phase_target(rgb,mask,'Yellow')['light_state_class'],0)

if __name__=='__main__': unittest.main()
