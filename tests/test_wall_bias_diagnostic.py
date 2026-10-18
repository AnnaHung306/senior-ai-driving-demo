from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_segmentation_prior import correction_bias


class WallBiasDiagnosticTests(unittest.TestCase):
    def test_wall_only_mask_preserves_all_other_logit_channels(self):
        support=np.arange(1,18,dtype=np.float64);full,weights=correction_bias(support);wall,_=correction_bias(support,True)
        self.assertTrue(torch.equal(wall[12],full[12]));self.assertGreater(float(wall[12]),0)
        self.assertEqual(int(torch.count_nonzero(wall)),1)
        np.testing.assert_allclose(weights,np.clip(np.sqrt(17/support),1,20))

    def test_invalid_or_unmeasured_support_rejected(self):
        for support in [np.ones(16),np.zeros(17),np.full(17,np.nan),np.full(17,-1)]:
            with self.assertRaises(ValueError):correction_bias(support,True)


if __name__=='__main__':unittest.main()
