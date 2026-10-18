"""CPU-only contracts for the optional post-training prior correction."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_segmentation_prior import corrected_bias,validate_grid


class PriorCalibrationTests(unittest.TestCase):
    def test_alpha_zero_identity_and_no_input_mutation(self):
        original=torch.tensor([2.,-1.,0.]);snapshot=original.clone()
        result=corrected_bias(original,[100,1,0],0)
        self.assertTrue(torch.equal(result,original))
        self.assertTrue(torch.equal(original,snapshot))

    def test_train_weight_formula_and_cap(self):
        original=torch.tensor([1.,2.,3.])
        result=corrected_bias(original,[10000,100,0],1)
        self.assertTrue(torch.allclose(result,original-torch.tensor([1.,10.,20.]).log()))

    def test_partial_alpha_is_linear(self):
        b=torch.tensor([2.,-1.]);s=[100,1]
        self.assertTrue(torch.allclose(corrected_bias(b,s,.5),(b+corrected_bias(b,s,1))/2))

    def test_reject_unbounded_duplicate_nonfinite_or_bool_grid(self):
        for grid in [[],[.5,1],[0,0],[0,2],[0,-1],[0,float('nan')],[0,float('inf')],[0,True],None]:
            with self.subTest(grid=grid),self.assertRaises(ValueError):validate_grid(grid)
        self.assertEqual(validate_grid([0,.25,.5,.75,1]),[0.,.25,.5,.75,1.])

    def test_reject_wrong_or_invalid_support(self):
        for support in [[1],[1,-1],[1,float('nan')],[1,float('inf')]]:
            with self.subTest(support=support),self.assertRaises(ValueError):corrected_bias(torch.zeros(2),support,.5)


if __name__=='__main__':unittest.main()
