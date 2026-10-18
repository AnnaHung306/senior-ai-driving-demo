from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.light_joint_policy import ignored_unmatched_light


class LightJointIgnoreTests(unittest.TestCase):
    def test_unscorable_foreground_not_false_light(self):
        self.assertTrue(ignored_unmatched_light([2,2,8,8],[],[[0,0,10,10]]))

    def test_wrong_class_and_duplicate_known_positive_still_false_positive(self):
        for label in [5,7]:
            self.assertFalse(ignored_unmatched_light([2,2,8,8],[{'label':label,'box':[2,2,8,8]}],[[0,0,10,10]]))

    def test_real_background_false_light_not_ignored(self):
        self.assertFalse(ignored_unmatched_light([20,20,28,28],[],[[0,0,10,10]]))


if __name__=='__main__':unittest.main()
