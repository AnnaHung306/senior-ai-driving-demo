import sys
from pathlib import Path
from itertools import product
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.acquisition_safety import forecast_conflict,tm_target_speed


def box(x,y): return [[x+a,y+b,c] for a,b,c in product([-2,2],[-.8,.8],[0,1.5])]


class AcquisitionSafetyTests(unittest.TestCase):
    def test_synchronized_forecast(self):
        self.assertTrue(forecast_conflict(box(0,0),[3,0,0],box(10,0),[-3,0,0]))
        self.assertFalse(forecast_conflict(box(0,0),[3,0,0],box(0,3.5),[3,0,0]))
        self.assertFalse(forecast_conflict(box(0,0),[0,0,0],box(20,0),[0,0,0]))
        with self.assertRaises(ValueError): forecast_conflict(box(0,0),[float('nan'),0,0],box(10,0),[0,0,0])

    def test_tm_speed_never_zero(self):
        self.assertEqual(tm_target_speed(50,True),.1)
        self.assertEqual(tm_target_speed(0,False),12)
        self.assertEqual(tm_target_speed(50,False),12)

if __name__=='__main__':unittest.main()
