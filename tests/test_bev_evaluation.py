import unittest,tempfile,json,sys
from unittest.mock import patch
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from evaluate_phase4_pilot import road_counts,line_counts,path_error,finalize

class EvalTests(unittest.TestCase):
    def test_unknown_prediction_counts_as_false_negative(self):
        p=np.zeros((5,5,2));mask=np.ones((5,5),bool);r=road_counts(p,~mask,mask,mask);self.assertEqual(r['FN'],25);self.assertEqual(r['TP'],0)
    def test_ground_truth_domain_is_independent_of_prediction(self):
        p=np.ones((5,5,2));mask=np.zeros((5,5),bool);mask[2,2]=1;r=road_counts(p,np.ones_like(mask),np.zeros_like(mask),mask);self.assertEqual(r['FP'],1);self.assertEqual(r['GT_visible_ground_cells'],1)
    def test_line_distance_tolerance(self):
        gt=np.zeros((20,20),bool);gt[:,10]=1;domain=np.ones_like(gt);p=np.zeros((20,20,2));p[:,12,1]=1;r=line_counts(p,domain,gt,domain);self.assertEqual(r['matched_predicted'],20);p[:]=0;p[:,14,1]=1;self.assertEqual(line_counts(p,domain,gt,domain)['matched_predicted'],0)
    def test_continuous_lane_across_road_identifiers(self):
        roads={'waypoints':[{'road_id':1 if x<8 else 2,'lane_id':-1,'is_junction':False,'pose':{'position_m':[x,0,0],'rotation_deg':[0,0,0]}} for x in [0,5,8,13,18,23]]}
        metadata={'path':{'status':'candidate','points_ego_m':[[x,.2] for x in np.arange(4.5,15,.5)]}}
        with patch('evaluate_phase4_pilot.read',return_value=roads):
            r=path_error(Path('unused_read_is_mocked'),metadata,np.eye(4),{'roads':'roads.json'})
        self.assertIsNotNone(r);self.assertAlmostEqual(r['lateral_rmse_m'],.2)

if __name__=='__main__':unittest.main()
