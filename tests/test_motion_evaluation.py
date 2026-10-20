from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from evaluate_phase7_motion import tracking_scores,match

class TrackingMetricTests(unittest.TestCase):
    def test_IDF1_counts_global_identity_errors(self):
        frames=[{'scene':'s','frame':i,'truth':[{'actor_id':1}],'pred':[{'track_id':key}],'pairs':[(0,0)]} for i,key in enumerate(['a','b','b'])]
        score=tracking_scores(frames);self.assertEqual(score['ID_switches'],1);self.assertEqual(score['IDTP'],2)
        self.assertAlmostEqual(score['IDF1'],2/3);self.assertAlmostEqual(score['MOTA'],2/3)
    def test_missing_detection_is_FN_and_fragmentation(self):
        frames=[{'scene':'s','frame':i,'truth':[{'actor_id':1}],'pred':[] if i==1 else [{'track_id':'a'}],'pairs':[] if i==1 else [(0,0)]} for i in range(3)]
        score=tracking_scores(frames);self.assertEqual(score['FN'],1);self.assertEqual(score['fragmentations'],1);self.assertEqual(score.get('ID_switches',0),0)
        self.assertAlmostEqual(score['IDF1'],.8)
    def test_class_and_distance_domain_prevent_false_matches(self):
        p=[{'class_id':1,'center_ego_m':[0,0,0]}]
        self.assertEqual(match(p,[{'class_id':2,'center_ego_m':[0,0,0]}]),[])
        self.assertEqual(match(p,[{'class_id':1,'center_ego_m':[2.1,0,0]}]),[])
        self.assertEqual(match(p,[{'class_id':1,'center_ego_m':[2,0,0]}]),[(0,0)])

if __name__=='__main__':unittest.main()
