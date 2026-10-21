import copy,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_safety import validate_evidence,project_to_path,path_positions,ProgressGuard
from senior_ai.navigation import Planner
import json

class Phase10SafetyTests(unittest.TestCase):
    def setUp(self):
        self.route={'local_xy_m':[[0,0],[4,0],[4,2],[3,4],[1,6]],'remaining_m':30,'cross_track_m':0,'goal_distance_m':30}
        self.ob={'valid':True,'timestamp_s':10.,'signals':[],'candidate_road_fractions':{'0.0':1}}
    def test_each_nonfinite_route_metric_rejected(self):
        for key in ['remaining_m','cross_track_m','goal_distance_m']:
            for value in [float('nan'),float('inf'),-1]:
                r=copy.deepcopy(self.route);r[key]=value
                with self.assertRaises(ValueError):validate_evidence(r,self.ob)
    def test_nonfinite_time_road_score_rejected(self):
        for value in [float('nan'),float('inf')]:
            ob=copy.deepcopy(self.ob);ob['timestamp_s']=value
            with self.assertRaises(ValueError):validate_evidence(self.route,ob)
            ob=copy.deepcopy(self.ob);ob['candidate_road_fractions']['0.0']=value
            with self.assertRaises(ValueError):validate_evidence(self.route,ob)
            ob=copy.deepcopy(self.ob);ob['signals']=[{'score':value,'phase':'red'}]
            with self.assertRaises(ValueError):validate_evidence(self.route,ob)
    def test_curved_vertical_and_nonmonotonic_x_project_exact(self):
        along,lateral=project_to_path([4,2],self.route['local_xy_m']);self.assertAlmostEqual(along,6);self.assertEqual(lateral,0)
    def test_future_position_follows_turn_not_straight_x(self):
        np.testing.assert_allclose(path_positions(self.route['local_xy_m'],[0,4,6]),[[0,0],[4,0],[4,2]])
    def test_no_progress_guard_exempts_long_signal_wait_and_restarts_grace(self):
        guard=ProgressGuard()
        for _ in range(200):self.assertFalse(guard.step(10,0,0,'SIGNAL_STOP',.2))
        for _ in range(50):self.assertFalse(guard.step(10,0,3,'CRUISE',.2))
        self.assertFalse(guard.step(10.2,1,3,'CRUISE',.2))
    def test_real_blocked_guard_still_stops(self):
        guard=ProgressGuard();outputs=[guard.step(1,0,0,'OBSTACLE_STOP',.2) for _ in range(65)]
        self.assertTrue(outputs[-1])
    def test_straight_projection_matches_longitudinal_distance(self):
        self.assertEqual(project_to_path([10,.3],[[1,0],[5,0],[15,0]]),(10.,.3))
    def test_bad_shape_and_degenerate_path_rejected(self):
        for p in [[1,2],[[0,0],[0,0]]]:
            with self.assertRaises(ValueError):project_to_path([2,0],p)
    def test_planner_rejects_nonfinite_provider_input(self):
        c=json.loads((ROOT/'configs/phase89_navigation_v1.json').read_text())
        self.ob['timestamp_s']=float('nan')
        with self.assertRaises(ValueError):Planner(c).plan(self.route,self.ob,2,10.1,.2)
    def test_planner_detects_obstacle_on_nonmonotonic_path(self):
        c=json.loads((ROOT/'configs/phase89_navigation_v1.json').read_text())
        self.ob['candidate_road_fractions']={str(x):1. for x in c['lateral_offsets_m']}
        self.ob['objects']=[{'center_ego_m':[4,2,1],'size_lwh_m':[1,1,2]}]
        result=Planner(c).plan(self.route,self.ob,2,10.1,.2)
        self.assertEqual(result['state'],'FOLLOW');self.assertIsNotNone(result['nearest_obstacle_m'])
        self.assertLess(result['target_speed_mps'],c['cruise_mps'])

if __name__=='__main__':unittest.main()
