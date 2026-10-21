import copy,json,math,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation import RouteCursor,Planner,Controller,ego_coordinates,validate_route,validate_maneuvers

class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((ROOT/'configs/phase89_navigation_v1.json').read_text())
        self.route={'local_xy_m':[[x,0] for x in range(1,16)],'remaining_m':30.,'cross_track_m':0.,'goal_distance_m':30.}
        self.ob={'valid':True,'timestamp_s':10.,'objects':[],'signals':[],'candidate_road_fractions':{'0.0':1.,'-0.45':1.,'0.45':1.}}
    def run_plan(self,ob=None,route=None,speed=2):return Planner(self.c).plan(route or self.route,ob or self.ob,speed,10.1)
    def test_transform_right_hand_navigation(self):np.testing.assert_allclose(ego_coordinates([[10,5]],[10,0,math.pi/2]),[[5,0]],atol=1e-10)
    def test_route_validation(self):
        for r in [[[0,0],[100,0],[101,0]],[[0,0],[1,0]],[[0,0],[1,float('nan')],[2,0]]]:
            with self.assertRaises(ValueError):validate_route(r)
    def test_unverified_lane_changes_are_rejected(self):
        self.assertTrue(validate_maneuvers(['LEFT','LANEFOLLOW']))
        for options in [[],['CHANGELANELEFT'],['RIGHT','CHANGELANERIGHT'],['VOID']]:
            with self.assertRaises(ValueError):validate_maneuvers(options)
    def test_cursor_segment_projection_and_no_backward_progress(self):
        c=RouteCursor([[x,0] for x in range(31)]);a=c.update([5.4,.3,0]);b=c.update([4.9,.2,0]);self.assertAlmostEqual(a['cross_track_m'],.3);self.assertEqual(b['progress_m'],a['progress_m'])
    def test_controller_follows_right_and_rate_limit(self):
        p=self.run_plan();p['local_path_xy_m']=[[x,2] for x in range(1,16)];o=Controller().step(p,2,.1);self.assertGreater(o['steer'],0);self.assertLessEqual(o['steer'],.12)
    def test_cruise_uses_RGB_road(self):self.assertEqual(self.run_plan()['state'],'CRUISE')
    def test_no_road_is_failsafe(self):
        self.ob['candidate_road_fractions']={};self.assertEqual(self.run_plan()['target_speed_mps'],0)
    def test_obstruction_is_named_when_RGB_road_unavailable(self):
        self.ob['candidate_road_fractions']={};self.ob['objects']=[{'center_ego_m':[9,0,1],'size_lwh_m':[4,2,2]}]
        result=self.run_plan();self.assertEqual(result['state'],'OBSTACLE_STOP');self.assertEqual(result['target_speed_mps'],0)
        self.ob['objects'][0]['center_ego_m'][1]=8;self.assertEqual(self.run_plan()['state'],'FAILSAFE')
    def test_stale_missing_and_future(self):
        for timestamp in [9.,11.]:
            o=copy.deepcopy(self.ob);o['timestamp_s']=timestamp;self.assertEqual(self.run_plan(o)['state'],'FAILSAFE')
        o=copy.deepcopy(self.ob);o['valid']=False;self.assertEqual(self.run_plan(o)['state'],'FAILSAFE')
    def test_reversed_clock_rejected(self):
        p=Planner(self.c);p.plan(self.route,self.ob,2,10.1)
        with self.assertRaises(ValueError):p.plan(self.route,self.ob,2,10.)
    def test_red_requires_stable_green(self):
        p=Planner(self.c)
        self.ob['signals']=[{'phase':'red','score':.9}]
        self.assertEqual(p.plan(self.route,self.ob,2,10.1)['state'],'SIGNAL_STOP')
        for i in range(3):
            self.ob['timestamp_s']+=.2;self.ob['signals'][0]['phase']='green';r=p.plan(self.route,self.ob,0,10.3+i*.2)
            self.assertEqual(r['state'],'CRUISE' if i==2 else 'SIGNAL_STOP')
    def test_unknown_signal_holds_and_disappearance_not_green(self):
        p=Planner(self.c);self.ob['signals']=[{'phase':'unknown_or_unobservable','score':.9}]
        self.assertEqual(p.plan(self.route,self.ob,0,10.1)['state'],'SIGNAL_STOP');self.ob['signals']=[];self.ob['timestamp_s']=10.2
        self.assertEqual(p.plan(self.route,self.ob,0,10.3)['state'],'SIGNAL_STOP')
    def test_blocker_stops_following_and_crossing(self):
        for class_id in [1,2]:
            o=copy.deepcopy(self.ob);o['objects']=[{'class_id':class_id,'center_ego_m':[5,0,1],'size_lwh_m':[4,2,2]}]
            self.assertEqual(self.run_plan(o)['state'],'OBSTACLE_STOP')
    def test_distant_lead_lowers_speed(self):
        self.ob['objects']=[{'center_ego_m':[10,0,1],'size_lwh_m':[4,2,2]}];p=self.run_plan();self.assertEqual(p['state'],'FOLLOW');self.assertLess(p['target_speed_mps'],3.)
    def test_adjacent_object_does_not_stop(self):
        self.ob['objects']=[{'center_ego_m':[8,6,1],'size_lwh_m':[4,2,2]}];self.assertEqual(self.run_plan()['state'],'CRUISE')
    def test_valid_same_lane_offset_clears_small_obstacle(self):
        self.ob['objects']=[{'center_ego_m':[6,1.45,1],'size_lwh_m':[.3,.2,1]}];self.assertEqual(self.run_plan()['selected_offset_m'],-.45)
    def test_unknown_offset_never_used(self):
        self.ob['candidate_road_fractions']={'0.0':1.};self.ob['objects']=[{'center_ego_m':[5,0,1],'size_lwh_m':[4,2,2]}];self.assertEqual(self.run_plan()['state'],'OBSTACLE_STOP')
    def test_goal_brake_then_arrive(self):
        self.route['goal_distance_m']=2.;self.assertEqual(self.run_plan()['state'],'GOAL_BRAKE');self.assertEqual(self.run_plan(speed=.1)['state'],'ARRIVED')
    def test_brake_has_no_throttle(self):
        for state in ['FAILSAFE','ARRIVED','OBSTACLE_STOP','SIGNAL_STOP']:
            self.assertEqual(Controller().step({'state':state,'target_speed_mps':0},3,.2)['brake'],1.)
    def test_curved_route_reduces_speed(self):
        self.route['local_xy_m']=[[x,x] for x in range(1,16)];self.assertLess(self.run_plan()['target_speed_mps'],2.)
    def test_large_deviation_stops(self):self.route['cross_track_m']=2.;self.assertEqual(self.run_plan()['state'],'FAILSAFE')
    def test_held_obstacle_remains_a_planning_constraint(self):
        self.ob['held_objects']=[{'center_ego_m':[5,0,1],'size_lwh_m':[4,2,2],'age_since_measurement_s':.4,'measurement_current':False,'used_for_occupancy':False}]
        self.assertEqual(self.run_plan()['state'],'OBSTACLE_STOP')
        self.assertEqual(self.ob['objects'],[])
        self.ob['held_objects'][0]['age_since_measurement_s']=.8
        self.assertEqual(self.run_plan()['state'],'CRUISE')
    def test_nonfinite_objects_and_feedback_rejected(self):
        self.ob['objects']=[{'center_ego_m':[float('nan'),0,1],'size_lwh_m':[4,2,2]}]
        with self.assertRaises(ValueError):self.run_plan()
        with self.assertRaises(ValueError):Controller().step(self.run_plan({'valid':False}),float('nan'),.2)
    def test_synthetic_bicycle_straight_and_turn_closed_loop(self):
        # Analytic plant diagnostic, not CARLA or learned-model quality.
        for curved in [False,True]:
            t=np.linspace(0,math.pi/2,48)
            points=np.c_[15*np.sin(t),15*(1-np.cos(t))] if curved else np.c_[np.arange(31),np.zeros(31)]
            cursor=RouteCursor(points);planner=Planner(self.c);controller=Controller();pose=np.zeros(3);speed=0.;worst=0.;arrived=False
            for i in range(600):
                route=cursor.update(pose);ob=copy.deepcopy(self.ob);ob['timestamp_s']=i*.1
                plan=planner.plan(route,ob,speed,i*.1+.01);control=controller.step(plan,speed,.1)
                speed=max(0.,speed+(.1*(3*control['throttle']-5*control['brake']-.12*speed)))
                pose[2]+=.1*speed/2.85*math.tan(control['steer']*.6)
                pose[:2]+=.1*speed*np.array([math.cos(pose[2]),math.sin(pose[2])])
                worst=max(worst,route['cross_track_m'])
                if plan['state']=='ARRIVED':arrived=True;break
            self.assertTrue(arrived);self.assertLess(worst,1.5);self.assertLessEqual(speed,.3)
if __name__=='__main__':unittest.main()
