import copy
import itertools
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.motion_forecast import MotionTracker,assignment,ego_forecast,swept_contact,collision_risk,appearance_histogram,source_overlap
C=json.loads((ROOT/'configs/phase7_motion_v1.json').read_text())

def obj(x=10,y=0,category=1):
    return {'class_id':category,'center_ego_m':[float(x),float(y),1.],'size_lwh_m':[4.,2.,2.],'yaw_rad':0.,'score_uncalibrated':.9,
            'track_id':'s:Told','measurement_current':True,'sources':[]}

def motion(dx=0,dy=0,angle=0):
    c,s=np.cos(angle),np.sin(angle)
    return {'status':'aligned','previous_to_current':[[c,-s,dx],[s,c,dy],[0,0,1]]}

class MotionTests(unittest.TestCase):
    def test_assignment_global_optimum_not_greedy(self):
        a=np.array([[1.,2.],[1.1,8.]])
        self.assertEqual(set(assignment(a)),{(0,1),(1,0)})
    def test_assignment_matches_exhaustive_random_rectangles(self):
        rng=np.random.default_rng(20261020)
        for n,d in [(1,3),(3,1),(3,3),(3,4)]:
            for _ in range(8):
                a=rng.uniform(0,25,(n,d));pairs=assignment(a,16.)
                actual=sum(a[i,j] for i,j in pairs)+(n-len(pairs))*16
                full=np.concatenate([a,np.full((n,n),16.)],axis=1)
                expected=min(sum(full[i,j] for i,j in enumerate(cols)) for cols in itertools.permutations(range(d+n),n))
                self.assertAlmostEqual(actual,expected)
    def test_assignment_gates_empty_and_nan(self):
        self.assertEqual(assignment(np.zeros((0,3))),[]);self.assertEqual(assignment([[np.inf]]),[])
        with self.assertRaises(ValueError):assignment([[np.nan]])
    def test_ground_velocity_subtracts_ego_translation(self):
        t=MotionTracker(C)
        for k in range(40):
            out,_=t.update([obj(15-k*.4)],'s',k*.2,'rig',motion(-.4))
        self.assertTrue(out[0]['motion']['velocity_available']);np.testing.assert_allclose(out[0]['motion']['velocity_ground_ego_mps'],[0,0],atol=1e-9)
    def test_ground_velocity_under_ego_rotation_and_translation(self):
        t=MotionTracker(C);p=np.array([15.,4.]);T=np.array(motion(-.1,.05,.03)['previous_to_current'])
        for k in range(35):
            if k:p=T[:2,:2]@p+T[:2,2]
            out,_=t.update([obj(*p)],'s',k*.2,'rig',motion(-.1,.05,.03))
        np.testing.assert_allclose(out[0]['motion']['velocity_ground_ego_mps'],[0,0],atol=1e-8)
    def test_converged_constant_velocity_and_fixed_ego_forecast(self):
        t=MotionTracker(C)
        for k in range(35):out,_=t.update([obj(5+k*.4,3+k*.2)],'s',k*.2,'rig',motion())
        o=out[0];np.testing.assert_allclose(o['motion']['velocity_ground_ego_mps'],[2,1],atol=1e-4)
        p=np.array(o['forecast']['positions_ego_xy_m']);np.testing.assert_allclose(p[-1],np.array(o['center_ego_m'][:2])+[6,3],atol=1e-4)
        self.assertEqual(o['forecast']['coordinate_frame'],'fixed_current_ego')
    def test_warmup_and_ambiguous_not_confident(self):
        t=MotionTracker(C);out,_=t.update([obj()],'s',0,'rig',motion());self.assertFalse(out[0]['motion']['velocity_available']);self.assertEqual(out[0]['risk']['status'],'unknown')
    def test_short_occlusion_retains_ID_not_occupancy_then_expires(self):
        t=MotionTracker(C);out,_=t.update([obj()],'s',0,'rig',motion());key=out[0]['track_id']
        for k in range(1,4):
            out,info=t.update([],'s',k*.2,'rig',motion());self.assertEqual(out,[]);self.assertEqual(info['held_unobserved_tracks'][0]['used_for_occupancy'],False)
        out,_=t.update([obj()],'s',.6,'rig',motion()) # same timestamp must reset, tested separately
        self.assertNotEqual(out[0]['track_id'],key)
        t=MotionTracker(C);out,_=t.update([obj()],'s',0,'rig',motion());key=out[0]['track_id']
        t.update([],'s',.2,'rig',motion());t.update([],'s',.4,'rig',motion());out,_=t.update([obj()],'s',.6,'rig',motion());self.assertEqual(out[0]['track_id'],key)
        for k in range(4,8):t.update([],'s',k*.2,'rig',motion())
        out,_=t.update([obj()],'s',1.6,'rig',motion());self.assertNotEqual(out[0]['track_id'],key)
    def test_crossing_same_class_with_appearance_and_predicted_velocity(self):
        t=MotionTracker(C);a=np.zeros(24);a[0]=1;b=np.zeros(24);b[8]=1;ids=None
        for k in range(31):
            objects=[obj(8+k*.2,-.15),obj(14-k*.2,.15)]
            out,_=t.update(objects,'s',k*.2,'rig',motion(),[a,b]);now=[o['track_id'] for o in out]
            if ids is None:ids=now
            self.assertEqual(ids,now)
    def test_prefix_causality_and_input_not_mutated(self):
        inputs=[obj(10+k*.2) for k in range(25)];old=copy.deepcopy(inputs)
        def run(n):
            t=MotionTracker(C);return [t.update([inputs[k]],'s',k*.2,'rig',motion()) for k in range(n)]
        self.assertEqual(run(10),run(25)[:10]);self.assertEqual(inputs,old)
    def test_gap_scene_calibration_and_bad_motion_reset(self):
        for scene,ts,cal,m in [('other',.2,'rig',motion()),('s',.2,'other',motion()),('s',1,'rig',motion()),('s',0,'rig',motion()),('s',.2,'rig',{'status':'failed'})]:
            t=MotionTracker(C);out,_=t.update([obj()],'s',0,'rig',motion());key=out[0]['track_id'];out,info=t.update([obj()],scene,ts,cal,m)
            self.assertIsNotNone(info['reset_reason']);self.assertNotEqual(key,out[0]['track_id']);self.assertFalse(out[0]['motion']['velocity_available'])
    def test_truth_nan_nonrigid_and_features_rejected(self):
        for key in ['actor_id','world_pose','velocity_mps','depth','future_positions']:
            t=MotionTracker(C);o=obj();o[key]=1
            with self.assertRaises(ValueError):t.update([o],'s',0,'rig',motion())
        t=MotionTracker(C);o=obj();o['center_ego_m'][0]=np.nan
        with self.assertRaises(ValueError):t.update([o],'s',0,'rig',motion())
        t.update([obj()],'s',0,'rig',motion());bad=motion();bad['previous_to_current'][0][0]=2
        with self.assertRaises(ValueError):t.update([obj()],'s',.2,'rig',bad)
    def test_history_and_covariance_are_bounded(self):
        t=MotionTracker(C)
        for k in range(300):out,_=t.update([obj(10+np.sin(k*.01))],'s',k*.2,'rig',motion())
        self.assertEqual(len(out[0]['history_ego_xy_m']),C['history_length']);tr=next(iter(t.tracks.values()))
        self.assertTrue((np.linalg.eigvalsh(tr['P'])>=-1e-12).all())
    def test_continuous_contact_catches_fast_crossing_between_samples(self):
        t,c=swept_contact([[10,0],[-10,0]],[0,.2],[1,1]);self.assertAlmostEqual(t,.09);self.assertLess(c,0)
        t,_=swept_contact([[10,10],[-10,10]],[0,.2],[1,1]);self.assertIsNone(t)
    def test_ego_twist_straight_and_turn(self):
        np.testing.assert_allclose(ego_forecast([2,1],0,[0,1,2]),[[0,0],[2,1],[4,2]])
        np.testing.assert_allclose(ego_forecast([2,0],np.pi/2,[1]),[[4/np.pi,4/np.pi]])
    def test_RGB_hist_is_normalized_and_uses_current_crops(self):
        image=np.zeros((20,20,3),np.uint8);image[:,:,0]=255
        h=appearance_histogram({'front':image},[{'camera':'front','bbox_xyxy':[0,0,20,20]}]);self.assertAlmostEqual(h.sum(),1);self.assertEqual(h[7],1/3)
    def test_2D_overlap_is_camera_local_and_v11_uses_current_visual_support(self):
        a=obj();b=obj();a['sources']=[{'camera':'front','bbox_xyxy':[0,0,10,10]}];b['sources']=[{'camera':'right','bbox_xyxy':[0,0,10,10]}]
        self.assertIsNone(source_overlap(a,b));b['sources'][0]['camera']='front';self.assertEqual(source_overlap(a,b),1.)
        cfg=copy.deepcopy(C);cfg.update(association_spatial_weight=2.,association_bbox_weight=3.)
        t=MotionTracker(cfg);a=obj(10,-.2);b=obj(10,.2)
        a['sources']=[{'camera':'front','bbox_xyxy':[10,10,20,30]}];b['sources']=[{'camera':'front','bbox_xyxy':[30,10,40,30]}]
        out,_=t.update([a,b],'s',0,'rig',motion());ids=[o['track_id'] for o in out]
        a['center_ego_m'][1]=.15;b['center_ego_m'][1]=-.15
        out,_=t.update([a,b],'s',.2,'rig',motion());self.assertEqual(ids,[o['track_id'] for o in out])
    def test_verified_backbone_preserves_source_ID_and_rewarms_outliers(self):
        cfg=copy.deepcopy(C);cfg['association_mode']='verified_upstream_identity';t=MotionTracker(cfg)
        for k in range(20):out,_=t.update([obj(10+k*.2)],'s',k*.2,'rig',motion())
        self.assertEqual(out[0]['track_id'],'s:Told');self.assertTrue(out[0]['motion']['velocity_available'])
        out,_=t.update([obj(25)],'s',4.,'rig',motion());self.assertEqual(out[0]['track_id'],'s:Told');self.assertFalse(out[0]['motion']['velocity_available'])
        self.assertEqual(out[0]['track_age_measurements'],21);self.assertEqual(out[0]['motion']['velocity_measurements'],1)
        a=obj();a['track_id']='s:Tnew';out,_=t.update([a],'s',4.2,'rig',motion());self.assertEqual(out[0]['track_id'],'s:Tnew')
        with self.assertRaises(ValueError):t.update([a,copy.deepcopy(a)],'s',4.4,'rig',motion())

if __name__=='__main__':unittest.main()
