import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.bev_geometry import Camera,Grid,OPTICAL,cameras_from_calibration,projection_maps,warp_metric
from senior_ai.bev_fusion import deduplicate,fuse_road
from senior_ai.bev_temporal import fit_se2,CausalRoadFusion
from senior_ai.bev_path import lane_center

def camera(yaw=0,position=(1.6,0,1.7),name='front'):
    a=np.deg2rad(yaw);t=np.eye(4);t[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]];t[:3,3]=position
    return Camera({'name':name,'K':[[320,0,320],[0,320,180],[0,0,1]],'T_ego_from_camera_local':t})

class GeometryTests(unittest.TestCase):
    def test_eight_camera_roundtrip_translation(self):
        for yaw in [0,45,90,135,180,-135,-90,-45]:
            c=camera(yaw);p=c.T[:3,3]+np.array([10.,2.,-1.7])@c.T[:3,:3].T;uv,z=c.project(p[None]);back,valid=c.ground(uv)
            self.assertTrue(valid[0]);np.testing.assert_allclose(back[0],p,atol=1e-9);self.assertAlmostEqual(z[0],10.)
    def test_optical_axes_and_non_square_image(self):
        c=camera(position=(0,0,1.7));uv,z=c.project([[10,3,0]])
        np.testing.assert_allclose(uv,[[416,234.4]],atol=1e-9)
    def test_horizon_sky_behind_and_range_rejected(self):
        _,valid=camera().ground([[320,180],[320,100],[320,181],[700,300]])
        self.assertFalse(valid.any())
    def test_grid_orientation(self):
        g=Grid(0,10,-5,5,1);self.assertEqual(g.shape,(10,10));np.testing.assert_allclose(g.points()[0,0],[9.5,-4.5,0]);np.testing.assert_allclose(g.uv_to_xy(g.xy_to_uv([[2,3]])),[[2,3]])
    def test_invalid_grid_and_extrinsics(self):
        with self.assertRaises(ValueError):Grid(resolution_m=0)
        c=camera();t=c.T.copy();t[0,0]=2
        with self.assertRaises(ValueError):Camera({'name':'front','K':c.K,'T_ego_from_camera_local':t})
    def test_calibration_convention_and_names(self):
        with self.assertRaises(ValueError):cameras_from_calibration({'optical_from_carla_camera':np.eye(3).tolist(),'cameras':[]})
    def test_self_vehicle_mask(self):
        g=Grid(-3,5,-3,3,.1);m=projection_maps(camera(),g,vehicle_mask={'half_length_m':2.5,'half_width_m':1.1});p=g.points();self.assertFalse(m[1][(abs(p[...,0])<2.5)&(abs(p[...,1])<1.1)].any())
    def test_remap_array_indices_use_half_pixel_centers(self):
        g=Grid(5,15,-5,5,.5);c=camera();m=projection_maps(c,g);back,valid=c.ground(m[0]+.5);mask=m[1]&valid;np.testing.assert_allclose(back[mask],g.points()[mask],atol=1e-5)
    def test_metric_translation_direction(self):
        g=Grid(0,10,-5,5,1);image=np.zeros(g.shape,np.float32);image[3,4]=1;t=np.eye(3);t[0,2]=-2
        result=warp_metric(image,t,g,True);self.assertEqual(result[5,4],1);self.assertEqual(result.sum(),1)
    def test_non_rigid_warp_rejected(self):
        t=np.diag([2,2,1])
        with self.assertRaises(ValueError):warp_metric(np.zeros((10,10),np.float32),t,Grid(0,10,-5,5,1))

class FusionTests(unittest.TestCase):
    def obj(self,cam,x,label='car',i=0):return {'camera':cam,'xy_m':[x,0.],'class_name':label,'score':.9,'detection_index':i}
    def test_overlap_deduplicates_and_keeps_sources(self):
        r=deduplicate([self.obj('front',5),self.obj('front_right',5.3)]);self.assertEqual(len(r),1);self.assertEqual(len(r[0]['sources']),2)
    def test_same_camera_and_different_class_stay_separate(self):
        self.assertEqual(len(deduplicate([self.obj('front',5,i=0),self.obj('front',5.1,i=1),self.obj('right',5.1,'pedestrian')])),3)
    def test_chain_does_not_swallow_separate_objects(self):
        self.assertEqual(len(deduplicate([self.obj('front',5),self.obj('right',6.4),self.obj('rear',7.8)])),2)
    def test_occluding_pixels_do_not_become_road(self):
        c=camera();g=Grid(0,10,-5,5,.25);cfg=json.loads((ROOT/'configs/phase4_bev_v1.json').read_text());ob={'ids':np.full((360,640),5,np.uint8),'ground_probabilities':np.ones((360,640,2),np.float32)}
        r=fuse_road({'front':ob},{'front':c},g,cfg);self.assertFalse(r['observed'].any());self.assertTrue((r['geometric_coverage']>0).any())

class MotionTests(unittest.TestCase):
    def test_rigid_motion_with_outliers(self):
        rng=np.random.default_rng(5);a=rng.uniform(-10,10,(60,2));angle=.04;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);b=a@r.T+[-.8,.2];b[-10:]+=10
        result=fit_se2(a,b);self.assertIsNotNone(result);np.testing.assert_allclose(result['previous_to_current'][:2,:2],r,atol=1e-8);np.testing.assert_allclose(result['previous_to_current'][:2,2],[-.8,.2],atol=1e-8)
    def test_insufficient_and_nonrigid_motion(self):
        self.assertIsNone(fit_se2([],[]));a=np.random.default_rng(3).normal(size=(40,2));self.assertIsNone(fit_se2(a,a*1.3))
    def temporal(self):
        cfg=json.loads((ROOT/'configs/phase4_bev_v1.json').read_text());g=Grid(0,10,-5,5,1);return CausalRoadFusion(g,cfg),{'probabilities':np.ones(g.shape+(2,),np.float32),'observed':np.ones(g.shape,bool)}
    def test_cold_gap_reverse_and_scene_reset(self):
        t,b=self.temporal()
        for scene,ts,expected in [('a',1,'cold_start'),('a',2,'timestamp_gap_or_reversal'),('a',1,'timestamp_gap_or_reversal'),('b',1.1,'stream_identity_changed')]:
            _,info=t.update(b,{}, {},scene,ts,'calib');self.assertEqual(info['status'],expected);self.assertEqual(info['history_frames_used'],0)
    def test_bounded_history_and_current_occlusion(self):
        t,b=self.temporal();motion={'previous_to_current':np.eye(3),'inliers':20}
        with patch('senior_ai.bev_temporal.rgb_ground_motion',return_value=motion):
            for i in range(6):result,info=t.update(b,{}, {},'a',i*.2,'calib')
            self.assertEqual(info['history_frames_used'],2)
            b['observed'][:]=False;result,_=t.update(b,{}, {},'a',1.2,'calib');self.assertFalse(result['probabilities'].any())

class LaneTests(unittest.TestCase):
    def make(self,curve=0):
        g=Grid(0,22,-8,8,.1);xy=g.points();center=curve*xy[...,0]**2;line=(abs(xy[...,1]-center-1.75)<.12)|(abs(xy[...,1]-center+1.75)<.12);p=np.zeros(g.shape+(2,),np.float32);p[...,0]=1;p[...,1]=line
        cfg=json.loads((ROOT/'configs/phase4_bev_v1.json').read_text())['path'];return g,{'probabilities':p,'observed':np.ones(g.shape,bool)},cfg
    def test_straight_center(self):
        g,b,c=self.make();r=lane_center(b,g,c);self.assertEqual(r['status'],'candidate');self.assertLess(np.abs(np.asarray(r['points_ego_m'])[:,1]).max(),.1)
    def test_curved_center(self):
        g,b,c=self.make(.01);r=lane_center(b,g,c);self.assertEqual(r['status'],'candidate');p=np.asarray(r['points_ego_m']);self.assertLess(np.abs(p[:,1]-.01*p[:,0]**2).max(),.15)
    def test_no_lines_no_fabricated_path(self):
        g,b,c=self.make();b['probabilities'][...,1]=0;self.assertEqual(lane_center(b,g,c)['status'],'unavailable')
    def test_one_line_is_insufficient(self):
        g,b,c=self.make();c['allow_single_boundary_prior']=False;b['probabilities'][...,1][g.points()[...,1]>0]=0;self.assertEqual(lane_center(b,g,c)['status'],'unavailable')
    def test_single_boundary_prior_is_explicit(self):
        g,b,c=self.make();b['probabilities'][...,1][g.points()[...,1]>0]=0;r=lane_center(b,g,c);self.assertEqual(r['status'],'candidate');self.assertIn('fixed_width_prior',r['source']);self.assertEqual(r['width_prior_m'],3.5)
    def test_single_boundary_without_ground_rejected(self):
        g,b,c=self.make();b['probabilities'][...,1][g.points()[...,1]>0]=0;b['observed'][:]=False;self.assertEqual(lane_center(b,g,c)['status'],'unavailable')
    def test_bounded_dashed_mark_gaps(self):
        g,b,c=self.make();x=g.points()[...,0];b['probabilities'][...,1][(x%8)>3]=0;self.assertEqual(lane_center(b,g,c)['status'],'candidate')
    def test_large_missing_mark_gap_is_rejected(self):
        g,b,c=self.make();x=g.points()[...,0];b['probabilities'][...,1][(x>6)&(x<16)]=0;self.assertEqual(lane_center(b,g,c)['status'],'unavailable')

if __name__=='__main__':unittest.main()
