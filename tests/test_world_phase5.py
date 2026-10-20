import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from senior_ai.bev_geometry import Camera
from senior_ai.cuboid import corners, box_iou, wrap, geometry_features, decode, fuse_cuboids, CuboidNet
from senior_ai.semantic_occupancy import VoxelGrid, fill_boxes, build_occupancy
from senior_ai.world_tracking import WorldTracker
from phase5_gt import target_box


def box(x=8., y=0., category=1, yaw=0., camera='front'):
    return {'class_id': category, 'class_name': str(category), 'center_ego_m': [x, y, 1.], 'size_lwh_m': [4., 2., 2.],
            'yaw_rad': yaw, 'orientation_reliable': True, 'score_uncalibrated': .9, 'camera': camera, 'detection_index': 0, 'output_clipped': False}


def camera():
    t = np.eye(4);t[2, 3] = 1.7
    return Camera({'name': 'front', 'K': [[320, 0, 320], [0, 320, 180], [0, 0, 1]], 'T_ego_from_camera_local': t.tolist()})


class CuboidTests(unittest.TestCase):
    def test_corners_full_dimensions_and_rotation(self):
        a = box(yaw=np.pi/2);p = corners(a)
        np.testing.assert_allclose(p.mean(0), a['center_ego_m'], atol=1e-12)
        np.testing.assert_allclose(np.ptp(p, axis=0), [2, 4, 2], atol=1e-12)

    def test_iou_overlap_height_and_nonoverlap(self):
        a = box();self.assertEqual(box_iou(a, a), (1., 1.))
        self.assertEqual(box_iou(a, box(x=20)), (0., 0.))
        b = box();b['center_ego_m'][2] = 5
        self.assertEqual(box_iou(a, b), (1., 0.))
        b = box(x=10);np.testing.assert_allclose(box_iou(a, b), [1/3, 1/3], atol=1e-6)

    def test_invalid_boxes_rejected(self):
        a = box();a['size_lwh_m'][0] = -1
        with self.assertRaises(ValueError):corners(a)

    def test_gt_center_and_local_bbox_rotation_from_vertices(self):
        a = box(x=6, yaw=.7);a['center_ego_m'][2] = 1.3
        actor = {'actor_id': 9, 'world_vertices_m': corners(a).tolist(), 'extent_m': (np.array(a['size_lwh_m'])/2).tolist()}
        t = np.eye(4);t[:3, 3] = [-1, 2, -3]
        result = target_box(actor, t)
        np.testing.assert_allclose(result['center_ego_m'], [5, 2, -1.7], atol=1e-12)
        self.assertAlmostEqual(result['yaw_rad'], .7)

    def test_gt_vertex_order_error_rejected(self):
        a = box();v = corners(a);v[[1, 4]] = v[[4, 1]]
        with self.assertRaises(ValueError):target_box({'actor_id': 1, 'world_vertices_m': v.tolist(), 'extent_m': [2, 1, 1]}, np.eye(4))

    def test_features_require_visible_ground_anchor_and_class(self):
        c = camera();f, anchor = geometry_features(c, [280, 170, 340, 250], 1)
        self.assertEqual(f.shape, (14,));self.assertTrue(np.isfinite(f).all());self.assertAlmostEqual(anchor[2], -1.7)
        self.assertIsNone(geometry_features(c, [0, 170, 340, 250], 1))
        self.assertIsNone(geometry_features(c, [280, 20, 340, 90], 1))
        self.assertIsNone(geometry_features(c, [280, 170, 340, 250], 7))

    def test_decode_transforms_center_and_yaw(self):
        c = camera();t = c.T.copy();t[:2, :2] = [[0, -1], [1, 0]]
        c = Camera({'name': 'right', 'K': c.K, 'T_ego_from_camera_local': t})
        p = decode([0, 0, .5, 0, 0, 0, 1, 0], np.array([8, 0, -1.7]), c, 1, {'1': [4, 2, 2]})
        np.testing.assert_allclose(p['center_ego_m'], [0, 8, 1], atol=1e-12);self.assertAlmostEqual(p['yaw_rad'], np.pi/2)

    def test_fusion_wrap_and_complete_link(self):
        a = box(yaw=np.deg2rad(179));b = box(x=8.1, yaw=np.deg2rad(-179), camera='left')
        fused = fuse_cuboids([a, b]);self.assertEqual(len(fused), 1);self.assertAlmostEqual(abs(fused[0]['yaw_rad']), np.pi)
        self.assertEqual(len(fuse_cuboids([a, box(x=9, camera='left'), box(x=10, camera='right')])), 2)

    def test_fusion_no_same_camera_or_cross_class(self):
        self.assertEqual(len(fuse_cuboids([box(), box(x=8.1)])), 2)
        self.assertEqual(len(fuse_cuboids([box(), box(category=2, camera='left')])), 2)

    def test_network_backward_and_serialization(self):
        torch.set_num_threads(2);torch.manual_seed(2);net = CuboidNet()
        x = torch.rand(2, 3, 64, 64);f = torch.rand(2, 14);p = net(x, f);p.square().mean().backward()
        self.assertGreater(float(net.features[0].weight.grad.norm()), 0)
        other = CuboidNet();other.load_state_dict(net.state_dict());self.assertTrue(torch.equal(net(x, f), other(x, f)))


class WorldTests(unittest.TestCase):
    def tracker(self):return WorldTracker({'maximum_gap_s': .4, 'association_distance_m': 2.5, 'maximum_missed_frames': 2})
    def motion(self, x=0):
        t = np.eye(3);t[0, 2] = x
        return {'status': 'aligned', 'previous_to_current': t.tolist()}

    def test_ego_compensated_ids_and_held_not_occupied(self):
        tracker = self.tracker();a, _ = tracker.update([box()], 'a', 1., 'sha', {})
        b, _ = tracker.update([box(x=4)], 'a', 1.2, 'sha', self.motion(-4))
        self.assertEqual(a[0]['track_id'], b[0]['track_id'])
        empty, meta = tracker.update([], 'a', 1.4, 'sha', self.motion());self.assertEqual(empty, [])
        self.assertFalse(meta['held_tracks_used_for_occupancy']);self.assertEqual(len(meta['held_unobserved_tracks']), 1)
        c, _ = tracker.update([box(x=4)], 'a', 1.6, 'sha', self.motion());self.assertEqual(c[0]['track_id'], b[0]['track_id'])

    def test_all_stream_reset_conditions_keep_ids_unique(self):
        for scene, timestamp, calibration, motion in [('b', 1.2, 'sha', self.motion()), ('a', 2., 'sha', self.motion()),
                ('a', .5, 'sha', self.motion()), ('a', 1.2, 'new', self.motion()), ('a', 1.2, 'sha', {})]:
            tracker = self.tracker();a, _ = tracker.update([box()], 'a', 1., 'sha', {})
            b, meta = tracker.update([box()], scene, timestamp, calibration, motion)
            self.assertNotEqual(a[0]['track_id'], b[0]['track_id']);self.assertIsNotNone(meta['reset_reason'])

    def test_tracking_class_gate_and_one_to_one(self):
        tracker = self.tracker();a, _ = tracker.update([box(), box(x=10)], 'a', 1., 's', {})
        b, _ = tracker.update([box(x=8.1), box(x=8.2), box(category=2)], 'a', 1.2, 's', self.motion())
        self.assertEqual(len({x['track_id'] for x in b}), 3)
        self.assertNotIn(b[-1]['track_id'], {x['track_id'] for x in a})

    def test_nonrigid_motion_rejected(self):
        tracker = self.tracker();tracker.update([box()], 'a', 1., 's', {})
        m = self.motion();m['previous_to_current'][0][0] = 2
        with self.assertRaises(ValueError):tracker.update([box()], 'a', 1.2, 's', m)

    def test_voxel_axes_boundaries_and_centers(self):
        g = VoxelGrid();xyz = g.centers();indices, valid = g.index(xyz)
        self.assertTrue(valid.all());np.testing.assert_array_equal(indices, np.moveaxis(np.indices(g.shape), 0, -1))
        _, valid = g.index([[0, 0, 4.5], [35, 0, 0], [-15.01, 0, 0], [0, 20, 0], [np.nan, 0, 0]])
        self.assertFalse(valid.any())

    def test_unknown_not_free_and_box_priority(self):
        g = VoxelGrid();values = np.full(g.shape, 255, np.uint8);fill_boxes(values, g, [box()])
        self.assertGreater(int((values == 3).sum()), 0);self.assertEqual(int((values == 0).sum()), 0)
        values[:] = 0;fill_boxes(values, g, [box()]);self.assertGreater(int((values == 3).sum()), 0)

    def test_ground_rays_do_not_fill_all_unknown_space(self):
        g = VoxelGrid();c = camera();ids = np.zeros((360, 640), np.uint8);ids[225:280, 280:360] = 1
        values, meta = build_occupancy(g, {'front': c}, {'front': ids}, [])
        self.assertGreater(int((values == 0).sum()), 0);self.assertGreater(int((values == 1).sum()), 0)
        self.assertGreater(float((values == 255).mean()), .9);self.assertFalse(meta['fine_geometry_reconstructed'])
        self.assertEqual(int((values[..., :2] == 0).sum()), 0)


if __name__ == '__main__':unittest.main()
