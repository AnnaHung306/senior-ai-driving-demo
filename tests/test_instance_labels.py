from itertools import product
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from senior_ai.instance_labels import decode_instance_rgb, depth_box_support, mask_bbox, project_box, validate_actor_ids, visible_candidate


class InstanceLabelTests(unittest.TestCase):
    def test_channel_order_and_actor_id_alias_guard(self):
        classes, ids = decode_instance_rgb(np.array([[[19, 0x34, 0x12], [13, 255, 255]]], np.uint8))
        self.assertEqual(classes.tolist(), [[19, 13]])
        self.assertEqual(ids.tolist(), [[0x1234, 65535]])
        for actors in [[{'actor_id': 65536, 'kind': 'car'}],
                       [{'actor_id': 2, 'kind': 'car'}, {'actor_id': 2, 'kind': 'pedestrian'}]]:
            with self.assertRaises(ValueError):
                validate_actor_ids(actors)

    def test_exclusive_mask_box_and_empty(self):
        self.assertEqual(mask_bbox(np.array([[False, True], [False, False]])), [1, 0, 2, 1])
        self.assertIsNone(mask_bbox(np.zeros((2, 2), bool)))

    def test_projection_near_plane_behind_and_outside(self):
        vertices = np.array(list(product([9., 11.], [-1., 1.], [-1., 1.])))
        k = np.array([[100., 0, 100], [0, 100, 50], [0, 0, 1]])
        result = project_box(vertices, np.eye(4), k, 200, 100)
        self.assertTrue(np.allclose(result['bbox_xyxy'], [100 - 100/9, 50 - 100/9, 100 + 100/9, 50 + 100/9]))
        self.assertIsNone(project_box(-vertices, np.eye(4), k, 200, 100)['bbox_xyxy'])
        cross = vertices.copy(); cross[:, 0] -= 10
        result = project_box(cross, np.eye(4), k, 200, 100)
        self.assertTrue(result['near_plane_clipped'])
        self.assertEqual(result['bbox_xyxy'], [0, 0, 200, 100])
        outside = vertices.copy(); outside[:, 1] += 100
        self.assertTrue(project_box(outside, np.eye(4), k, 200, 100)['outside_view'])
        vertices[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            project_box(vertices, np.eye(4), k, 200, 100)

    def test_depth_axial_and_range_are_distinct_off_axis(self):
        k = [[100, 0, .5], [0, 100, .5], [0, 0, 1]]
        mask = np.zeros((1, 101), bool); mask[0, 100] = True
        depth = np.full((1, 101), 10.)
        world_box = np.eye(4); world_box[:3, 3] = [10, 10, 0]
        result = depth_box_support(mask, depth, k, np.eye(4), world_box, [.5, .5, .5])
        self.assertEqual(result['axial_in_box_fraction'], 1)
        self.assertEqual(result['range_in_box_fraction'], 0)
        result = depth_box_support(mask, depth * np.sqrt(2), k, np.eye(4), world_box, [.5, .5, .5])
        self.assertEqual(result['range_in_box_fraction'], 1)
        self.assertEqual(result['axial_in_box_fraction'], 0)

    def test_rider_union_and_no_visible_mask_means_no_box(self):
        actor = {'actor_id': 1, 'kind': 'bicycle', 'extent_m': [1, 1, 1],
                 'world_vertices_m': list(product([9., 11.], [-1., 1.], [-1., 1.]))}
        actor['T_world_from_box'] = np.eye(4); actor['T_world_from_box'][0, 3] = 10
        camera = {'K': [[100, 0, 1.5], [0, 100, .5], [0, 0, 1]], 'T_world_from_camera': np.eye(4)}
        classes = np.array([[19, 13, 12]], np.uint8); ids = np.array([[1, 1, 2]], np.uint16)
        result = visible_candidate(actor, classes, ids, classes, np.full((1, 3), 10.), camera)
        self.assertEqual(result['visible_bbox_xyxy'], [0, 0, 2, 1])
        self.assertEqual(result['semantic_pixels_by_tag'], {'13': 1, '19': 1})
        self.assertFalse(result['unexpected_semantic_tags'])
        actor['kind'] = 'car'
        self.assertEqual(visible_candidate(actor, classes, ids, classes, np.full((1, 3), 10.), camera)['unexpected_semantic_tags'], [13, 19])
        actor['actor_id'] = 3
        result = visible_candidate(actor, classes, ids, classes, np.full((1, 3), 10.), camera)
        self.assertIsNone(result['visible_bbox_xyxy'])
        self.assertEqual(result['status'], 'not_visible')
