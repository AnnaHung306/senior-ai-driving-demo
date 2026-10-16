import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.visibility_policy import classify_candidate, export_view, ignore_background_proposals


class VisibilityPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / 'configs/visibility_policy_v1.json').read_text())
        self.candidate = {'actor_id': 1, 'kind': 'car', 'visible_pixels': 64, 'visible_bbox_xyxy': [0, 0, 8, 8],
            'unexpected_semantic_tags': [], 'instance_semantic_disagreement_pixels': 0,
            'semantic_pixels_by_tag': {'14': 64}, 'depth_geometry': {'axial_in_box_fraction': 1.},
            'amodal_projection': {'near_plane_clipped': False, 'bbox_xyxy': [0, 0, 8, 8]}}

    def test_rider_only_is_ignored_but_verified_body_and_rider_is_kept(self):
        c = copy.deepcopy(self.candidate); c['kind'] = 'motorcycle'
        c['semantic_pixels_by_tag'] = {'13': 64}
        status, reasons = classify_candidate(c, self.policy)
        self.assertEqual(status, 'ignore')
        self.assertIn('two_wheeler_body_not_visible', reasons)
        c['semantic_pixels_by_tag'] = {'13': 32, '18': 32}
        self.assertEqual(classify_candidate(c, self.policy), ('positive', []))

    def test_unknown_foreground_is_ignore_never_background(self):
        classes = np.full((8, 16), 14, np.uint8); classes[:, 8:] = 7
        ids = np.ones((8, 16), np.uint16); ids[:, 8:] = 9
        result, mask = export_view([self.candidate], classes, ids, self.policy)
        self.assertEqual(len(result['positive_objects']), 1)
        self.assertEqual(result['ignore_regions'][0]['bbox_xyxy'], [8, 0, 16, 8])
        self.assertEqual(int(mask[:, :8].sum()), 0)
        self.assertEqual(int(mask[:, 8:].sum()), 64)

    def test_hidden_actor_does_not_create_amodal_ignore_region(self):
        c = copy.deepcopy(self.candidate); c['actor_id'] = 2; c['visible_pixels'] = 0; c['visible_bbox_xyxy'] = None
        result, mask = export_view([self.candidate, c], np.full((8, 8), 14), np.ones((8, 8)), self.policy)
        self.assertEqual(result['not_visible_objects'], 1)
        self.assertFalse(result['ignore_regions'])
        self.assertEqual(int(mask.sum()), 0)

    def test_near_plane_bad_depth_and_geometry_are_explicit_ignore(self):
        c = copy.deepcopy(self.candidate)
        c['amodal_projection']['near_plane_clipped'] = True
        c['depth_geometry']['axial_in_box_fraction'] = .2
        c['visible_bbox_xyxy'][2] = 100
        status, reasons = classify_candidate(c, self.policy)
        self.assertEqual(status, 'ignore')
        self.assertEqual(set(reasons), {'near_plane', 'depth_geometry', 'mask_outside_projected_geometry'})

    def test_positive_matches_take_precedence_over_ignore_for_samplers(self):
        boxes = [[0, 0, 10, 10], [2, 2, 4, 4], [10, 0, 20, 10], [0, 0, 100, 100]]
        labels = ignore_background_proposals(boxes, [1, 0, 0, -1], [[0, 0, 10, 10]])
        self.assertEqual(labels.tolist(), [1, -1, 0, -1])
        with self.assertRaises(ValueError):
            ignore_background_proposals([[1, 0, 0, 1]], [0], [])

    def test_projection_and_depth_tolerance_are_consistent_in_revised_policy(self):
        c = copy.deepcopy(self.candidate)
        c['visible_bbox_xyxy'] = [0, 0, 12, 8]
        c['geometry_tolerance_m'] = .15
        c['projection_support_envelope'] = {'bbox_xyxy': [0, 0, 12, 8]}
        self.assertEqual(classify_candidate(c, self.policy)[0], 'ignore')
        policy = {**self.policy, 'geometry_projection': 'projection_support_envelope', 'geometry_tolerance_m': .15}
        self.assertEqual(classify_candidate(c, policy)[0], 'positive')
        policy['geometry_tolerance_m'] = .2
        with self.assertRaises(ValueError):
            classify_candidate(c, policy)
