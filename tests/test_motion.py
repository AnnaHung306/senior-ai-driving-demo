import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from senior_ai.motion import audit_episode, require_class_coverage


class MotionAuditTests(unittest.TestCase):
    def setUp(self):
        self.criteria = {'bundles_per_scene': 3, 'capture_stride_ticks': 4, 'fixed_delta_s': 0.05,
                         'max_timestamp_error_s': 0.0001, 'min_ego_displacement_m': 1,
                         'min_label_duration_s': 0.39, 'min_participant_displacement_m': 0.1}
        self.labels = [{'episode_tick': i * 4, 'frame': 100 + 4 * i, 'timestamp_s': 1 + i * 0.2,
                        'ego_world_pose': {'position_m': [i, 0, 0]},
                        'objects_3d': [{'actor_id': 7, 'world_pose': {'position_m': [0, i * 0.2, 0]}}]} for i in range(3)]

    def test_measured_motion_passes_but_stationary_ego_fails(self):
        self.assertEqual(audit_episode(self.labels, self.criteria)['ego_displacement_m'], 2)
        for label in self.labels:
            label['ego_world_pose']['position_m'] = [0, 0, 0]
        with self.assertRaises(ValueError):
            audit_episode(self.labels, self.criteria)

    def test_bad_cadence_identity_or_static_participant_rejected(self):
        for kind in ['time', 'identity', 'static', 'frame']:
            labels = copy.deepcopy(self.labels)
            if kind == 'time':
                labels[1]['timestamp_s'] = 1.25
            elif kind == 'identity':
                labels[1]['objects_3d'][0]['actor_id'] = 8
            elif kind == 'static':
                for label in labels:
                    label['objects_3d'][0]['world_pose']['position_m'] = [0, 0, 0]
            else:
                labels[1]['frame'] = 105
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit_episode(labels, self.criteria)

    def test_spawned_class_does_not_replace_pixel_coverage(self):
        criteria = {'required_semantic_ids_per_split': [12, 19],
                    'min_semantic_pixels_per_required_class_per_split': 1000,
                    'min_positive_images_per_required_class_per_split': 10}
        pixels = {s: {12: 2000, 19: 2000} for s in ['train', 'validation', 'test']}
        positives = {s: {12: 20, 19: 20} for s in pixels}
        require_class_coverage(pixels, positives, criteria)
        pixels['test'][19] = 999
        with self.assertRaises(ValueError):
            require_class_coverage(pixels, positives, criteria)


if __name__ == '__main__':
    unittest.main()
