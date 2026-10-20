import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.world_schema import validate_world_frame


class WorldSchemaTests(unittest.TestCase):
    def frame(self):
        return {'schema_version': 1, 'coordinate_frame': 'current_ego', 'axes': 'x_forward_y_right_z_up',
                'units': {'position': 'meters', 'size': 'full_LWH_meters', 'angle': 'radians'},
                'GT_used_in_inference': False, 'control_issued': False, 'frame': 1, 'timestamp_s': 1., 'scene_id': 'a',
                'objects': [{'center_ego_m': [5., 0., 1.], 'size_lwh_m': [4., 2., 2.], 'yaw_rad': 0., 'class_id': 1,
                             'score_uncalibrated': .9, 'track_id': 'a:T000001', 'measurement_current': True,
                             'sources': [{'class_id': 1, 'camera': 'front'}]}],
                'occupancy': {'file': 'occupancy.npz', 'sha256': '0'*64}, 'tracking': {'held_tracks_used_for_occupancy': False},
                'occupancy_grid': {'x_min': -1, 'x_max': 1, 'y_min': -1, 'y_max': 1, 'z_min': -.5, 'z_max': 1.5, 'resolution_m': .5}}

    def test_metric_schema_and_unknown_occupancy_valid(self):
        self.assertTrue(validate_world_frame(self.frame(), np.full((4, 4, 4), 255, np.uint8)))

    def test_wrong_axes_units_and_duplicate_ID_rejected(self):
        f = self.frame();f['axes'] = 'x_right_y_forward'
        with self.assertRaises(ValueError):validate_world_frame(f)
        f = self.frame();f['units']['angle'] = 'degrees'
        with self.assertRaises(ValueError):validate_world_frame(f)
        f = self.frame();f['objects'].append(copy.deepcopy(f['objects'][0]))
        with self.assertRaises(ValueError):validate_world_frame(f)

    def test_GT_and_held_measurement_rejected(self):
        f = self.frame();f['objects'][0]['sources'][0]['actor_id'] = 12
        with self.assertRaises(ValueError):validate_world_frame(f)
        f = self.frame();f['objects'][0]['measurement_current'] = False
        with self.assertRaises(ValueError):validate_world_frame(f)

    def test_unsafe_artifact_path_rejected(self):
        for path in ['../outside.npz', 'C:/outside.npz', '/outside.npz', '..\\outside.npz']:
            f = self.frame();f['occupancy']['file'] = path
            with self.assertRaises(ValueError):validate_world_frame(f)

    def test_free_ground_and_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):validate_world_frame(self.frame(), np.zeros((4, 4, 4), np.uint8))
        with self.assertRaises(ValueError):validate_world_frame(self.frame(), np.full((4, 4, 3), 255, np.uint8))


if __name__ == '__main__':unittest.main()
