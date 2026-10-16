import math
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from senior_ai.core import assess, intrinsics, receive_frame, write_json


class FrameSynchronizationTests(unittest.TestCase):
    def test_stale_frames_are_not_used(self):
        q = queue.Queue()
        for frame in [7, 8, 9]:
            q.put(SimpleNamespace(frame=frame))
        self.assertEqual(receive_frame(q, 9, 0.1).frame, 9)

    def test_future_frame_fails_instead_of_silent_misalignment(self):
        q = queue.Queue()
        q.put(SimpleNamespace(frame=10))
        with self.assertRaises(ValueError):
            receive_frame(q, 9, 0.1)

    def test_missing_camera_times_out(self):
        with self.assertRaises(TimeoutError):
            receive_frame(queue.Queue(), 1, 0.001)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        import json
        self.criteria = json.loads((Path(__file__).resolve().parents[1] /
                                    'configs/phase0_acceptance.json').read_text())
        self.metrics = dict(distance_m=2, max_speed_mps=2, final_speed_mps=0,
                            saved_images=14, max_timestamp_error_s=0,
                            observed_frames=140, collision_count=0, cleanup_ok=True)

    def test_nominal_acceptance_and_failures(self):
        self.assertTrue(all(assess(self.metrics, self.criteria).values()))
        for key, bad in [('distance_m', 0), ('max_speed_mps', 0),
                         ('final_speed_mps', 1), ('saved_images', 0),
                         ('max_timestamp_error_s', 1), ('observed_frames', 139),
                         ('collision_count', 1), ('cleanup_ok', False), ('distance_m', math.nan)]:
            with self.subTest(metric=key):
                self.assertFalse(all(assess({**self.metrics, key: bad}, self.criteria).values()))

    def test_known_projection(self):
        matrix = intrinsics(640, 360, 90)
        self.assertAlmostEqual(matrix[0][0], 320)
        self.assertEqual(matrix[0][2], 320)
        self.assertEqual(matrix[1][2], 180)
        with self.assertRaises(ValueError):
            intrinsics(640, 360, 180)

    def test_invalid_numbers_cannot_enter_report(self):
        path = Path(__file__).resolve().parents[1] / 'artifacts/reports/rejected-nan.json'
        with self.assertRaises(ValueError):
            write_json(path, {'accuracy': math.nan})
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
