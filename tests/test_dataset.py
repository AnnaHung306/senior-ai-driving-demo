import copy
import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from senior_ai.dataset import check_group_splits, decode_depth_bgra, read_sensor_frame, semantic_ids_bgra


class DatasetBoundaryTests(unittest.TestCase):
    def test_group_leak_rejected(self):
        plan = {'scenes': [{'id': str(i), 'group_id': str(i), 'split': split}
                           for i, split in enumerate(['train', 'validation', 'test'])]}
        check_group_splits(plan)
        plan['scenes'][2]['group_id'] = '0'
        with self.assertRaises(ValueError):
            check_group_splits(plan)

    def test_rgb_manifest_does_not_read_labels_and_rejects_gt(self):
        temporary_root = Path(__file__).resolve().parents[1] / 'artifacts/runs/test-temp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            path = root / 'inputs/frames/1.json'
            path.parent.mkdir(parents=True)
            feature = {'schema_version': 1, 'source': 'carla_rgb_only', 'scene_id': 'a', 'frame': 1,
                       'timestamp_s': 0.05, 'calibration': 'calibration/a.json',
                       'cameras': {str(i): {'image': f'inputs/rgb/{i}.png', 'frame': 1, 'timestamp_s': 0.05} for i in range(8)}}
            path.write_text(json.dumps(feature))
            # No labels folder exists. Loading metadata must not import GT or CARLA.
            self.assertEqual(read_sensor_frame(root, 'inputs/frames/1.json'), feature)
            bad = copy.deepcopy(feature)
            bad['objects_3d'] = []
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                read_sensor_frame(root, 'inputs/frames/1.json')
            bad = copy.deepcopy(feature)
            bad['cameras']['0']['image'] = 'labels/semantic_segmentation/0.png'
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                read_sensor_frame(root, 'inputs/frames/1.json')
            bad['cameras']['0']['image'] = '../outside.png'
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                read_sensor_frame(root, 'inputs/frames/1.json')

    def test_depth_channel_order_and_range(self):
        raw = bytes([0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 1, 255, 1, 0, 0, 255])
        values = decode_depth_bgra(raw, 4, 1)[0]
        self.assertEqual(values[0], 0)
        self.assertAlmostEqual(values[1], 1000)
        self.assertAlmostEqual(values[3] / values[2], 65536)

    def test_semantic_ids_are_red_not_palette(self):
        self.assertEqual(semantic_ids_bgra(bytes([0, 0, 14, 255, 0, 0, 1, 255]), 2, 1).tolist(), [[14, 1]])


if __name__ == '__main__':
    unittest.main()
