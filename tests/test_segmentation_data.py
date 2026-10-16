import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from senior_ai.frames import RgbFrame
from senior_ai.segmentation_data import load_inference_rgb, SegmentationSamples


class SegmentationInputTests(unittest.TestCase):
    def test_inference_decodes_actual_rgb_without_labels(self):
        temporary_root = Path(__file__).resolve().parents[1] / 'artifacts/runs/test-temp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            (root / 'inputs/frames').mkdir(parents=True)
            (root / 'inputs/rgb').mkdir()
            RgbFrame(7, 0.35, 2, 1, bytes([0, 0, 255, 255, 0, 255, 0, 255])).save_to_disk(root / 'inputs/rgb/a.png')
            feature = {'schema_version': 1, 'source': 'carla_rgb_only', 'scene_id': 'a', 'frame': 7, 'timestamp_s': 0.35,
                       'calibration': 'calibration/a.json', 'cameras': {str(i): {'image': 'inputs/rgb/a.png', 'frame': 7, 'timestamp_s': 0.35} for i in range(8)}}
            (root / 'inputs/frames/a.json').write_text(json.dumps(feature))
            self.assertEqual(load_inference_rgb(root, 'inputs/frames/a.json', '0').tolist(), [[[255, 0, 0], [0, 255, 0]]])
            self.assertFalse((root / 'labels').exists())

    def test_partial_acquisition_cannot_enter_training(self):
        temporary_root = Path(__file__).resolve().parents[1] / 'artifacts/runs/test-temp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            (root / 'report.json').write_text(json.dumps({'task': 'dataset-probe', 'status': 'passed'}))
            (root / 'audit.json').write_text(json.dumps({'status': 'passed'}))
            with self.assertRaises(ValueError):
                SegmentationSamples(root, 'train')


if __name__ == '__main__':
    unittest.main()
