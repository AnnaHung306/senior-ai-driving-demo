import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_error_catalog import catalog


class ErrorCatalogTests(unittest.TestCase):
    def test_false_positives_misses_and_semantic_support_bins(self):
        temp = ROOT / 'artifacts/runs/test-temp'
        temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp) as folder:
            run = Path(folder)
            (run / 'config.json').write_text(json.dumps({'class_names': ['other', 'road', 'a', 'b', 'c', 'd', 'e', 'f']}))
            frames = []
            for number, support, predicted, tp in [(7, 10, 8, 4), (8, 2, 9, 1)]:
                classes = [{'support_pixels': 0, 'predicted_pixels': 0, 'recall': None, 'precision': None, 'iou': None} for _ in range(8)]
                classes[1] = {'support_pixels': support, 'predicted_pixels': predicted, 'recall': tp / support,
                              'precision': tp / predicted, 'iou': tp / (support + predicted - tp)}
                frames.append({'scene_id': 's', 'group_id': 'g', 'frame': number, 'timestamp_s': number * .05,
                    'camera': 'front', 'weather': 'rain', 'input_manifest': 'a', 'rgb': 'b', 'semantic': 'c', 'classes': classes})
            for split in ['validation', 'test']:
                (run / f'{split}_errors').mkdir()
                (run / f'{split}_errors/frames.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
            self.assertEqual(catalog(run), 8)
            result = json.loads((run / 'error_catalog.json').read_text())
            fp = [c for c in result['cases'] if c['split'] == 'test' and c['ranked_by'] == 'false_positive_pixels']
            fn = [c for c in result['cases'] if c['split'] == 'test' and c['ranked_by'] == 'false_negative_pixels']
            self.assertEqual(fp[0]['frame'], 8)
            self.assertEqual(fn[0]['frame'], 7)
            stats = result['semantic_support_strata']['test']['road']['1..100']
            self.assertAlmostEqual(stats['recall'], 5 / 12)
            self.assertAlmostEqual(stats['iou'], 5 / 24)
            self.assertIsNone(result['semantic_support_strata']['test']['a']['absent']['iou'])
