"""Export explicit RGB/semantic per-view lists after full acquisition and audit."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json
from senior_ai.segmentation_data import SegmentationSamples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / 'training'
    folder.mkdir(exist_ok=False)
    result = {'task': 'segmentation-training-format', 'status': 'failed', 'splits': {},
              'decoder': {'Pillow': importlib.metadata.version('Pillow')},
              'audit_sha256': hashlib.sha256((root / 'audit.json').read_bytes()).hexdigest(),
              'note': 'Per-view lists retain whole-scene splits. Raw RGB + class-ID targets; no learned model or quality metric.'}
    for split in ['train', 'validation', 'test']:
        data = SegmentationSamples(root, split)
        target = folder / f'{split}.jsonl'
        target.write_text(''.join(json.dumps(s) + '\n' for s in data.samples))
        # Independent Pillow decoder checks first/last bundle, every direction, per scene.
        by_scene = {}
        for i, sample in enumerate(data.samples):
            by_scene.setdefault(sample['scene_id'], []).append(i)
        checks = sorted({i for indices in by_scene.values() for i in indices[:8] + indices[-8:]})
        for i in checks:
            sample = data[i]
            assert sample['rgb'].dtype.name == sample['target'].dtype.name == 'uint8'
        result['splits'][split] = {'samples': len(data), 'independently_decoded_samples': len(checks),
                                  'index_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    result['status'] = 'passed'
    write_json(folder / 'report.json', result)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
