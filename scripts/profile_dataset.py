"""Exploratory RGB quality samples; no model evaluation or post-hoc pass gate."""
import argparse
import json
from pathlib import Path

import numpy as np

from validate_dataset import read_png
from collect_dataset import ROOT
from senior_ai.core import write_json
from senior_ai.dataset import read_sensor_frame, safe_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rig = json.loads((root / 'rig.json').read_text())
    scenes = {s['id']: s for s in json.loads((root / 'plan.json').read_text())['scenes']}
    seen, samples = set(), []
    for split in ['train', 'validation', 'test']:
        for line in (root / f'indexes/{split}.jsonl').read_text().splitlines():
            index = json.loads(line)
            if index['scene_id'] in seen:
                continue
            seen.add(index['scene_id'])
            feature = read_sensor_frame(root, index['input'])
            for name, entry in feature['cameras'].items():
                pixels, _, _ = read_png(safe_path(root, entry['image']), rig['width'], rig['height'])
                luminance = pixels.astype(np.float32).mean(axis=2)
                samples.append({'scene_id': index['scene_id'], 'weather': scenes[index['scene_id']]['weather'],
                                'camera': name, 'image': entry['image'], 'mean_rgb': float(pixels.mean()),
                                'std_rgb': float(pixels.std()), 'near_black_fraction': float((luminance < 3).mean()),
                                'near_white_fraction': float((luminance > 252).mean())})
    report = {'task': 'exploratory-rgb-quality-sample', 'sampling': 'First bundle of each scene, all eight cameras',
              'sampled_images': len(samples), 'samples': samples,
              'limitations': 'Descriptive anomaly inspection only, no quality acceptance thresholds and no model metrics. Short stationary sequences are strongly correlated.'}
    write_json(root / 'rgb_profile.json', report)
    print(json.dumps({'sampled_images': len(samples), 'mean_rgb_range': [min(s['mean_rgb'] for s in samples), max(s['mean_rgb'] for s in samples)],
                      'max_near_black_fraction': max(s['near_black_fraction'] for s in samples)}))


if __name__ == '__main__':
    main()
