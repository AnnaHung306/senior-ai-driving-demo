"""Framework-neutral RGB inference and separately authorized segmentation labels."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import check_group_splits, read_sensor_frame, safe_path


def _rgb(path):
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGB':
            raise ValueError('Expected raw three-channel RGB PNG')
        return np.array(image, dtype=np.uint8)


def load_inference_rgb(root, input_manifest, camera):
    """No label reads, CARLA connection, depth or simulator object access."""
    frame = read_sensor_frame(root, input_manifest)
    return _rgb(safe_path(root, frame['cameras'][camera]['image']))


class SegmentationSamples:
    """Lazy per-view training/evaluation samples from an audited grouped split.

    Returns uint8 RGB HWC and uint8 class-ID HW; normalization, mapping and tensor
    conversion belong to a versioned model experiment. This class is not inference.
    """
    def __init__(self, root, split):
        self.root = Path(root).resolve()
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Unknown split')
        acquisition = json.loads((self.root / 'report.json').read_text())
        audit = json.loads((self.root / 'audit.json').read_text())
        if acquisition['task'] != 'dataset-collect' or acquisition['status'] != 'passed' or audit['status'] != 'passed':
            raise ValueError('Training input requires complete acquisition and successful audit')
        plan = json.loads((self.root / 'plan.json').read_text())
        check_group_splits(plan)
        scenes = {s['id']: s for s in plan['scenes']}
        self.rig = json.loads((self.root / 'rig.json').read_text())
        self.samples = []
        for line in (self.root / f'indexes/{split}.jsonl').read_text().splitlines():
            index = json.loads(line)
            scene = scenes[index['scene_id']]
            if split != scene['split'] or split != index['split'] or scene['group_id'] != index['group_id']:
                raise ValueError('Training index crosses scene/split boundary')
            frame = read_sensor_frame(self.root, index['input'])
            label_path = safe_path(self.root, index['label'])
            if not label_path.is_relative_to(self.root / 'labels/frames'):
                raise ValueError('Training label outside label boundary')
            label = json.loads(label_path.read_text())
            if label['source'] != 'simulator_ground_truth' or label['frame'] != frame['frame'] or label['timestamp_s'] != frame['timestamp_s']:
                raise ValueError('Training GT/feature frame mismatch')
            if frame['scene_id'] != index['scene_id'] or frame['frame'] != index['frame']:
                raise ValueError('Training frame/index mismatch')
            if set(label['semantic_segmentation']) != set(frame['cameras']):
                raise ValueError('Missing per-view semantic target')
            for name, camera in frame['cameras'].items():
                target = label['semantic_segmentation'][name]
                if target['frame'] != frame['frame'] or target['timestamp_s'] != camera['timestamp_s']:
                    raise ValueError('Per-camera semantic target mismatch')
                path = safe_path(self.root, target['image'])
                if not path.is_relative_to(self.root / 'labels/semantic_segmentation'):
                    raise ValueError('Semantic target outside label boundary')
                self.samples.append({'scene_id': index['scene_id'], 'group_id': index['group_id'], 'split': split,
                    'frame': frame['frame'], 'timestamp_s': frame['timestamp_s'], 'camera': name,
                    'rgb': camera['image'], 'semantic': target['image'], 'input_manifest': index['input']})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        rgb = load_inference_rgb(self.root, sample['input_manifest'], sample['camera'])
        semantic = _rgb(safe_path(self.root, sample['semantic']))
        shape = (self.rig['height'], self.rig['width'], 3)
        if rgb.shape != shape or semantic.shape != shape or semantic[..., 1:].any() or semantic[..., 0].max() > 28:
            raise ValueError('Invalid training image shape or raw semantic encoding')
        return {'rgb': rgb, 'target': semantic[..., 0].copy(), 'metadata': dict(sample),
                'source': 'simulator_gt_for_training_or_evaluation_only'}
