"""Explicit RGB-only inference boundary and independently located simulator labels."""
import json
from pathlib import Path

import numpy as np


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Dataset path escapes root')
    return path


def read_sensor_frame(root, relative):
    """Reads only the feature manifest; never opens a label file or CARLA connection."""
    path = safe_path(root, relative)
    if not path.is_relative_to((Path(root) / 'inputs').resolve()):
        raise ValueError('Inference manifest must be under inputs/')
    frame = json.loads(path.read_text())
    if set(frame) != {'schema_version', 'source', 'scene_id', 'frame', 'timestamp_s', 'cameras', 'calibration'}:
        raise ValueError('Unexpected inference fields; GT is forbidden')
    if frame['source'] != 'carla_rgb_only' or len(frame['cameras']) != 8:
        raise ValueError('Eight RGB inputs required')
    for camera in frame['cameras'].values():
        if set(camera) != {'image', 'frame', 'timestamp_s'} or camera['frame'] != frame['frame']:
            raise ValueError('Invalid camera input or frame mismatch')
        if abs(camera['timestamp_s'] - frame['timestamp_s']) > 0.0001:
            raise ValueError('Input timestamp mismatch')
        image = safe_path(root, camera['image'])
        if not image.is_relative_to((Path(root) / 'inputs/rgb').resolve()):
            raise ValueError('Inference image must be RGB under inputs/rgb/')
    calibration = safe_path(root, frame['calibration'])
    if not calibration.is_relative_to((Path(root) / 'calibration').resolve()):
        raise ValueError('Calibration must be separate from labels')
    return frame


def check_group_splits(plan):
    ids = set()
    groups = {name: set() for name in ('train', 'validation', 'test')}
    for scene in plan['scenes']:
        if scene['id'] in ids or scene['split'] not in groups:
            raise ValueError('Duplicate scene or invalid split')
        ids.add(scene['id'])
        groups[scene['split']].add(scene['group_id'])
    if not all(groups.values()):
        raise ValueError('Every split requires at least one scene group')
    if any(groups[a] & groups[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Scene group leakage across dataset splits')
    return groups


def decode_depth_bgra(raw, width, height):
    """CARLA 24-bit depth, meters; preserve packed channels when storing PNG."""
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4).astype(np.uint32)
    return (pixels[..., 2] + pixels[..., 1] * 256 + pixels[..., 0] * 65536) * (1000.0 / 16777215)


def semantic_ids_bgra(raw, width, height):
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)[..., 2].copy()
