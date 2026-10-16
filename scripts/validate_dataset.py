"""Independent on-disk audit of grouped splits, every PNG and GT isolation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json
from senior_ai.dataset import check_group_splits, read_sensor_frame, safe_path, validate_calibration


def read_png(path, width, height):
    raw = path.read_bytes()
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'PNG signature: {path}')
    cursor, compressed, header = 8, bytearray(), None
    while cursor < len(raw):
        size = struct.unpack_from('>I', raw, cursor)[0]
        kind = raw[cursor + 4:cursor + 8]
        payload = raw[cursor + 8:cursor + 8 + size]
        crc = struct.unpack_from('>I', raw, cursor + 8 + size)[0]
        if zlib.crc32(kind + payload) != crc:
            raise ValueError(f'PNG CRC: {path}')
        if kind == b'IHDR':
            header = struct.unpack('>IIBBBBB', payload)
        elif kind == b'IDAT':
            compressed.extend(payload)
        cursor += size + 12
        if kind == b'IEND':
            break
    if cursor != len(raw) or header != (width, height, 8, 2, 0, 0, 0):
        raise ValueError(f'PNG dimensions/encoding: {path}')
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(height, width * 3 + 1)
    if rows[:, 0].any():
        raise ValueError('Unexpected PNG row filter; collector schema requires lossless filter 0')
    return rows[:, 1:].reshape(height, width, 3), hashlib.sha256(raw).hexdigest(), len(raw)


def validate(root, criteria_reference, rig_reference):
    started = time.monotonic()
    plan = json.loads((root / 'plan.json').read_text())
    criteria = json.loads((root / 'criteria.json').read_text())
    rig = json.loads((root / 'rig.json').read_text())
    acquisition = json.loads((root / 'report.json').read_text())
    if acquisition['task'] != 'dataset-collect' or acquisition['status'] != 'passed':
        raise ValueError('Only complete acquisition may pass a full dataset audit')
    # Export JSON formatting can differ; verify original frozen bytes AND values.
    for reference, saved, expected in [(criteria_reference, criteria, plan['criteria_sha256']),
                                       (rig_reference, rig, plan['rig_sha256'])]:
        if hashlib.sha256(reference.read_bytes()).hexdigest() != expected or json.loads(reference.read_text()) != saved:
            raise ValueError('Frozen source config hash or exported values differ')
    groups = check_group_splits(plan)
    scenes = {s['id']: s for s in plan['scenes']}
    if len(scenes) != criteria['scene_count']:
        raise ValueError('Wrong scene count')
    positions = [np.array(s['start']['position_m']) for s in scenes.values()]
    separation = min(float(np.linalg.norm(a - b)) for i, a in enumerate(positions) for b in positions[i + 1:])
    if separation < criteria['min_scene_origin_separation_m']:
        raise ValueError('Insufficient scene origin separation')
    counts, pixels, files, seen, metadata_files, objects = Counter(), Counter(), [], set(), set(), Counter()
    depth_min, depth_max, max_error = float('inf'), 0.0, 0.0
    names = {c['name'] for c in rig['cameras']}
    for split in ['train', 'validation', 'test']:
        for line in (root / f'indexes/{split}.jsonl').read_text().splitlines():
            if time.monotonic() - started > 600:
                raise TimeoutError('Dataset audit exceeded 600 second wall-clock limit')
            index = json.loads(line)
            scene = scenes[index['scene_id']]
            if scene['split'] != split or index['split'] != split or index['group_id'] != scene['group_id']:
                raise ValueError('Index assigned to wrong scene/split/group')
            feature = read_sensor_frame(root, index['input'])
            if index['input'] in metadata_files:
                raise ValueError('Repeated input frame')
            metadata_files.add(index['input'])
            if feature['scene_id'] != scene['id'] or feature['frame'] != index['frame']:
                raise ValueError('Input index mismatch')
            label_path = safe_path(root, index['label'])
            if not label_path.is_relative_to((root / 'labels/frames').resolve()):
                raise ValueError('GT labels outside label boundary')
            labels = json.loads(label_path.read_text())
            if labels['source'] != 'simulator_ground_truth' or labels['frame'] != feature['frame'] or labels['timestamp_s'] != feature['timestamp_s']:
                raise ValueError('GT/feature alignment failure')
            if len(labels['objects_3d']) != scene['requested_npcs']:
                raise ValueError('Object labels incomplete')
            for box in labels['objects_3d']:
                values = np.array(box['world_vertices_m'])
                if values.shape != (8, 3) or not np.isfinite(values).all() or min(box['extent_m']) <= 0:
                    raise ValueError('Malformed 3D box')
                objects[box['blueprint']] += 1
            for modality in criteria['modalities']:
                cameras = feature['cameras'] if modality == 'rgb' else labels[modality]
                if set(cameras) != names:
                    raise ValueError('Missing/extra camera direction')
                for entry in cameras.values():
                    error = abs(entry['timestamp_s'] - feature['timestamp_s'])
                    if entry['frame'] != feature['frame'] or not np.isfinite(error) or error > criteria['max_timestamp_error_s']:
                        raise ValueError('Per-image frame/timestamp mismatch')
                    max_error = max(error, max_error)
                    relative = entry['image']
                    prefix = 'inputs/rgb' if modality == 'rgb' else f'labels/{modality}'
                    path = safe_path(root, relative)
                    if not path.is_relative_to((root / prefix).resolve()) or relative in seen:
                        raise ValueError('Cross-boundary or reused image file')
                    seen.add(relative)
                    rgb, digest, size = read_png(path, rig['width'], rig['height'])
                    files.append({'path': relative, 'sha256': digest, 'bytes': size})
                    if modality == 'semantic_segmentation':
                        ids, totals = np.unique(rgb[..., 0], return_counts=True)
                        if ids.max() > 28 or rgb[..., 1:].any():
                            raise ValueError('Semantic encoding corrupt')
                        pixels.update({int(k): int(v) for k, v in zip(ids, totals)})
                    elif modality == 'depth':
                        data = rgb.astype(np.uint32)
                        meters = (data[..., 0] + data[..., 1] * 256 + data[..., 2] * 65536) * (1000.0 / 16777215)
                        depth_min, depth_max = min(depth_min, float(meters.min())), max(depth_max, float(meters.max()))
            calibration = json.loads(safe_path(root, feature['calibration']).read_text())
            validate_calibration(calibration, rig)
            roads = json.loads(safe_path(root, labels['roads']).read_text())
            if roads['source'] != 'simulator_ground_truth' or not roads['waypoints']:
                raise ValueError('Missing road labels')
            counts[scene['id']] += 1
            if counts[scene['id']] > criteria['bundles_per_scene']:
                raise ValueError('Too many frames in scene')
            if counts[scene['id']] == criteria['bundles_per_scene']:
                print(json.dumps({'audited_scene': scene['id'], 'checked_pngs': len(files)}), flush=True)
    if set(counts) != set(scenes) or any(n != criteria['bundles_per_scene'] for n in counts.values()):
        raise ValueError('Incomplete scene bundles')
    if len(files) != criteria['scene_count'] * criteria['bundles_per_scene'] * 24:
        raise ValueError('Image count mismatch')
    if not 0 <= depth_min < depth_max <= 1000:
        raise ValueError('Invalid depth ranges')
    inventory = root / 'image_inventory.jsonl'
    inventory.write_text(''.join(json.dumps(item) + '\n' for item in files))
    return {'task': 'dataset-on-disk-audit', 'status': 'passed', 'bundles_by_scene': dict(counts),
            'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'auditor_git_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True, timeout=5).stdout.strip(),
            'wall_s': time.monotonic() - started,
            'split_bundles': {s: sum(counts[c['id']] for c in scenes.values() if c['split'] == s) for s in groups},
            'scene_groups': {k: sorted(v) for k, v in groups.items()}, 'scene_group_overlap': 0,
            'min_scene_origin_separation_m': separation, 'checked_pngs': len(files), 'png_crc_errors': 0,
            'image_bytes': sum(f['bytes'] for f in files), 'max_timestamp_error_s': max_error,
            'semantic_pixel_counts': dict(sorted(pixels.items())), 'depth_min_m': depth_min, 'depth_max_m': depth_max,
            'box_annotations_by_blueprint': dict(objects), 'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest(),
            'scope': criteria['scope'], 'ai_model': None,
            'limitations': ['Stationary ego and 1.5 simulation seconds per scene', 'Only six groups in one shared town',
                            'NPC vehicles are one blueprint; no pedestrians/bicycles/motorcycles collected',
                            'Origin separation/group disjointness do not guarantee zero visual content overlap',
                            '3D boxes are GT and include occluded/out-of-view actors', 'No trained perception or model quality metric']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--criteria', type=Path, default=ROOT / 'configs/dataset_pilot_criteria_v1.json')
    parser.add_argument('--rig', type=Path, default=ROOT / 'configs/camera_rig_v1.json')
    args = parser.parse_args()
    try:
        report = validate(args.root.resolve(), args.criteria, args.rig)
    except Exception as exc:
        report = {'task': 'dataset-on-disk-audit', 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    write_json(args.root / 'audit.json', report)
    print(json.dumps(report))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
