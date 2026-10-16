"""Audit a short moving 32-sensor probe, explicitly NOT a complete dataset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json
from senior_ai.dataset import read_sensor_frame, safe_path, validate_calibration
from senior_ai.instance_labels import decode_instance_rgb, validate_actor_ids, visible_candidate
from collect_dataset import sha, transform
from validate_dataset import read_png


def audit(root):
    acquisition = json.loads((root / 'report.json').read_text())
    if acquisition['task'] != 'dataset-probe' or acquisition['status'] != 'passed' or acquisition['training_approved'] is not False:
        raise ValueError('Requires passed non-training moving probe')
    rig, criteria, plan = [json.loads((root / name).read_text()) for name in ['rig.json', 'criteria.json', 'plan.json']]
    if criteria['modalities'] != ['rgb', 'semantic_segmentation', 'depth', 'instance_segmentation']:
        raise ValueError('Four raw modalities required')
    entries = [json.loads(line) for split in ['train', 'validation', 'test']
               for line in (root / f'indexes/{split}.jsonl').read_text().splitlines()]
    if len(entries) != acquisition['total_bundles'] or len({e['scene_id'] for e in entries}) != 1:
        raise ValueError('Expected one bounded scene probe')
    scene = next(s for s in plan['scenes'] if s['id'] == entries[0]['scene_id'])
    cal = json.loads((root / f'calibration/{scene["id"]}.json').read_text()); validate_calibration(cal, rig)
    cameras = {c['name']: c for c in cal['cameras']}
    if any(set(c.get('sensor_attributes', {})) != set(criteria['modalities']) for c in cameras.values()):
        raise ValueError('Actual sensor attribute receipt missing')
    inventory, frames, candidate_frames, known_ids = [], [], [], None
    max_pose, max_translation, max_rotation, disagreement, pixels, max_stamp_error = 0., 0., 0., 0, 0, 0.
    seen = set()
    qa = root / 'qa-motion'; qa.mkdir(exist_ok=False)
    for entry in entries:
        feature = read_sensor_frame(root, entry['input'])
        label = json.loads(safe_path(root, entry['label']).read_text())
        if label['source'] != 'simulator_ground_truth' or label['frame'] != feature['frame'] or label['timestamp_s'] != feature['timestamp_s']:
            raise ValueError('GT/feature mismatch')
        if entry['group_id'] != scene['group_id'] or entry['split'] != scene['split']:
            raise ValueError('Probe group mismatch')
        expected_actors = Counter((d['blueprint'], d['kind']) for d in scene['participants'])
        if Counter((a['blueprint'], a['kind']) for a in label['objects_3d']) != expected_actors:
            raise ValueError('Participant inventory mismatch')
        validate_actor_ids(label['objects_3d'])
        current_ids = {a['actor_id'] for a in label['objects_3d']}
        if known_ids is not None and current_ids != known_ids:
            raise ValueError('Actor identity changed')
        known_ids = current_ids
        world_ego = np.array(transform(label['ego_world_pose']).get_matrix())
        objects = [{**a, 'T_world_from_box': (np.array(transform(a['world_pose']).get_matrix()) @
                    np.array(transform(a['bbox_local_pose']).get_matrix())).tolist()} for a in label['objects_3d']]
        views = {}
        for name, camera in cameras.items():
            images = {}
            poses = label['capture_world_from_camera'][name]
            if set(poses) != set(criteria['modalities']):
                raise ValueError('Capture-time modality poses incomplete')
            actual = np.array(poses['rgb'])
            if actual.shape != (4, 4) or not np.isfinite(actual).all():
                raise ValueError('Invalid capture transform')
            max_pose = max(max_pose, max(float(np.abs(actual - poses[m]).max()) for m in poses))
            expected = world_ego @ np.array(camera['T_ego_from_camera_local'])
            max_translation = max(max_translation, float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])))
            max_rotation = max(max_rotation, float(np.abs(actual[:3, :3] - expected[:3, :3]).max()))
            for modality in criteria['modalities']:
                source = feature['cameras'][name] if modality == 'rgb' else label[modality][name]
                error = abs(source['timestamp_s'] - feature['timestamp_s'])
                if source['frame'] != feature['frame'] or error > criteria['max_timestamp_error_s'] or not np.isfinite(error):
                    raise ValueError('Raw image alignment mismatch')
                max_stamp_error = max(max_stamp_error, error)
                prefix = 'inputs/rgb' if modality == 'rgb' else 'labels/' + modality
                path = safe_path(root, source['image'])
                if not path.is_relative_to((root / prefix).resolve()) or source['image'] in seen:
                    raise ValueError('GT boundary or duplicated image')
                seen.add(source['image'])
                images[modality], digest, size = read_png(path, rig['width'], rig['height'])
                inventory.append({'path': source['image'], 'sha256': digest, 'bytes': size})
            classes, ids = decode_instance_rgb(images['instance_segmentation'])
            semantic = images['semantic_segmentation'][..., 0]
            disagreement += int((classes != semantic).sum()); pixels += semantic.size
            packed = images['depth'].astype(np.uint32)
            depth = (packed[..., 0] + 256 * packed[..., 1] + 65536 * packed[..., 2]) * (1000 / 16777215)
            candidates = [visible_candidate(a, classes, ids, semantic, depth, {'K': camera['K'], 'T_world_from_camera': actual}) for a in objects]
            views[name] = candidates
            if name == 'front' and entry in [entries[0], entries[-1]]:
                panel = Image.fromarray(images['rgb']); draw = ImageDraw.Draw(panel)
                for c in candidates:
                    if c['visible_bbox_xyxy']:
                        b = c['visible_bbox_xyxy']; draw.rectangle((b[0], b[1], b[2]-1, b[3]-1), outline='yellow', width=2)
                draw.text((4, 4), f'GT ONLY moving probe frame {feature["frame"]}', fill='red')
                panel.save(qa / f'{feature["frame"]}.png')
        frames.append(label)
        candidate_frames.append({'frame': feature['frame'], 'views': views})
    expected_dt = criteria['fixed_delta_s'] * criteria['capture_stride_ticks']
    diffs = np.diff([f['timestamp_s'] for f in frames])
    ego_displacement = float(np.linalg.norm(np.array(frames[-1]['ego_world_pose']['position_m']) - frames[0]['ego_world_pose']['position_m']))
    checks = {'five_bundles': len(frames) == 5, 'images_160': len(inventory) == 160,
        'frame_cadence': np.diff([f['frame'] for f in frames]).tolist() == [criteria['capture_stride_ticks']] * (len(frames)-1),
        'timestamp_cadence': bool(np.allclose(diffs, expected_dt, atol=1e-4, rtol=0)),
        'all_modality_poses_match': max_pose < 1e-5,
        'capture_matches_snapshot_rig_translation': max_translation < .02,
        'capture_matches_snapshot_rig_rotation': max_rotation < 1e-4,
        'ego_actually_moves': ego_displacement > .01,
        'instance_semantic_parity': disagreement / pixels < .005,
        'known_ids_semantically_compatible': all(not c['unexpected_semantic_tags'] for f in candidate_frames for v in f['views'].values() for c in v)}
    result = {'task': 'moving-instance-probe-audit', 'status': 'passed' if all(checks.values()) else 'failed',
        'training_approved': False, 'scope': 'Five saved bundles from one seen Town01 group; not a v3 dataset or quality benchmark',
        'checks': checks, 'checked_images': len(inventory), 'max_timestamp_error_s': max_stamp_error,
        'max_modality_pose_error': max_pose, 'max_capture_vs_snapshot_rig_translation_m': max_translation,
        'max_capture_vs_snapshot_rig_rotation_matrix_error': max_rotation, 'ego_displacement_m': ego_displacement,
        'semantic_disagreement_fraction': disagreement / pixels, 'candidate_frames': candidate_frames,
        'auditor_sha256': sha(Path(__file__)), 'geometry_source_sha256': sha(ROOT / 'src/senior_ai/instance_labels.py')}
    (root / 'image_inventory.jsonl').write_text(''.join(json.dumps(i) + '\n' for i in inventory))
    write_json(root / 'audit-instance-motion.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'candidate_frames'}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    raise SystemExit(audit(parser.parse_args().root.resolve()))
