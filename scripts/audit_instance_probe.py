"""Independent Pillow/NumPy audit and GT QA panels for the instance pilot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json
from senior_ai.dataset import read_sensor_frame, safe_path, validate_calibration
from senior_ai.instance_labels import decode_instance_rgb, validate_actor_ids, visible_candidate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_rgb(path, shape):
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGB':
            raise ValueError('Raw RGB PNG required')
        image.verify()
    with Image.open(path) as image:
        array = np.array(image)
    if array.shape != shape:
        raise ValueError('PNG dimensions differ from rig')
    return array


def audit(root, revision='v2'):
    if not re.fullmatch('[A-Za-z0-9_-]+', revision):
        raise ValueError('Invalid audit revision')
    config = json.loads((root / 'config.json').read_text())
    rig = json.loads((root / 'rig.json').read_text())
    acquisition = json.loads((root / 'report.json').read_text())
    if acquisition['status'] != 'passed' or acquisition['training_approved'] is not False:
        raise ValueError('Requires completed engineering acquisition, not training-approved labels')
    validate_calibration(json.loads((root / 'calibration/probe.json').read_text()), rig)
    inventory = [json.loads(s) for s in (root / 'image_inventory.jsonl').read_text().splitlines()]
    verified = {}
    shape = (rig['height'], rig['width'], 3)
    for item in inventory:
        path = safe_path(root, item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256'] or item['path'] in verified:
            raise ValueError('Inventory corruption or duplicate')
        verified[item['path']] = raw_rgb(path, shape)
    index = json.loads((root / 'index.json').read_text())
    metadata_names = ['config.json', 'rig.json', 'report.json', 'index.json', 'calibration/probe.json'] + [
        entry[key] for entry in index for key in ['input', 'label']]
    metadata_hashes = {name: sha(safe_path(root, name)) for name in metadata_names}
    results, mismatches, pixels, references = [], 0, 0, []
    depth_totals = {'axial': 0., 'range': 0., 'pixels': 0}
    unknown_foreground = Counter()
    qa = root / ('qa-' + revision); qa.mkdir(exist_ok=False)
    palette = {'car': (0, 140, 255), 'pedestrian': (255, 40, 100), 'bicycle': (255, 220, 0), 'motorcycle': (0, 230, 180)}
    for entry in index:
        feature = read_sensor_frame(root, entry['input'])
        label = json.loads(safe_path(root, entry['label']).read_text())
        if feature['frame'] != label['frame'] or feature['timestamp_s'] != label['timestamp_s'] or entry['case'] != label['case']:
            raise ValueError('GT/frame mismatch')
        validate_actor_ids(label['objects'])
        known = {a['actor_id'] for a in label['objects']}
        views = {}
        for name, camera in label['cameras'].items():
            if set(camera['modalities']) != set(config['modalities']):
                raise ValueError('Missing modality')
            for modality, image in camera['modalities'].items():
                if image['frame'] != feature['frame'] or image['timestamp_s'] != feature['timestamp_s'] or image['image'] not in verified:
                    raise ValueError('Unverified modality or frame/timestamp mismatch')
                references.append(image['image'])
                prefix = 'inputs/rgb/' if modality == 'rgb' else f'labels/{modality}/'
                if not image['image'].startswith(prefix):
                    raise ValueError('GT/input path boundary violation')
            if camera['modalities']['rgb'] != feature['cameras'][name]:
                raise ValueError('Feature RGB disagrees with label reference')
            image = lambda kind: verified[camera['modalities'][kind]['image']]
            semantic_rgb = image('semantic_segmentation')
            if semantic_rgb[..., 1:].any() or semantic_rgb[..., 0].max() > 28:
                raise ValueError('Invalid raw semantic encoding')
            semantic = semantic_rgb[..., 0]
            classes, ids = decode_instance_rgb(image('instance_segmentation'))
            mismatches += int((semantic != classes).sum()); pixels += semantic.size
            packed = image('depth').astype(np.uint32)
            depth = (packed[..., 0] + 256 * packed[..., 1] + 65536 * packed[..., 2]) * (1000 / 16777215)
            candidates = [visible_candidate(a, classes, ids, semantic, depth, camera, config['near_plane_m']) for a in label['objects']]
            for candidate in candidates:
                if candidate['kind'] == 'car' and candidate['visible_pixels']:
                    support = candidate['depth_geometry']
                    depth_totals['pixels'] += support['pixels']
                    for hypothesis in ['axial', 'range']:
                        depth_totals[hypothesis] += support['pixels'] * support[hypothesis + '_in_box_fraction']
            # Unmatched foreground must become explicit ignore regions or be resolved,
            # never implicit background negatives for a detector.
            unknown = np.isin(classes, [7, 12, 13, 14, 15, 16, 17, 18, 19]) & ~np.isin(ids, list(known))
            unknown_foreground.update({int(t): int(n) for t, n in zip(*np.unique(classes[unknown], return_counts=True))})
            panel = Image.fromarray(image('rgb')).copy()
            draw = ImageDraw.Draw(panel)
            for actor, candidate in zip(label['objects'], candidates):
                candidate['role'] = actor['role']
                bbox = candidate['visible_bbox_xyxy']
                if bbox:
                    draw.rectangle((bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), outline=palette[actor['kind']], width=2)
                    draw.text((bbox[0], max(0, bbox[1] - 12)), f'{actor["role"]} id={actor["actor_id"]}', fill=palette[actor['kind']])
            draw.rectangle((0, 0, 395, 18), fill='black')
            draw.text((3, 3), f'GT ONLY | {entry["case"]} / {name} / F={feature["frame"]}', fill='white')
            filename = f'{entry["case"]}-{name}.png'
            panel.save(qa / filename)
            views[name] = {'candidates': candidates, 'qa_image': filename,
                           'unmatched_foreground_pixels': int(unknown.sum())}
        results.append({'case': entry['case'], 'frame': feature['frame'], 'views': views})
    if set(references) != set(verified) or len(references) != len(verified):
        raise ValueError('Inventory contains unused/multiply referenced images')
    front = {case['case']: {c['role']: c for c in case['views']['front']['candidates']} for case in results}
    known_kinds = {c['kind'] for case in results for view in case['views'].values() for c in view['candidates']
                   if c['visible_pixels'] >= config['min_known_visible_pixels'] and not c['unexpected_semantic_tags']}
    checks = {'complete_160_pngs': len(verified) == 160,
        'complete_5_cases': [x['case'] for x in results] == [x['name'] for x in config['cases']],
        'all_four_known_actor_kinds_visible': known_kinds == {'car', 'pedestrian', 'bicycle', 'motorcycle'},
        'semantic_parity': mismatches / pixels <= config['max_semantic_disagreement_fraction'],
        'known_tags_compatible': all(not c['unexpected_semantic_tags'] for case in results for v in case['views'].values() for c in v['candidates']),
        'removing_occluder_reveals_target': front['revealed']['target']['visible_pixels'] > front['occluded']['target']['visible_pixels'] + config['min_known_visible_pixels'],
        'front_truncation_observed': front['truncated']['occluder'].get('touches_image_boundary', False),
        'near_plane_clipping_observed': front['near_plane']['occluder']['amodal_projection']['near_plane_clipped']}
    summary = {'task': 'instance-probe-independent-audit', 'status': 'passed' if all(checks.values()) else 'failed',
        'checks': checks, 'training_approved': False, 'checked_pngs': len(verified),
        'inventory_sha256': sha(root / 'image_inventory.jsonl'), 'auditor_sha256': sha(Path(__file__)),
        'metadata_sha256': metadata_hashes, 'qa_directory': qa.name, 'revision': revision,
        'semantic_disagreement_pixels': mismatches, 'semantic_disagreement_fraction': mismatches / pixels,
        'front_target_visible_pixels': {name: roles['target']['visible_pixels'] for name, roles in front.items()},
        'car_depth_geometry_hypotheses': {h: depth_totals[h] / depth_totals['pixels'] if depth_totals['pixels'] else None for h in ['axial', 'range']},
        'unmatched_foreground_semantic_pixels': dict(unknown_foreground), 'cases': results,
        'limitations': ['Controlled static GT engineering fixture; not natural driving data',
            'No training-approved labels or detector AP', 'Unknown foreground requires resolution/ignore-aware training',
            'Mask/bbox fill is not an occlusion fraction', 'Visual review must be recorded separately']}
    destination = root / f'audit-{revision}.json'
    if destination.exists():
        raise FileExistsError('Audit revision already exists; retain prior evidence')
    write_json(destination, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}))
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--revision', default='v2')
    args = parser.parse_args()
    raise SystemExit(audit(args.root.resolve(), args.revision))
