"""Fixed grouped GT-visible crop dataset; no model inference or heldout reads."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
from senior_ai.bev_geometry import cameras_from_calibration
from senior_ai.cuboid import crop, geometry_features
from senior_ai.dataset import safe_path, read_sensor_frame
from senior_ai.core import write_json
from phase5_gt import read, sha, target_box, audited_inventory


def main(args):
    root = args.root.resolve();out = args.output.resolve();config = read(args.config)
    if out.exists():
        raise FileExistsError('New crop dataset output required')
    out.mkdir(parents=True)
    golden, inventory = audited_inventory(root, ROOT);verified = {};sets = {};scene_sets = {}

    def verify(relative):
        if relative not in verified:
            path = safe_path(root, relative);digest = sha(path)
            if inventory.get(relative) != digest:
                raise ValueError('Unbound/changed source file: '+relative)
            verified[relative] = digest

    for split in ['train', 'validation']:
        index = f'training/{split}.jsonl';verify(index)
        groups = defaultdict(list);seen = {}
        for line in (root/index).read_text(encoding='utf-8').splitlines():
            row = __import__('json').loads(line)
            if row['split'] != split:
                raise ValueError('Wrong split label')
            key = (row['scene_id'], row['frame'])
            if key in seen:
                if seen[key] != row['input_manifest']:
                    raise ValueError('Conflicting canonical camera manifests')
                continue
            seen[key] = row['input_manifest']
            row['input'] = row['input_manifest']
            row['label'] = f"labels/frames/{row['scene_id']}/{row['frame']:08}.json"
            groups[row['scene_id']].append(row)
        scene_sets[split] = sorted(groups)
        rows = [];crops = [];features = [];excluded = Counter()
        for scene, frames in sorted(groups.items()):
            for row in sorted(frames, key=lambda r: r['frame'])[::config['data_stride']]:
                verify(row['input']);verify(row['label'])
                frame = read_sensor_frame(root, row['input']);label = read(root/row['label'])
                verify(frame['calibration']);cameras = cameras_from_calibration(read(root/frame['calibration']))
                world_from_ego = np.asarray(label['capture_world_from_camera']['front']['rgb'])@cameras['front'].invT
                ego_from_world = np.linalg.inv(world_from_ego)
                actors = {a['actor_id']: a for a in label['objects_3d']}
                for name, camera in cameras.items():
                    visible_path = f"labels/visible/{scene}/{name}/{row['frame']:08}.json";verify(visible_path)
                    image = None
                    for obj in read(root/visible_path)['positive_objects']:
                        category = obj['category_id']
                        if category not in range(1, 7):
                            excluded['non_actor_class'] += 1;continue
                        geometry = geometry_features(camera, obj['visible_bbox_xyxy'], category, config['maximum_anchor_range_m'])
                        if geometry is None:
                            excluded['truncation_small_or_invalid_ground_anchor'] += 1;continue
                        actor = actors[obj['actor_id']];ego_box = target_box(actor, ego_from_world)
                        if np.linalg.norm(ego_box['center_ego_m'][:2]) > config['maximum_GT_center_range_m']:
                            excluded['GT_center_out_of_fixed_range'] += 1;continue
                        camera_box = target_box(actor, np.linalg.inv(np.asarray(label['capture_world_from_camera'][name]['rgb'])))
                        if image is None:
                            verify(frame['cameras'][name]['image']);image = np.asarray(Image.open(root/frame['cameras'][name]['image']).convert('RGB'))
                        f, anchor = geometry
                        crops.append(crop(image, obj['visible_bbox_xyxy'], config['crop_size']));features.append(f)
                        rows.append({'scene_id': scene, 'frame': row['frame'], 'camera': name, 'actor_id': obj['actor_id'],
                                     'class_id': category, 'bbox_xyxy': obj['visible_bbox_xyxy'], 'anchor_local_m': anchor.tolist(),
                                     'camera_target': camera_box, 'ego_target': ego_box, 'calibration': frame['calibration'],
                                     'image': frame['cameras'][name]['image'], 'input_manifest': row['input']})
        if not rows:
            raise ValueError('No eligible crops for '+split)
        sets[split] = (rows, crops, features, dict(excluded))
        print(f'{split}: {len(rows)} crops, {len(groups)} whole scenes', flush=True)
    if set(scene_sets['train']) & set(scene_sets['validation']):
        raise ValueError('Scene leakage')
    unique = {c: {} for c in range(1, 7)}
    for row in sets['train'][0]:
        unique[row['class_id']][(row['scene_id'], row['actor_id'])] = row['camera_target']['size_lwh_m']
    priors = {str(c): np.median(list(items.values()), axis=0).tolist() for c, items in unique.items() if items}
    counts = Counter(r['class_id'] for r in sets['train'][0])
    supported = sorted(c for c, n in counts.items() if n >= config['minimum_class_samples'])
    for split, (rows, crops, features, excluded) in sets.items():
        targets = [];eligible = []
        for i, row in enumerate(rows):
            c = row['class_id']
            if str(c) not in priors:
                excluded['class_absent_in_train'] = excluded.get('class_absent_in_train', 0)+1;continue
            box = row['camera_target'];yaw = box['yaw_rad']
            center = (np.array(box['center_ego_m'])-row['anchor_local_m'])/config['center_residual_scale_m']
            targets.append(np.r_[center, np.log(np.array(box['size_lwh_m'])/priors[str(c)]), np.cos(yaw), np.sin(yaw)])
            eligible.append(i)
        np.savez_compressed(out/(split+'.npz'), crops=np.stack(crops)[eligible], geometry=np.stack(features)[eligible],
                            targets=np.asarray(targets, np.float32), classes=np.array([rows[i]['class_id'] for i in eligible]))
        write_json(out/(split+'_records.json'), [rows[i] for i in eligible])
        sets[split] = {'crops': len(eligible), 'class_counts': dict(Counter(rows[i]['class_id'] for i in eligible)), 'excluded': excluded, 'scenes': scene_sets[split]}
    write_json(out/'source_inventory.json', verified)
    sources = ['scripts/prepare_phase5_crops.py', 'scripts/phase5_gt.py', 'src/senior_ai/cuboid.py', 'src/senior_ai/bev_geometry.py', 'src/senior_ai/dataset.py']
    for name in sources:
        dest = out/'source'/name;dest.parent.mkdir(parents=True, exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out/'contract.json', {'status': 'completed_fixed_train_validation_crops', 'dataset': str(root), 'approved_parent_data': golden,
               'config_sha256': sha(args.config), 'split_summaries': sets, 'class_size_priors_train_unique_actors': priors,
               'supported_class_ids': supported, 'scenes_disjoint': True, 'GT_used_as_targets_only': True,
               'no_depth_inputs_or_targets': True, 'files': files, 'sources': {n: sha(ROOT/n) for n in sources}})
    print(str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True);p.add_argument('--config', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        main(args)
    except Exception as e:
        if args.output.exists():
            write_json(args.output/'failure.json', {'message': str(e), 'traceback': traceback.format_exc()})
        raise
