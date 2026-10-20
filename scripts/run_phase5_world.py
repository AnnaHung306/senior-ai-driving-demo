"""RGB-only WorldFrame replay using hash-bound Phase4 predictions. No GT imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
from senior_ai.bev_geometry import cameras_from_calibration
from senior_ai.cuboid import CuboidNet, geometry_features, crop, decode, prior_box, fuse_cuboids
from senior_ai.world_tracking import WorldTracker
from senior_ai.semantic_occupancy import VoxelGrid, build_occupancy
from senior_ai.dataset import safe_path, read_sensor_frame
from senior_ai.core import write_json
from phase5_render import panel


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(args):
    root = args.root.resolve();parent = args.phase4.resolve();out = args.output.resolve();config = read(args.config)
    if out.exists():raise FileExistsError('New WorldFrame run required')
    spec = read(args.input_list);inventory = read(args.inventory)
    if set(spec) != {'schema_version', 'scope', 'inventory_sha256', 'sequences'} or spec['schema_version'] != 1:
        raise ValueError('Strict RGB sequence schema required')
    if sha(args.inventory) != spec['inventory_sha256']:raise ValueError('RGB inventory changed')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(inventory) or any(not n.startswith(('inputs/', 'calibration/')) for n in actual):
        raise ValueError('RGB staging contains extra/GT files')
    for relative, digest in inventory.items():
        if sha(safe_path(root, relative)) != digest:raise ValueError('Changed RGB staging file')
    for sequence in spec['sequences']:
        if set(sequence) != {'scene_id', 'manifests'}:raise ValueError('GT fields forbidden in sequence')
    bound = read(args.phase4_integrity)
    if bound['status'] != 'passed_artifact_integrity_not_formal_quality':raise ValueError('Phase4 integrity proof required')
    for relative, digest in bound['output_manifest'].items():
        if sha(safe_path(parent, relative)) != digest:raise ValueError('Phase4 output changed: '+relative)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['task'] != 'RGB_crop_3D_regression' or saved['config'] != config:raise ValueError('Wrong 3D model/config')
    for relative, digest in saved['contract']['sources'].items():
        if sha(ROOT/relative) != digest:raise ValueError('3D model source changed: '+relative)
    if args.device == 'cuda':
        from train_sunny import require_idle_gpu
        require_idle_gpu()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model = CuboidNet().to(args.device).eval();model.load_state_dict(saved['model']);grid = VoxelGrid(**config['occupancy'])
    priors = saved['class_size_priors'];supported = saved['supported_class_ids'];tracker = WorldTracker(config['tracker'])
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT/'src/senior_ai').glob('*.py')}
    for name in ['scripts/run_phase5_world.py', 'scripts/phase5_render.py', 'configs/phase5_world_v1.json']:sources[name] = sha(ROOT/name)
    out.mkdir(parents=True)
    for name in sources:
        dest = out/'source'/name;dest.parent.mkdir(parents=True, exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    write_json(out/'contract.json', {'task': 'RGB_learned_cuboid_WorldFrame', 'sources': sources, 'config_sha256': sha(args.config),
               'checkpoint_sha256': sha(args.checkpoint), 'training_contract': saved['contract'],
               'phase4_contract_sha256': sha(parent/'contract.json'), 'phase4_integrity_sha256': sha(args.phase4_integrity),
               'RGB_inventory_sha256': sha(args.inventory), 'input_list_sha256': sha(args.input_list),
               'inference_GT_inputs': False, 'control_issued': False, 'derived_parent_predictions': 'Hash-bound RGB-only Phase4 output'})
    write_json(out/'config.json', config);records = []
    for sequence in spec['sequences']:
        for manifest in sequence['manifests']:
            frame = read_sensor_frame(root, manifest)
            if frame['scene_id'] != sequence['scene_id']:raise ValueError('Mixed scene')
            folder = Path(frame['scene_id'])/str(frame['frame']);previous = read(parent/folder/'frame.json')
            if previous['manifest_sha256'] != sha(root/manifest) or previous['timestamp_s'] != frame['timestamp_s']:raise ValueError('Parent/input mismatch')
            cameras = cameras_from_calibration(read(root/frame['calibration']));detections = read(parent/folder/'detections.json')
            if args.device == 'cuda':torch.cuda.synchronize()
            started = time.perf_counter();rgb = {n: np.asarray(Image.open(root/v['image']).convert('RGB')) for n, v in frame['cameras'].items()}
            candidates = [];features = [];crops = [];rejected = []
            for name, camera in cameras.items():
                for index, detection in enumerate(detections[name]):
                    if detection['score'] < config['minimum_detection_score'] or detection['class_id'] not in supported:continue
                    geometry = geometry_features(camera, detection['bbox_xyxy'], detection['class_id'], config['maximum_anchor_range_m'])
                    if geometry is None:
                        rejected.append({'camera': name, 'detection_index': index, 'reason': 'truncated_small_or_ground_anchor_unavailable'});continue
                    f, anchor = geometry;features.append(f);crops.append(crop(rgb[name], detection['bbox_xyxy'], config['crop_size']))
                    candidates.append((name, index, detection, anchor))
            predictions = np.empty((0, 8))
            if candidates:
                with torch.inference_mode():predictions = model(torch.from_numpy(np.stack(crops)).to(args.device).float()/255, torch.from_numpy(np.stack(features)).to(args.device)).cpu().numpy()
            single = [];baseline = []
            for (name, index, detection, anchor), pred in zip(candidates, predictions):
                extra = {'camera': name, 'detection_index': index, 'bbox_xyxy': detection['bbox_xyxy'], 'score_uncalibrated': detection['score']}
                obj = decode(pred, anchor, cameras[name], detection['class_id'], priors, config['center_residual_scale_m'])
                base = prior_box(anchor, cameras[name], detection['class_id'], priors)
                if np.linalg.norm(base['center_ego_m'][:2]) <= 30:
                    baseline.append({**base, **extra})
                if np.linalg.norm(obj['center_ego_m'][:2]) > 30:
                    rejected.append({**extra, 'reason': 'learned_center_outside_30m'});continue
                single.append({**obj, **extra})
            unique = fuse_cuboids(single, config['merge_distance_m']);prior_unique = fuse_cuboids(baseline, config['merge_distance_m'])
            measured, tracking = tracker.update(unique, frame['scene_id'], frame['timestamp_s'], sha(root/frame['calibration']), previous['motion'])
            if args.device == 'cuda':torch.cuda.synchronize()
            object_ms = (time.perf_counter()-started)*1000;occupancy_start = time.perf_counter()
            with np.load(parent/folder/'camera_observations.npz') as a:ids = {name: a[name+'_ids'] for name in cameras}
            values, occupancy = build_occupancy(grid, cameras, ids, measured);occupancy_ms = (time.perf_counter()-occupancy_start)*1000
            target = out/folder;target.mkdir(parents=True)
            np.savez_compressed(target/'occupancy.npz', semantic=values)
            world = {'schema_version': 1, 'source': 'learned_RGB_3D_head_plus_geometric_occupancy', 'scene_id': frame['scene_id'],
                     'frame': frame['frame'], 'timestamp_s': frame['timestamp_s'], 'coordinate_frame': 'current_ego', 'axes': 'x_forward_y_right_z_up',
                     'units': {'position': 'meters', 'size': 'full_LWH_meters', 'angle': 'radians'}, 'objects': measured,
                     'prior_baseline_objects': prior_unique, 'tracking': tracking, 'rejected': rejected, 'single_camera_cuboids': len(single),
                     'occupancy': {**occupancy, 'file': 'occupancy.npz', 'sha256': sha(target/'occupancy.npz')}, 'occupancy_grid': config['occupancy'],
                     'road_path': previous['path'], 'parent_manifest': manifest, 'parent_frame_sha256': sha(parent/folder/'frame.json'),
                     'timing_ms': {'decode_crops_3D_fusion_tracking': object_ms, 'occupancy': occupancy_ms, 'upstream_Phase4_compute': sum(previous['timing_ms'].values())},
                     'limitations': ['upright cuboids and flat ground', 'Orientation vector norm/consensus is a validity heuristic, not calibrated accuracy',
                                     'No velocity/future prediction, fine surfaces or obstacle-clear path', 'Known short train/validation pilot, no formal quality pass'],
                     'control_issued': False, 'GT_used_in_inference': False}
            write_json(target/'world.json', world)
            panel(rgb, cameras, measured, values, grid, f"{frame['scene_id']} frame{frame['frame']} | learned boxes={len(measured)} | 3D prototype; low-confidence orientation").save(target/'prediction.png')
            record = {'scene_id': frame['scene_id'], 'frame': frame['frame'], 'folder': str(folder), 'objects': len(measured), 'object_ms': object_ms, 'occupancy_ms': occupancy_ms}
            records.append(record);write_json(out/'progress.json', {'bundles': len(records), 'latest': record});print(json.dumps(record), flush=True)
    for name, digest in sources.items():
        if sha(ROOT/name) != digest:raise ValueError('World implementation changed during replay')
    write_json(out/'state.json', {'status': 'completed_world_inference_pending_independent_evaluation', 'bundles': len(records), 'records': records,
               'GT_used_in_inference': False, 'control_issued': False, 'source_integrity_exact': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'phase4', 'phase4-integrity', 'input-list', 'inventory', 'checkpoint', 'config', 'output']:p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda');args = p.parse_args()
    try:main(args)
    except Exception as e:
        if args.output.exists():write_json(args.output/'failure.json', {'message': str(e), 'traceback': traceback.format_exc(), 'auto_retry': False})
        raise
