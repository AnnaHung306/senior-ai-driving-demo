"""Fresh frozen-model CUDA interventions and independent raw target checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
from senior_ai.cuboid import CuboidNet
from senior_ai.core import write_json
from senior_ai.bev_geometry import cameras_from_calibration
from senior_ai.segmentation_training import configure_cuda, seed_all
from phase5_gt import read, sha, audited_inventory
from train_phase5_cuboids import condition_metrics


def main(args):
    out = args.output
    if out.exists():raise FileExistsError('New runtime audit output required')
    model_root = ROOT/'artifacts/runs/phase5-cuboid-v1-001';data_root = ROOT/'artifacts/runs/phase5-crops-v1-002'
    contract = read(model_root/'contract.json');data_contract = read(data_root/'contract.json');root = Path(data_contract['dataset'])
    original_checkpoint_sha = sha(model_root/'best.pt')
    assert original_checkpoint_sha == read(model_root/'state.json')['best_checkpoint_sha256']
    for name, digest in contract['sources'].items():assert sha(ROOT/name) == digest
    for name, digest in data_contract['files'].items():assert sha(data_root/name) == digest
    _, inventory = audited_inventory(root, ROOT);bound = read(data_root/'source_inventory.json')
    verified = set();labels = {};calibrations = {};images = {};errors = [];crops_checked = 0
    def checked(relative):
        if relative not in verified:
            assert sha(root/relative) == bound[relative] == inventory[relative], relative;verified.add(relative)
        return root/relative
    for split in ['train', 'validation']:
        with np.load(data_root/(split+'.npz')) as package:rgb_crops = package['crops']
        rows = read(data_root/(split+'_records.json'))
        for index, row in enumerate(rows):
            relative = f"labels/frames/{row['scene_id']}/{row['frame']:08}.json"
            if relative not in labels:labels[relative] = read(checked(relative))
            label = labels[relative]
            if row['calibration'] not in calibrations:calibrations[row['calibration']] = cameras_from_calibration(read(checked(row['calibration'])))
            actor = next(a for a in label['objects_3d'] if a['actor_id'] == row['actor_id'])
            vertices = np.array(actor['world_vertices_m'], dtype=np.float64)
            T = np.array(label['capture_world_from_camera'][row['camera']]['rgb'], dtype=np.float64)
            local_center = np.linalg.solve(T[:3, :3], vertices.mean(0)-T[:3, 3])
            heading = np.linalg.solve(T[:3, :3], vertices[4]-vertices[0]);yaw = np.arctan2(heading[1], heading[0])
            center_error = float(np.max(abs(local_center-row['camera_target']['center_ego_m'])))
            yaw_error = float(abs((yaw-row['camera_target']['yaw_rad']+np.pi)%(2*np.pi)-np.pi))
            size_error = float(np.max(abs(2*np.array(actor['extent_m'])-row['camera_target']['size_lwh_m'])))
            if max(center_error, yaw_error, size_error) > 1e-6:raise ValueError('Raw GT target transform mismatch')
            errors.append([center_error, yaw_error, size_error])
            if row['image'] not in images:images[row['image']] = Image.open(checked(row['image'])).convert('RGB')
            x0, y0, x1, y1 = row['bbox_xyxy'];image = images[row['image']]
            bounds = (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))), min(image.width, int(np.ceil(x1))), min(image.height, int(np.ceil(y1))))
            patch = np.asarray(image.crop(bounds).resize((64, 64), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
            assert np.array_equal(patch, rgb_crops[index]);crops_checked += 1
    from train_sunny import require_idle_gpu
    require_idle_gpu();configure_cuda();seed_all(20261020)
    checkpoint = torch.load(model_root/'best.pt', map_location='cpu', weights_only=False)
    model = CuboidNet().cuda().eval();model.load_state_dict(checkpoint['model'])
    with np.load(data_root/'validation.npz') as package:
        crops = package['crops'].copy();features = package['geometry'].copy();classes = package['classes'].copy()
    rng = np.random.default_rng(20261020);permutation = np.arange(len(classes))
    for c in np.unique(classes):
        indices = np.flatnonzero(classes == c);permutation[indices] = rng.permutation(indices)
    interventions = [('RGB_original', crops, features), ('RGB_zero', np.zeros_like(crops), features),
                     ('RGB_permuted_within_class', crops[permutation], features),
                     ('geometry_permuted_within_class', crops, features[permutation])]
    results = {};baseline = None;rows = read(data_root/'validation_records.json')
    with torch.inference_mode():
        for name, pixels, geometry in interventions:
            outputs = []
            for start in range(0, len(classes), 64):
                p = model(torch.from_numpy(pixels[start:start+64]).cuda().float()/255, torch.from_numpy(geometry[start:start+64]).cuda())
                assert torch.isfinite(p).all();outputs.append(p.cpu().numpy())
            predicted = np.concatenate(outputs)
            if name == 'RGB_original':
                baseline = predicted
                if not np.array_equal(predicted, np.load(model_root/'validation_predictions.npy')):raise ValueError('Fresh CUDA validation is not exact checkpoint replay')
            metrics = condition_metrics(predicted, rows, checkpoint['class_size_priors'], checkpoint['config']['center_residual_scale_m'])
            results[name] = {'overall': metrics['overall'], 'by_class': metrics['by_class'],
                             'max_output_change_vs_original': float(np.max(abs(predicted-baseline))),
                             'mean_output_change_vs_original': float(np.mean(abs(predicted-baseline)))}
    assert sha(model_root/'best.pt') == original_checkpoint_sha
    for name, digest in contract['sources'].items():assert sha(ROOT/name) == digest
    out.mkdir(parents=True)
    for name in ['scripts/audit_phase5_runtime.py', 'src/senior_ai/cuboid.py']:
        target = out/'source'/name;target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write_json(out/'report.json', {'status': 'completed_frozen_model_runtime_audit', 'checkpoint_sha256': original_checkpoint_sha,
        'fresh232_CUDA_validation_exact': True, 'all891_RGB_crops_exact_to_original_files': crops_checked == 891,
        'raw_GT_targets_checked': len(errors), 'raw_transform_max_center_yaw_size_errors': np.max(errors, axis=0).tolist(),
        'original_files_reverified': len(verified), 'interventions': results, 'weights_updated': False,
        'scope': 'Known model-selection validation; interventions diagnose input dependence, not a retrained geometry-only baseline or heldout causal generalization',
        'tool_sha256': sha(Path(__file__)), 'quality_passed': False})
    print({'status': 'completed', 'targets': len(errors), 'crops': crops_checked, 'CUDA_replay_exact': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True);main(p.parse_args())
