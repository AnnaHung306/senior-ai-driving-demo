"""One bounded from-scratch CUDA crop-3D experiment, with epoch checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import json
import sys
import time
import traceback
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
from senior_ai.cuboid import CuboidNet, wrap, box_iou
from senior_ai.core import write_json
from senior_ai.segmentation_training import configure_cuda, seed_all
from phase5_gt import read, sha


def objective(output, targets):
    center = F.smooth_l1_loss(output[:, :3], targets[:, :3])
    size = F.smooth_l1_loss(output[:, 3:6], targets[:, 3:6])
    orientation = F.mse_loss(output[:, 6:8], targets[:, 6:8])
    return center+.5*size+.5*orientation


def condition_metrics(predictions, rows, priors, scale):
    records = []
    for pred, row in zip(predictions, rows):
        category = row['class_id'];prior = np.array(priors[str(category)]);truth = row['camera_target']
        base_center = np.array(row['anchor_local_m'])+np.array([0., 0., prior[2]/2])
        center = np.array(row['anchor_local_m'])+np.clip(pred[:3], -2, 2)*scale
        size = prior*np.exp(np.clip(pred[3:6], -1, 1));yaw = np.arctan2(pred[7], pred[6])
        p = {'center_ego_m': center.tolist(), 'size_lwh_m': size.tolist(), 'yaw_rad': float(yaw)}
        base = {'center_ego_m': base_center.tolist(), 'size_lwh_m': prior.tolist(), 'yaw_rad': 0.}
        scores = {}
        for name, box in [('learned', p), ('prior', base)]:
            error = np.asarray(box['center_ego_m'])-truth['center_ego_m'];angle = float(abs(wrap(box['yaw_rad']-truth['yaw_rad'])))
            bev_iou, iou3d = box_iou(box, truth)
            scores[name] = {'center_3D_m': float(np.linalg.norm(error)), 'center_XY_m': float(np.linalg.norm(error[:2])), 'center_Z_m': float(abs(error[2])),
                            'size_LWH_absolute_m': abs(np.array(box['size_lwh_m'])-truth['size_lwh_m']).tolist(),
                            'yaw_absolute_deg': float(np.rad2deg(angle)), 'yaw_pi_symmetric_deg': float(np.rad2deg(min(angle, np.pi-angle))),
                            'BEV_IoU': bev_iou, 'IoU3D': iou3d}
        records.append({'scene_id': row['scene_id'], 'class_id': category, 'frame': row['frame'], 'camera': row['camera'],
                        'actor_id': row['actor_id'], 'predicted_camera_local_box': p, **scores})

    def aggregate(subset):
        result = {'samples': len(subset)}
        for name in ['learned', 'prior']:
            result[name] = {key: np.asarray([r[name][key] for r in subset]).mean(0).tolist() for key in subset[0][name]} if subset else None
        return result
    return {'scope': 'GT positive untruncated 2D crop/class conditional regression, validation model selection, not end-to-end detection',
            'overall': aggregate(records), 'by_class': {str(c): aggregate([r for r in records if r['class_id'] == c]) for c in range(1, 7)},
            'by_scene': {s: aggregate([r for r in records if r['scene_id'] == s]) for s in sorted({r['scene_id'] for r in records})}, 'records': records}


def main(args):
    root = args.dataset.resolve();out = args.output.resolve();config = read(args.config);data = read(root/'contract.json')
    if out.exists():
        raise FileExistsError('New training run required; epoch checkpoints retained for explicit recovery')
    if data['config_sha256'] != sha(args.config) or not data['scenes_disjoint']:
        raise ValueError('Data/config contract mismatch')
    for name, digest in data['files'].items():
        if sha(root/name) != digest:
            raise ValueError('Prepared dataset changed')
    for name, digest in data['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Prepared target source changed')
    from train_sunny import require_idle_gpu
    require_idle_gpu();configure_cuda();seed_all(config['seed']);torch.cuda.reset_peak_memory_stats()
    out.mkdir(parents=True);sources = dict(data['sources'])
    for name in ['scripts/train_phase5_cuboids.py', 'configs/phase5_world_v1.json', 'src/senior_ai/segmentation_training.py']:
        sources[name] = sha(ROOT/name)
    for name in sources:
        dest = out/'source'/name;dest.parent.mkdir(parents=True, exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    contract = {'task': 'RGB_crop_3D_regression', 'pretrained': False, 'data_contract_sha256': sha(root/'contract.json'),
                'config_sha256': sha(args.config), 'sources': sources, 'class_size_priors': data['class_size_priors_train_unique_actors'],
                'supported_class_ids': data['supported_class_ids'], 'torch': torch.__version__, 'numpy': np.__version__,
                'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'precision': 'float32',
                'GT_model_features': False, 'GT_training_targets': True, 'development_lockbox_opened': False}
    write_json(out/'contract.json', contract);write_json(out/'config.json', config)
    arrays = {};rows = {}
    for split in ['train', 'validation']:
        with np.load(root/(split+'.npz')) as package:
            arrays[split] = {k: torch.from_numpy(package[k].copy()) for k in ['crops', 'geometry', 'targets', 'classes']}
        rows[split] = read(root/(split+'_records.json'))
    model = CuboidNet().cuda();optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    counts = Counter(arrays['train']['classes'].tolist());weights = torch.tensor([1/counts[int(c)] for c in arrays['train']['classes']], dtype=torch.float64)
    best = float('inf');history = [];first_gradient = None;best_predictions = None;updates = 0;started = time.perf_counter()

    def predict(split):
        model.eval();outputs = []
        with torch.inference_mode():
            for start in range(0, len(rows[split]), config['batch_size']):
                sl = slice(start, start+config['batch_size']);a = arrays[split]
                pred = model(a['crops'][sl].cuda().float()/255, a['geometry'][sl].cuda())
                if not torch.isfinite(pred).all():
                    raise FloatingPointError('Nonfinite validation output')
                outputs.append(pred.cpu())
        return torch.cat(outputs)

    for epoch in range(config['epochs']):
        model.train();generator = torch.Generator().manual_seed(config['seed']+epoch)
        order = torch.multinomial(weights, len(rows['train']), replacement=True, generator=generator);losses = []
        for ids in order.split(config['batch_size']):
            a = arrays['train'];optimizer.zero_grad(set_to_none=True)
            output = model(a['crops'][ids].cuda().float()/255, a['geometry'][ids].cuda());loss = objective(output, a['targets'][ids].cuda())
            if not torch.isfinite(output).all() or not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training output/loss')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient')
            if first_gradient is None:
                first_gradient = {'encoder_norm': float(model.features[0].weight.grad.norm()), 'head_norm': float(model.head[-1].weight.grad.norm())}
                if min(first_gradient.values()) <= 0:
                    raise ValueError('No meaningful encoder/head gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.);optimizer.step();updates += 1;losses.append(float(loss.detach()))
        predictions = predict('validation');score = float(objective(predictions, arrays['validation']['targets']))
        improved = score < best
        checkpoint = {'task': 'RGB_crop_3D_regression', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch, 'updates': updates, 'config': config, 'contract': contract,
                      'class_size_priors': contract['class_size_priors'], 'supported_class_ids': contract['supported_class_ids'],
                      'CPU_RNG': torch.get_rng_state(), 'CUDA_RNG': torch.cuda.get_rng_state_all()}
        torch.save(checkpoint, out/'last.pending.pt');(out/'last.pending.pt').replace(out/'last.pt')
        if improved:
            best = score;torch.save(checkpoint, out/'best.pt');best_predictions = predictions.numpy().copy()
        record = {'epoch': epoch, 'updates': updates, 'training_loss_mean': float(np.mean(losses)), 'validation_objective': score, 'best': improved}
        history.append(record);write_json(out/'history.json', history);write_json(out/'state.json', {'status': 'training', **record})
        print(json.dumps(record), flush=True)
    saved = torch.load(out/'best.pt', map_location='cpu', weights_only=False);model.load_state_dict(saved['model'])
    reloaded = predict('validation').numpy()
    if not np.array_equal(reloaded, best_predictions):
        raise ValueError('Checkpoint reload not exact')
    if not any(not torch.equal(v.detach().cpu(), initial[k]) for k, v in model.state_dict().items()):
        raise ValueError('No parameter updates')
    for name, digest in sources.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Training source changed during run')
    np.save(out/'validation_predictions.npy', reloaded)
    write_json(out/'conditional_validation.json', condition_metrics(reloaded, rows['validation'], contract['class_size_priors'], config['center_residual_scale_m']))
    write_json(out/'state.json', {'status': 'completed_finite_baseline_training', 'epochs': config['epochs'], 'updates': updates,
               'best_epoch': saved['epoch'], 'best_checkpoint_sha256': sha(out/'best.pt'), 'validation_reload_exact': True,
               'first_gradient': first_gradient, 'parameters': sum(p.numel() for p in model.parameters()),
               'peak_cuda_MiB': torch.cuda.max_memory_allocated()/1024**2, 'elapsed_s': time.perf_counter()-started,
               'quality_formally_passed': False, 'source_integrity_exact': True})
    print(json.dumps(read(out/'state.json')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--dataset', type=Path, required=True);p.add_argument('--config', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        main(args)
    except Exception as e:
        if args.output.exists():
            write_json(args.output/'failure.json', {'message': str(e), 'traceback': traceback.format_exc(), 'auto_retry': False})
        raise
