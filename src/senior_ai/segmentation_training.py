"""Reproducible grouped segmentation experiment; no simulator connection."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .core import write_json
from .dataset import safe_path
from .segmentation_data import SegmentationSamples
from .segmentation_metrics import confusion, metrics, prior_baselines
from .segmentation_model import TinyUNet, colorize, map_target


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing silent CPU fallback')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_config(config):
    if config['ignore_index'] != 255 or config['ignore_raw_ids'] != [0]:
        raise ValueError('This experiment implementation requires raw 0 -> ignore 255')
    if len(config['class_names']) != 8 or len(config['raw_ids']) != 8 or config['raw_ids'][0] is not None:
        raise ValueError('Expected 8 model classes, beginning with other_annotated')
    ids = config['raw_ids'][1:]
    if len(set(ids)) != 7 or any(i < 1 or i > 28 for i in ids):
        raise ValueError('Invalid focus class mapping')
    if config['width'] < 4 or config['width'] % 4 or config['epochs'] < 1 or config['batch_size'] < 1:
        raise ValueError('Invalid model/training dimensions')
    if config['learning_rate'] <= 0 or config['pretrained'] is not False:
        raise ValueError('Only positive-LR scratch training is implemented')


def fingerprints(root, repository):
    """Rehash inventory bytes AND metadata; detect cross-split exact RGB duplicates."""
    plan = json.loads((root / 'plan.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    acquisition = json.loads((root / 'report.json').read_text())
    if audit['status'] != 'passed' or acquisition['status'] != 'passed':
        raise ValueError('Dataset must have passed acquisition and audit')
    if sha(root / 'image_inventory.jsonl') != audit['inventory_sha256']:
        raise ValueError('Inventory differs from audited inventory')
    split_for_scene = {s['id']: s['split'] for s in plan['scenes']}
    inventory, rgb_hashes = [], {}
    for line in (root / 'image_inventory.jsonl').read_text().splitlines():
        item = json.loads(line)
        path = safe_path(root, item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError(f'Image integrity mismatch: {item["path"]}')
        inventory.append(item)
        if item['path'].startswith('inputs/rgb/'):
            split = split_for_scene[Path(item['path']).parts[2]]
            rgb_hashes.setdefault(item['sha256'], set()).add(split)
    if any(len(splits) > 1 for splits in rgb_hashes.values()):
        raise ValueError('Cross-split exact RGB duplicate detected')
    if len({i['path'] for i in inventory}) != len(inventory):
        raise ValueError('Duplicate inventory path')
    if len(inventory) != audit['checked_pngs'] or len(inventory) != acquisition['total_images']:
        raise ValueError('Incomplete image inventory')
    metadata = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
                if p.is_file() and p.suffix in ['.json', '.jsonl']}
    source_paths = list((repository / 'src').rglob('*.py')) + [repository / 'scripts' / name for name in
                   ['train_segmentation.py', 'predict_segmentation.py']]
    source = {p.relative_to(repository).as_posix(): sha(p) for p in sorted(source_paths)}
    return {'data_sha256': digest({'metadata': metadata, 'images': inventory}),
            'source_sha256': digest(source), 'source_files': source, 'metadata_files': metadata,
            'verified_images': len(inventory), 'cross_split_exact_rgb_duplicates': 0,
            'inventory': {i['path']: i['sha256'] for i in inventory}}


class CachedSamples(Dataset):
    def __init__(self, root, split, config, inventory):
        source = SegmentationSamples(root, split)
        self.samples = source.samples
        self.rgb, self.target = [], []
        counts, positives = np.zeros(8, np.int64), np.zeros(8, np.int64)
        for i, sample in enumerate(self.samples):
            for key in ['rgb', 'semantic']:
                if sample[key] not in inventory:
                    raise ValueError(f'Unverified image referenced by index: {sample[key]}')
            item = source[i]
            if item['rgb'].shape[:2][::-1] != tuple(config['resolution_wh']):
                raise ValueError('Experiment resolution mismatch; implicit resizing forbidden')
            target = map_target(item['target'], config)
            count = np.bincount(target[target != 255], minlength=8)
            counts += count
            positives += count > 0
            self.rgb.append(torch.from_numpy(item['rgb'].transpose(2, 0, 1).copy()))
            self.target.append(torch.from_numpy(target))
        self.support = counts
        self.statistics = {'samples': len(self), 'independent_groups': sorted({s['group_id'] for s in self.samples}),
                           'class_pixels': counts.tolist(), 'positive_images': positives.tolist(),
                           'ignored_pixels': int(len(self) * np.prod(config['resolution_wh']) - counts.sum())}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.rgb[index], self.target[index], index


def loader(data, config, shuffle=False):
    # Zero workers avoids Windows process duplication of the ~1 GB in-memory cache.
    return DataLoader(data, batch_size=config['batch_size'], shuffle=shuffle, num_workers=0)


def tensors(rgb, target, augment=False):
    rgb = rgb.cuda().float().div_(255)
    if augment:
        shape = (rgb.shape[0], 1, 1, 1)
        contrast = torch.rand(shape, device='cuda') * 0.4 + 0.8
        brightness = torch.rand(shape, device='cuda') * 0.4 + 0.8
        mean = rgb.mean((2, 3), keepdim=True)
        rgb = ((rgb - mean) * contrast + mean).mul(brightness).clamp_(0, 1)
    return rgb, target.cuda().long()


def weighted_loss(logits, target, weights):
    # Explicit float32 CE keeps tiny-class loss stable under float16 autocast.
    # PyTorch CUDA NLLLoss2d mean/sum uses atomicAdd (v2.10.0). Reduce the
    # deterministic per-pixel kernel ourselves; preserve weighted-mean semantics.
    pixels = F.cross_entropy(logits.float(), target, weight=weights, ignore_index=255, reduction='none')
    denominator = weights[target[target != 255]].sum()
    if denominator <= 0:
        raise ValueError('Batch contains no supervised pixels')
    return pixels.sum() / denominator


def train_batch(model, rgb, target, optimizer, scaler, weights, clip):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        loss = weighted_loss(model(rgb), target, weights)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), float(norm)


@torch.inference_mode()
def evaluate(model, data, config, weights, plan, gallery=None):
    model.eval()
    total = np.zeros((8, 8), np.int64)
    grouped = {key: {} for key in ['scene_id', 'camera', 'weather']}
    weather = {s['id']: s['weather'] for s in plan['scenes']}
    frames, loss_numerator, loss_denominator = [], 0., 0.
    for rgb, target, indices in loader(data, config):
        rgb, target = tensors(rgb, target)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(rgb)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite evaluation logits')
        losses = F.cross_entropy(logits.float(), target, weight=weights, ignore_index=255, reduction='none').sum()
        valid = target != 255
        loss_numerator += float(losses)
        loss_denominator += float(weights[target[valid]].sum())
        predictions = logits.argmax(1).cpu().numpy().astype(np.uint8)
        targets = target.cpu().numpy()
        for local, index in enumerate(indices.tolist()):
            sample = dict(data.samples[index], weather=weather[data.samples[index]['scene_id']])
            matrix = confusion(targets[local], predictions[local])
            total += matrix
            for key, groups in grouped.items():
                groups.setdefault(sample[key], np.zeros((8, 8), np.int64))[:] += matrix
            detail = metrics(matrix, config['class_names'])
            frames.append({'index': index, **sample, 'focus_macro_iou': detail['focus_macro_iou'],
                           'classes': detail['classes']})
    result = {**metrics(total, config['class_names']), 'weighted_loss': loss_numerator / loss_denominator,
        'by': {key: {name: metrics(matrix, config['class_names']) for name, matrix in groups.items()}
               for key, groups in grouped.items()}}
    if gallery is not None:
        gallery.mkdir(exist_ok=True)
        (gallery / 'frames.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames), encoding='utf-8')
        selected = set()
        for scene in grouped['scene_id']:
            selected.add(min((f for f in frames if f['scene_id'] == scene), key=lambda f: f['focus_macro_iou'])['index'])
        for cls in range(1, 8):
            candidates = [f for f in frames if f['classes'][cls]['support_pixels'] > 0]
            if candidates:
                selected.add(min(candidates, key=lambda f: f['classes'][cls]['iou'])['index'])
        result['error_gallery'] = []
        for index in sorted(selected):
            rgb, target, _ = data[index]
            x = rgb.unsqueeze(0).cuda().float() / 255
            with torch.autocast('cuda', dtype=torch.float16):
                pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
            original = rgb.permute(1, 2, 0).numpy()
            panel = Image.new('RGB', (original.shape[1] * 3, original.shape[0] + 28), 'white')
            for col, (array, title) in enumerate([(original, 'RGB'), (colorize(target.numpy()), 'GT (evaluation only)'),
                                                    (colorize(pred), 'Learned prediction')]):
                panel.paste(Image.fromarray(array), (col * original.shape[1], 28))
                ImageDraw.Draw(panel).text((col * original.shape[1] + 8, 6), title, fill='black')
            filename = f'{index:04d}.png'
            panel.save(gallery / filename)
            result['error_gallery'].append({'image': filename, **frames[index]})
    return result


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_score, config, provenance):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(), 'epoch': epoch, 'best_score': best_score, 'config': config,
        'provenance': provenance, 'rng': {'python': random.getstate(),
            'numpy': [np.random.get_state()[0], np.random.get_state()[1].tolist(), *np.random.get_state()[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Only our own tensor/primitive checkpoint format; no arbitrary custom pickle objects.
    return torch.load(path, map_location='cpu', weights_only=True)


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def cuda_smoke_and_overfit(train, config, weights):
    seed_all(config['seed'])
    model = TinyUNet(config['width']).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scaler = torch.amp.GradScaler('cuda')
    # Fixed train-only first two views, independent from formal model initialization.
    rgb = torch.stack(train.rgb[:2])
    target = torch.stack(train.target[:2])
    rgb, target = tensors(rgb, target)
    losses = []
    for _ in range(config['overfit_steps']):
        loss, _ = train_batch(model, rgb, target, optimizer, scaler, weights, config['gradient_clip_norm'])
        losses.append(loss)
    ratio = float(np.mean(losses[-5:]) / np.mean(losses[:5]))
    if ratio >= config['overfit_required_loss_ratio']:
        raise RuntimeError(f'Train-only overfit sanity failed: loss ratio {ratio}')
    return {'status': 'passed', 'steps': len(losses), 'first_5_mean_loss': float(np.mean(losses[:5])),
            'last_5_mean_loss': float(np.mean(losses[-5:])), 'loss_ratio': ratio,
            'parameter_count': sum(p.numel() for p in model.parameters())}


def environment(repository):
    import importlib.metadata
    return {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=repository, text=True),
        'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                             '--format=csv,noheader'], text=True).strip(),
        'deterministic_algorithms': True, 'tf32': False, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def runtime_signature(env):
    # Git status is evidence, not a numerical runtime dependency. Documentation may
    # change while an interrupted run retains the exact same numerical source.
    keys = ['python', 'torch', 'cuda_runtime', 'cudnn', 'gpu', 'nvidia_smi',
            'deterministic_algorithms', 'tf32', 'cublas_workspace_config']
    values = {k: env[k] for k in keys}
    values['numeric_packages'] = {k: v for k, v in env['packages'].items()
                                  if k.lower() in ['numpy', 'pillow', 'torch']}
    return digest(values)


@torch.inference_mode()
def benchmark(model, data, root, config):
    from .segmentation_data import load_inference_rgb
    rgb = data.rgb[0].unsqueeze(0).cuda().float() / 255
    def forward(x):
        with torch.autocast('cuda', dtype=torch.float16):
            return model(x)
    for _ in range(config['timing_warmup']):
        forward(rgb)
    torch.cuda.synchronize()
    network, end_to_end = [], []
    for _ in range(config['timing_samples']):
        start = time.perf_counter()
        forward(rgb)
        torch.cuda.synchronize()
        network.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        sample = data.samples[0]
        image = load_inference_rgb(root, sample['input_manifest'], sample['camera'])
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).cuda().float() / 255
        logits = forward(x).float()
        logits.argmax(1).cpu().numpy()
        logits.softmax(1).amax(1).cpu().numpy()
        torch.cuda.synchronize()
        end_to_end.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    for i in range(8):
        sample = data.samples[i]
        image = load_inference_rgb(root, sample['input_manifest'], sample['camera'])
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).cuda().float() / 255
        forward(x).argmax(1).cpu().numpy()
    torch.cuda.synchronize()
    return {'samples': config['timing_samples'], 'warmup': config['timing_warmup'], 'batch_size': 1,
        'network_ms': {'p50': float(np.median(network)), 'p95': float(np.percentile(network, 95))},
        'warm_file_cache_load_to_mask_and_confidence_ms': {'p50': float(np.median(end_to_end)), 'p95': float(np.percentile(end_to_end, 95))},
        'one_eight_view_sequential_mask_only_ms': (time.perf_counter() - start) * 1000,
        'excludes': ['model loading', 'sensor acquisition', 'disk output writing', 'simulator render'],
        'note': '8-view measurement is one bundle, not a p95 or a live real-time guarantee'}


def run_experiment(repository, config_path, run, resume=False):
    start = time.perf_counter()
    config = json.loads(config_path.read_text())
    validate_config(config)
    root = safe_path(repository, config['dataset'])
    if not resume:
        run.mkdir(parents=True, exist_ok=False)
        (run / 'config.json').write_bytes(config_path.read_bytes())
    elif not run.is_dir() or sha(run / 'config.json') != sha(config_path):
        raise ValueError('Resume requires same saved experiment config')
    if (run / 'test_started.json').exists():
        raise ValueError('Test already opened; refusing experiment overwrite or repeated test selection')
    configure_cuda()
    torch.cuda.reset_peak_memory_stats()
    print('preflight: verifying image inventory and source fingerprints', flush=True)
    fp = fingerprints(root, repository)
    provenance = {k: fp[k] for k in ['data_sha256', 'source_sha256']}
    provenance['config_sha256'] = sha(config_path)
    env = environment(repository)
    provenance['runtime_sha256'] = runtime_signature(env)
    if resume:
        checkpoint = load_checkpoint(run / 'last.pt')
        if checkpoint['provenance'] != provenance:
            raise ValueError('Resume data/config/source mismatch')
    else:
        write_json(run / 'provenance.json', {k: v for k, v in fp.items() if k != 'inventory'})
        write_json(run / 'environment.json', env)
        source_folder = run / 'source'
        for name in fp['source_files']:
            destination = source_folder / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((repository / name).read_bytes())
    plan = json.loads((root / 'plan.json').read_text())
    train = CachedSamples(root, 'train', config, fp['inventory'])
    validation = CachedSamples(root, 'validation', config, fp['inventory'])
    frequency = train.support / train.support.sum()
    if (train.support == 0).any():
        raise ValueError('Missing training class; revise dataset/version before training')
    raw_weights = 1 / np.sqrt(frequency)
    weights = torch.tensor(np.clip(raw_weights / raw_weights.min(), 1, 20), device='cuda', dtype=torch.float32)
    write_json(run / 'train_validation_statistics.json', {'train': train.statistics, 'validation': validation.statistics,
                                                          'train_only_loss_weights': weights.cpu().tolist()})
    if not resume:
        write_json(run / 'cuda_overfit.json', cuda_smoke_and_overfit(train, config, weights))
    seed_all(config['seed'])
    model = TinyUNet(config['width']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.amp.GradScaler('cuda')
    best_score, first_epoch = -1., 1
    history = []
    if resume:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'])
        first_epoch, best_score = checkpoint['epoch'] + 1, checkpoint['best_score']
        history = json.loads((run / 'history.json').read_text())[:checkpoint['epoch']]
    else:
        write_json(run / 'untrained_validation.json', evaluate(model, validation, config, weights, plan))
    for epoch in range(first_epoch, config['epochs'] + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses, norms = [], []
        learning_rate = optimizer.param_groups[0]['lr']
        for rgb, target, _ in loader(train, config, shuffle=True):
            rgb, target = tensors(rgb, target, augment=True)
            loss, norm = train_batch(model, rgb, target, optimizer, scaler, weights, config['gradient_clip_norm'])
            losses.append(loss)
            norms.append(norm)
        val = evaluate(model, validation, config, weights, plan)
        score = val['focus_macro_iou']
        improved = score > best_score
        best_score = max(best_score, score)
        scheduler.step()
        entry = {'epoch': epoch, 'learning_rate': learning_rate, 'train_mean_batch_weighted_loss': float(np.mean(losses)),
                 'max_preclip_gradient_norm': max(norms), 'validation': val, 'selected': improved,
                 'elapsed_s': time.perf_counter() - epoch_start}
        history.append(entry)
        write_json(run / 'history.json', history)
        if improved:
            save_checkpoint(run / 'best.pt', model, optimizer, scheduler, scaler, epoch, best_score, config, provenance)
        save_checkpoint(run / 'last.pt', model, optimizer, scheduler, scaler, epoch, best_score, config, provenance)
        print(json.dumps({'epoch': epoch, 'loss': float(np.mean(losses)), 'validation_focus_mIoU': score,
                          'best': best_score, 'elapsed_s': entry['elapsed_s']}), flush=True)
    best = load_checkpoint(run / 'best.pt')
    model.load_state_dict(best['model'])
    model.eval()
    # Independent model instance reload; exact forward equality in the same environment.
    clone = TinyUNet(config['width']).cuda().eval()
    clone.load_state_dict(load_checkpoint(run / 'best.pt')['model'])
    probe = validation.rgb[0].unsqueeze(0).cuda().float() / 255
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        if not torch.equal(model(probe), clone(probe)):
            raise ValueError('Checkpoint roundtrip forward mismatch')
    del clone
    val = evaluate(model, validation, config, weights, plan, run / 'validation_errors')
    write_json(run / 'validation.json', {**val, 'baselines': prior_baselines(validation.support, train.support, config['class_names'])})
    # No test model forward occurs before this durable marker.
    write_json(run / 'test_started.json', {'best_epoch': best['epoch'], 'checkpoint_sha256': sha(run / 'best.pt'),
        'selection': config['selection'], 'provenance': provenance})
    test = CachedSamples(root, 'test', config, fp['inventory'])
    write_json(run / 'test_statistics.json', test.statistics)
    result = evaluate(model, test, config, weights, plan, run / 'test_errors')
    write_json(run / 'test.json', {**result, 'baselines': prior_baselines(test.support, train.support, config['class_names'])})
    timing = benchmark(model, test, root, config)
    write_json(run / 'latency.json', timing)
    receipt = {'task': 'segmentation-exploratory-training', 'status': 'completed', 'quality_acceptance': 'not_defined',
        'best_epoch': best['epoch'], 'validation_focus_macro_iou': val['focus_macro_iou'],
        'test_focus_macro_iou': result['focus_macro_iou'], 'checkpoint_sha256': sha(run / 'best.pt'),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_vram_bytes': torch.cuda.max_memory_reserved(), 'elapsed_this_invocation_s': time.perf_counter() - start,
        'checkpoint_forward_roundtrip': 'passed', 'provenance': provenance,
        'limits': ['6 scene groups in shared Town01', 'validation=1 group, test=2 groups',
                   '7 focus semantic classes + other; no instance detection, light states or 3D',
                   'softmax confidence uncalibrated', 'not a driving-quality acceptance']}
    write_json(run / 'report.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt
