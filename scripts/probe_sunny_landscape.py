"""CPU probe on the four previously selected Tiny train/validation examples.

Reuses the historical sample selection exactly. This small, biased diagnostic
does not estimate full training accuracy or substitute for validation gates.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from PIL import Image, ImageDraw
import torch
from check_sunny_segmentation_artifacts import check
from predict_sunny import load_model, PALETTE
from senior_ai.core import write_json
from senior_ai.dataset import safe_path
from senior_ai.sunny_data import sha, png, segmentation_target, SEGMENTATION_NAMES
from senior_ai.segmentation_metrics import confusion, metrics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    run, root, output = args.run.resolve(), args.root.resolve(), args.output.resolve()
    if output.is_relative_to(run) or output.is_relative_to(root):
        raise ValueError('Diagnostic must be outside dataset/run')
    health = check(run)
    selection = ROOT / 'evidence/phase3/sunny-v3/tiny-early-overfit-diagnosis.json'
    baseline = read(selection)
    cases = baseline['diagnostics']
    if len(cases) != 4 or any(c['split'] not in ['train', 'validation'] or c['category'] not in [12, 13] for c in cases):
        raise ValueError('Unexpected fixed probe selection')
    signature = read(run / 'provenance.json')
    for filename, key in [('audit.json', 'audit_sha256'), ('visual_review.json', 'review_sha256'),
                          ('image_inventory.jsonl', 'image_inventory_sha256'), ('metadata_inventory.json', 'metadata_inventory_sha256')]:
        if sha(root / filename) != signature['data'][key]:
            raise ValueError('Bound dataset inventory changed')
    for relative, digest in signature['source']['files'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Runtime source differs from model snapshot')
    metadata = {r['path']: r['sha256'] for r in read(root / 'metadata_inventory.json')}
    canonical = {}
    for split in ['train', 'validation']:
        name = f'training/{split}.jsonl'
        if sha(root / name) != metadata[name]:
            raise ValueError('Canonical index changed')
        canonical[split] = {r['rgb']: r for r in [json.loads(line) for line in (root / name).read_text(encoding='utf-8').splitlines()]}
    for case in cases:
        if case['sample'] != canonical[case['split']][case['sample']['rgb']]:
            raise ValueError('Probe sample differs from canonical index')
    required = {c['sample'][key] for c in cases for key in ['rgb', 'semantic', 'instance']}
    verified = {}
    with (root / 'image_inventory.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item['path'] in required:
                path = safe_path(root, item['path'])
                if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
                    raise ValueError('Selected raw bytes changed')
                verified[item['path']] = item['sha256']
    if set(verified) != required:
        raise ValueError('Incomplete raw verification')
    identity = {'checkpoint_sha256': health['hashes']['best.pt'], 'selection_sha256': sha(selection),
                'tool_sha256': sha(Path(__file__)), 'selected_raw_sha256': verified}
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'report.json'
    report = read(report_path) if report_path.exists() else {'identity': identity, 'status': 'in_progress', 'cases': [],
        'scope': 'Four fixed examples only, CPU float32. Historical Tiny epoch1 and current Mobile best epoch6; not matched training budgets or full train/generalization estimates.',
        'selection': baseline['method'], 'baseline_checkpoint_sha256': baseline['checkpoint_sha256']}
    if report['identity'] != identity:
        raise ValueError('Resume identity differs')
    for case in report['cases']:
        if sha(output / case['panel']) != case['panel_sha256']:
            raise ValueError('Panel changed')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    model, _, _ = load_model(run / 'best.pt', 'cpu')
    completed = {c['id'] for c in report['cases']}
    for case in cases:
        name = case['split'] + '-' + SEGMENTATION_NAMES[case['category']]
        if name in completed:
            continue
        row = case['sample']; rgb = png(safe_path(root, row['rgb']))
        target = segmentation_target(png(safe_path(root, row['semantic']))[..., 0], png(safe_path(root, row['instance'])), row['verified_obstacle_actor_ids'])
        with torch.inference_mode():
            prediction = model(torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255)[0].argmax(0).numpy().astype(np.uint8)
        result = metrics(confusion(target, prediction, len(SEGMENTATION_NAMES)), SEGMENTATION_NAMES)['classes'][case['category']]
        colors = np.zeros_like(rgb); valid = target != 255; colors[valid] = PALETTE[target[valid]]
        overlay = rgb.copy(); truth = target == case['category']; pred = prediction == case['category']
        overlay[truth] = [0, 255, 255]; overlay[pred] = [255, 0, 0]; overlay[truth & pred] = [255, 255, 0]
        panel = Image.new('RGB', (1920, 310)); draw = ImageDraw.Draw(panel)
        for i, (pixels, label) in enumerate([(rgb, 'RGB'), (colors, 'GT'), (PALETTE[prediction], 'CPU PREDICTION'), (overlay, 'FOCUS GT=CYAN PRED=RED BOTH=YELLOW')]):
            panel.paste(Image.fromarray(pixels).resize((480, 270), Image.Resampling.NEAREST if i in [1, 2] else Image.Resampling.BILINEAR), (480 * i, 40))
            draw.text((480 * i + 4, 22), label, fill='white')
        draw.text((4, 4), f"{name} | {row['scene_id']} {row['camera']} | focus IoU {result['iou']:.5f} | deliberately selected example only", fill='white')
        filename = name + '.png'; temp = output / (filename + '.tmp'); panel.save(temp, format='PNG'); temp.replace(output / filename)
        record = {'id': name, 'sample': row, 'category': case['category'], 'current_CPU_metrics': result,
                  'historical_Tiny_CPU_metrics': case['metrics'], 'panel': filename, 'panel_sha256': sha(output / filename)}
        report['cases'].append(record); write_json(report_path, report)
        print(json.dumps({'saved': name, 'current_IoU': result['iou'], 'historical_Tiny_IoU': case['metrics']['iou']}), flush=True)
    report['status'] = 'completed'; write_json(report_path, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); main(p.parse_args())
