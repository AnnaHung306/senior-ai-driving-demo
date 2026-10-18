"""Resumable validation error panels from a completed sunny model.

Selects one worst supported wall/fence and visible-margin case per validation
scene using the original saved GPU metrics. CPU float32 renders are diagnostic;
they do not replace the frozen GPU validation report or open heldout data.
Optional CUDA replay uses original evaluation batch boundaries and requires
selected confusion matrices and boundary counts to match the saved report.
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
from senior_ai.road_boundaries import visible_margin, boundary_counts, boundary_metrics
from senior_ai.segmentation_metrics import confusion, metrics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def selected_cases(frames, all_classes=False):
    if any(row['split'] != 'validation' for row in frames):
        raise ValueError('Only validation error selection is allowed')
    cases = []
    for scene in sorted({r['scene_id'] for r in frames}):
        scene_frames = [r for r in frames if r['scene_id'] == scene]
        for name in ['wall', 'fence']:
            i = SEGMENTATION_NAMES.index(name)
            supported = [r for r in scene_frames if r['classes'][i]['support_pixels'] >= 512]
            if supported:
                row = min(supported, key=lambda r: (r['classes'][i]['iou'], -r['classes'][i]['support_pixels'], r['index']))
                cases.append({'id': scene + '-' + name, 'reason': name, 'sample': row})
        supported = [r for r in scene_frames if r['visible_margin']['gt_pixels'] >= 50]
        if supported:
            row = min(supported, key=lambda r: (r['visible_margin']['f1'] or 0, -r['visible_margin']['gt_pixels'], r['index']))
            cases.append({'id': scene + '-margin', 'reason': 'visible_margin', 'sample': row})
    if all_classes:
        indices={c['sample']['index'] for c in cases}
        for i,name in enumerate(SEGMENTATION_NAMES[1:],1):
            supported=[r for r in frames if r['classes'][i]['support_pixels']>=16]
            if supported:
                row=min(supported,key=lambda r:(r['classes'][i]['iou'],-r['classes'][i]['support_pixels'],r['index']))
                if row['index'] not in indices:
                    cases.append({'id':'class-'+name,'reason':'worst_supported_class/'+name,'sample':row});indices.add(row['index'])
        for scene in sorted({r['scene_id'] for r in frames}):
            def wall_fp(row):
                m=row['confusion_true_rows_predicted_columns'];return sum(m[i][12] for i in range(len(m)) if i!=12)
            row=max((r for r in frames if r['scene_id']==scene),key=lambda r:(wall_fp(r),-r['index']))
            if wall_fp(row)>0 and row['index'] not in indices:
                cases.append({'id':scene+'-wall-false-positive','reason':'most_Wall_false_positive_pixels','sample':row});indices.add(row['index'])
    return cases


def batch_indices(index,total,size):
    if type(index)is not int or not 0<=index<total or type(size)is not int or size<1:raise ValueError('Valid original batch index/size required')
    start=index//size*size;return list(range(start,min(start+size,total)))


def supervised_margins(target,prediction):
    valid=target!=255;supervised=valid.copy()
    supervised[1:] &= valid[:-1];supervised[:-1] &= valid[1:]
    supervised[:,1:] &= valid[:,:-1];supervised[:,:-1] &= valid[:,1:]
    return visible_margin(target)&supervised,visible_margin(prediction)&supervised


def main(args):
    run, root, output = args.run.resolve(), args.root.resolve(), args.output.resolve()
    if output.is_relative_to(root) or output.is_relative_to(run):
        raise ValueError('Write diagnostic outside dataset and frozen run')
    receipt = check(run)
    provenance = read(run / 'provenance.json')
    signature = provenance['data']
    for filename, key in [('audit.json', 'audit_sha256'), ('visual_review.json', 'review_sha256'),
                          ('image_inventory.jsonl', 'image_inventory_sha256'), ('metadata_inventory.json', 'metadata_inventory_sha256')]:
        if sha(root / filename) != signature[key]:
            raise ValueError('Dataset inventory changed: ' + filename)
    # Verify exact runtime dependency bytes, not just the recorded Git revision.
    for relative, digest in provenance['source']['files'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Diagnostic runtime differs from frozen source: ' + relative)
    metadata = {r['path']: r['sha256'] for r in read(root / 'metadata_inventory.json')}
    if sha(root / 'training/validation.jsonl') != metadata['training/validation.jsonl']:
        raise ValueError('Validation index changed')
    canonical = [json.loads(line) for line in (root / 'training/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    frames = [json.loads(line) for line in (run / 'validation_frames.jsonl').read_text(encoding='utf-8').splitlines()]
    cases = selected_cases(frames,args.all_classes)
    batch_size=read(run/'config.json')['segmentation']['batch_size'] if args.device=='cuda' else 1
    required = {case['sample'][key] for case in cases for key in ['rgb', 'semantic', 'instance']}
    if args.device=='cuda':
        for case in cases:
            required.update(canonical[i]['rgb'] for i in batch_indices(case['sample']['index'],len(canonical),batch_size))
    inventory = {}
    with (root / 'image_inventory.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item['path'] in required:
                path = safe_path(root, item['path'])
                if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
                    raise ValueError('Selected raw image changed: ' + item['path'])
                inventory[item['path']] = item['sha256']
    if set(inventory) != required:
        raise ValueError('Missing selected raw inventory')
    for case in cases:
        row = case['sample']
        if any(row[k] != v for k, v in canonical[row['index']].items()):
            raise ValueError('Report sample differs from canonical validation index')

    identity = {'checkpoint_sha256': receipt['hashes']['best.pt'], 'tool_sha256': sha(Path(__file__)),
                'source_manifest_sha256': provenance['source']['sha256'],
                'validation_frames_sha256': receipt['hashes']['validation_frames.jsonl'],
                'selected_raw_sha256': inventory, 'case_ids': [c['id'] for c in cases],
                'render_device':args.device,'original_evaluation_batch_size':batch_size}
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'catalog.json'
    report = read(report_path) if report_path.exists() else {
        'identity': identity, 'status': 'in_progress', 'cases': [],
        'scope': ('Validation-only original-batch CUDA replay; selected confusion matrices and boundary counts must match saved GPU reports.' if args.device=='cuda' else 'Validation-only CPU float32 diagnostic; original GPU BF16 metrics retained separately.')+' Not exhaustive visual review or new acceptance scoring.',
        'selection': 'Per validation scene: lowest original IoU with >=512 GT wall/fence pixels; lowest margin F1 with >=50 GT margin pixels; ties by support then index.',
        'raw_verification': 'Selected target RGB/semantic/instance and every replay-batch RGB byte, plus bound inventory/index signatures; not a full raw-data re-audit.'}
    if args.all_classes:report['selection']='Per scene worst supported Wall/Fence and margin; also worst >=16px supported case per focus class and maximum Wall false-positive pixels per scene, additional cases deduplicated by image.'
    if report['identity'] != identity:
        raise ValueError('Resume identity changed; use a new output directory')
    completed = {case['id']: case for case in report['cases']}
    for case in completed.values():
        if sha(output / case['panel']) != case['panel_sha256']:
            raise ValueError('Completed panel changed')
    write_json(report_path, report)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.device=='cuda':
        from train_sunny import require_idle_gpu
        require_idle_gpu()
    model, task, _ = load_model(run / 'best.pt', args.device)
    if task != 'segmentation':
        raise ValueError('Segmentation model required')
    replay_batches={}
    for case in cases:
        if case['id'] in completed:
            continue
        row = case['sample']
        rgb = png(safe_path(root, row['rgb']))
        target = segmentation_target(png(safe_path(root, row['semantic']))[..., 0],
                                     png(safe_path(root, row['instance'])), row['verified_obstacle_actor_ids'])
        if args.device=='cuda':
            indices=batch_indices(row['index'],len(canonical),batch_size);start=indices[0]
            if start not in replay_batches:
                tensors=[torch.from_numpy(png(safe_path(root,canonical[i]['rgb'])).transpose(2,0,1).copy()) for i in indices]
                tensor=torch.stack(tensors).cuda().float().div_(255)
                with torch.inference_mode():replay_batches[start]=model(tensor).argmax(1).cpu().numpy().astype(np.uint8)
            prediction=replay_batches[start][row['index']-start]
        else:
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255
            with torch.inference_mode():prediction = model(tensor)[0].argmax(0).numpy().astype(np.uint8)
        computed=metrics(confusion(target,prediction,len(SEGMENTATION_NAMES)),SEGMENTATION_NAMES)
        computed_margin=boundary_metrics(boundary_counts(target,prediction))
        if args.device=='cuda' and (computed['confusion_true_rows_predicted_columns']!=row['confusion_true_rows_predicted_columns'] or computed_margin!=row['visible_margin']):
            raise ValueError('CUDA replay differs from saved original-batch confusion/margin: '+case['id'])
        gt_color = np.zeros_like(rgb); valid = target != 255; gt_color[valid] = PALETTE[target[valid]]
        margins = rgb.copy(); truth, predicted = supervised_margins(target,prediction)
        margins[truth] = [0, 255, 255]; margins[predicted] = [255, 0, 0]; margins[truth & predicted] = [255, 255, 0]
        panel = Image.new('RGB', (1920, 310))
        draw = ImageDraw.Draw(panel)
        for i, (pixels, label) in enumerate([(rgb, 'RGB'), (gt_color, 'GROUND TRUTH'),
                                            (PALETTE[prediction], 'CUDA ORIGINAL-BATCH PREDICTION' if args.device=='cuda' else 'CPU FLOAT32 PREDICTION'), (margins, 'MARGIN GT=CYAN PRED=RED BOTH=YELLOW')]):
            panel.paste(Image.fromarray(pixels).resize((480, 270), Image.Resampling.NEAREST if i in [1, 2] else Image.Resampling.BILINEAR), (480 * i, 40))
            draw.text((480 * i + 4, 22), label, fill='white')
        draw.text((4, 4), f"{case['id']} | {row['camera']} frame {row['frame']} | {args.device} replay, GPU metrics frozen", fill='white')
        filename = case['id'] + '.png'; temporary = output / (filename + '.tmp')
        panel.save(temporary, format='PNG'); temporary.replace(output / filename)
        record = {'id': case['id'], 'reason': case['reason'], 'sample_index': row['index'],
                  'scene_id': row['scene_id'], 'group_id': row['group_id'], 'camera': row['camera'], 'frame': row['frame'],
                  'RGB': row['rgb'], 'original_GPU_classes': row['classes'], 'original_GPU_visible_margin': row['visible_margin'],
                  ('CUDA_replay_classes' if args.device=='cuda' else 'CPU_diagnostic_classes'): computed['classes'],
                  'render_device':args.device,'CUDA_confusion_and_margin_exact':True if args.device=='cuda' else None,
                  'panel': filename, 'panel_sha256': sha(output / filename)}
        report['cases'].append(record); write_json(report_path, report)
        print(json.dumps({'saved': case['id'], 'completed': len(report['cases']), 'total': len(cases)}), flush=True)
    # Rebuild contact sheets from durable panels, so interruption is recoverable.
    pages = []
    for start in range(0, len(report['cases']), 3):
        subset = report['cases'][start:start + 3]
        page = Image.new('RGB', (1920, 310 * len(subset)))
        for i, case in enumerate(subset):
            with Image.open(output / case['panel']) as panel:
                page.paste(panel, (0, 310 * i))
        filename = f'page-{start // 3:02d}.png'; temporary = output / (filename + '.tmp')
        page.save(temporary, format='PNG'); temporary.replace(output / filename)
        pages.append({'path': filename, 'sha256': sha(output / filename)})
    report.update(status='completed', pages=pages); write_json(report_path, report)
    (output/'tool_used.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--all-classes',action='store_true')
    main(p.parse_args())
