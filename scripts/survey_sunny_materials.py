"""Resumable, model-free train/validation appearance survey of raw semantic GT.

Samples four fixed bundle positions per scene and all eight cameras, without
opening development/lockbox indexes. RGB descriptors are appearance proxies,
not automatic material labels. Each view is journaled for interruption recovery.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

RAW_CLASSES = {'wall': [4], 'fence': [5], 'terrain_or_ground': [10, 25]}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    # Windows readers/antivirus may briefly hold a handle without delete sharing.
    # Retain the last complete JSON and fail after a bounded retry interval.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.025 * 2 ** attempt, .25))


def safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes dataset')
    return path


def main(args):
    root, output = args.root.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Output must be outside dataset')
    audit = read(root / 'audit.json'); review = read(root / 'visual_review.json')
    if audit['status'] != 'passed' or not review['training_approved'] or review['audit_sha256'] != sha(root / 'audit.json'):
        raise ValueError('Bound dataset approval required')
    for filename in ['image_inventory.jsonl', 'metadata_inventory.json']:
        key = filename.split('.')[0] + '_sha256'
        if sha(root / filename) != audit[key]:
            raise ValueError('Dataset inventory changed')
    metadata = {r['path']: r['sha256'] for r in read(root / 'metadata_inventory.json')}
    selected = []
    for split in ['train', 'validation']:
        name = f'training/{split}.jsonl'
        if sha(root / name) != metadata[name]:
            raise ValueError('Index changed')
        groups = defaultdict(list)
        for line in (root / name).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['split'] != split:
                raise ValueError('Split mismatch')
            groups[row['scene_id']].append(row)
        for scene in sorted(groups):
            frames = sorted({r['frame'] for r in groups[scene]})
            positions = list(range(len(frames))) if split == 'train' and args.training_all else [0, 40, 80, 120]
            if len(frames) <= max(positions):
                raise ValueError('Expected full 150-bundle scenes')
            chosen = {frames[i] for i in positions}
            selected.extend(sorted((r for r in groups[scene] if r['frame'] in chosen), key=lambda r: (r['frame'], r['camera'])))
    needed = {row[key] for row in selected for key in ['rgb', 'semantic']}
    inventory = {}
    with (root / 'image_inventory.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item['path'] in needed:
                inventory[item['path']] = item
    if set(inventory) != needed:
        raise ValueError('Selected images missing from inventory')
    identity = {'tool_sha256': sha(Path(__file__)), 'audit_sha256': sha(root / 'audit.json'),
                'image_inventory_sha256': sha(root / 'image_inventory.jsonl'),
                'selection': 'all train bundles; validation positions0/40/80/120; all cameras' if args.training_all else 'bundle positions0/40/80/120 per train/validation scene, all cameras',
                'selected_views': len(selected)}
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / 'progress.json'
    if progress_path.exists() and read(progress_path)['identity'] != identity:
        raise ValueError('Resume identity differs')
    # Each view is a separate atomic JSON, avoiding an incomplete trailing JSONL.
    records_dir = output / 'views'; records_dir.mkdir(exist_ok=True)
    records = []
    for i, row in enumerate(selected):
        hashes = {}
        for key in ['rgb', 'semantic']:
            path = safe(root, row[key]); item = inventory[row[key]]
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('Selected raw bytes changed')
            hashes[key] = item['sha256']
        path = records_dir / f'{i:04d}.json'
        if path.exists():
            record = read(path)
            if record['sample'] != row or record['raw_sha256'] != hashes:
                raise ValueError('Completed survey record differs')
        else:
            with Image.open(safe(root, row['rgb'])) as image:
                rgb = np.array(image.convert('RGB'))
            with Image.open(safe(root, row['semantic'])) as image:
                raw = np.array(image.convert('RGB'))[..., 0]
            descriptions = {}
            for name, ids in RAW_CLASSES.items():
                mask = np.isin(raw, ids); count = int(mask.sum())
                item = {'pixels': count}
                if count >= 512:
                    colors = rgb[mask].astype(np.float32); y, x = np.where(mask)
                    item.update(mean_rgb=colors.mean(0).tolist(), std_rgb=colors.std(0).tolist(),
                                bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
                                warm_color_fraction=float(((colors[:, 0] > colors[:, 1] * 1.15) & (colors[:, 0] > colors[:, 2] * 1.3)).mean()),
                                bright_neutral_fraction=float(((colors.max(1) - colors.min(1) < 25) & (colors.mean(1) > 160)).mean()))
                descriptions[name] = item
            record = {'sample': row, 'raw_sha256': hashes, 'classes': descriptions}
            write(path, record)
        records.append(record)
        write(progress_path, {'identity': identity, 'status': 'scanning', 'completed_views': len(records)})
        if (i + 1) % 32 == 0:
            print(json.dumps({'surveyed_views': i + 1, 'total': len(selected)}), flush=True)
    # Support/warm/bright-neutral contexts per group. Color selection deliberately
    # does not assert wood/concrete, and identical views are not duplicated.
    gallery = []
    for group in sorted({r['sample']['group_id'] for r in records}):
        candidates = [r for r in records if r['sample']['group_id'] == group and r['classes']['fence']['pixels'] >= 512]
        if candidates:
            gallery.append(('max_fence_support', max(candidates, key=lambda r: r['classes']['fence']['pixels'])))
            if args.training_all:
                for key in ['warm_color_fraction', 'bright_neutral_fraction']:
                    chosen = max(candidates, key=lambda r: r['classes']['fence'][key])
                    if not any(r['sample']['rgb'] == chosen['sample']['rgb'] for _, r in gallery):
                        gallery.append(('max_' + key, chosen))
    for split in ['train', 'validation']:
        candidates = [r for r in records if r['sample']['split'] == split and r['classes']['fence']['pixels'] >= 512]
        if candidates:
            chosen = max(candidates, key=lambda r: r['classes']['fence']['warm_color_fraction'])
            if not any(r['sample']['rgb'] == chosen['sample']['rgb'] for _, r in gallery):
                gallery.append(('max_warm_color_fraction', chosen))
    panels = []
    for reason, record in gallery:
        row = record['sample']; descriptor = record['classes']['fence']; box = descriptor['bbox_xyxy']
        with Image.open(safe(root, row['rgb'])) as im:
            rgb = im.convert('RGB'); context = rgb.copy(); ImageDraw.Draw(context).rectangle(box, outline='cyan', width=2)
            crop = rgb.crop((max(0, box[0] - 8), max(0, box[1] - 8), min(640, box[2] + 8), min(360, box[3] + 8)))
        panel = Image.new('RGB', (640, 220)); panel.paste(context.resize((320, 180)), (0, 40))
        crop.thumbnail((320, 180)); panel.paste(crop, (320, 40)); draw = ImageDraw.Draw(panel)
        draw.text((4, 2), f"{row['split']} {row['scene_id']} {row['camera']} | raw5 Fence {descriptor['pixels']}px", fill='white')
        draw.text((4, 20), reason + ' | appearance proxy, not material annotation', fill='white')
        panels.append((panel, {'selection': reason, **record}))
    pages = []
    for start in range(0, len(panels), 8):
        subset = panels[start:start + 8]; page = Image.new('RGB', (1280, 220 * ((len(subset) + 1) // 2)))
        for i, (panel, _) in enumerate(subset):
            page.paste(panel, ((i % 2) * 640, (i // 2) * 220))
        name = f'page-{start // 8:02d}.png'; temporary = output / (name + '.tmp'); page.save(temporary, format='PNG'); temporary.replace(output / name)
        pages.append({'path': name, 'sha256': sha(output / name)})
    summary = {'identity': identity, 'status': 'completed', 'completed_views': len(records), 'pages': pages,
               'io_recovery': read(output / 'io-recovery.json') if (output / 'io-recovery.json').exists() else None,
               'scope': 'Systematic train/validation RGB+raw semantic support survey; selection mode defines coverage. Not model metrics or automatic material labels; no development/lockbox index access.',
               'gallery': [r for _, r in panels], 'class_support_in_survey': {
                   split: {name: {'pixels': sum(r['classes'][name]['pixels'] for r in records if r['sample']['split'] == split),
                                  'positive_views_512px': sum(r['classes'][name]['pixels'] >= 512 for r in records if r['sample']['split'] == split)}
                           for name in RAW_CLASSES} for split in ['train', 'validation']}}
    write(output / 'summary.json', summary); write(progress_path, {'identity': identity, 'status': 'completed', 'completed_views': len(records)})
    print(json.dumps({'status': 'completed', 'surveyed_views': len(records), 'fence_gallery_cases': len(panels)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-all', action='store_true', help='All train views, with validation kept at fixed sparse positions')
    main(p.parse_args())
