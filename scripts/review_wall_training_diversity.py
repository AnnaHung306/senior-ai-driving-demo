"""Train-only wall appearance survey; descriptors select QA, never new labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    if args.output.exists():
        raise FileExistsError('Use a new survey version')
    root = args.root.resolve()
    survey = json.loads((args.survey / 'report.json').read_text(encoding='utf-8'))
    if sha(root / 'audit.json') != survey['main_audit_sha256'] or sha(args.survey / 'records.jsonl.gz') != survey['records_archive_sha256']:
        raise ValueError('Bound original training survey required')
    with gzip.open(args.survey / 'records.jsonl.gz', 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    # Select one frame per camera/group/second BEFORE appearance descriptors.
    candidates = []; seen = set()
    for row in rows:
        s = row['sample']
        if s['split'] != 'train' or s['scene_id'].split('-')[0] not in {'town01', 'town03'}:
            raise ValueError('Original train boundary only')
        key = (s['group_id'], s['camera'], s['frame'] // 20)
        if row['wall_pixels'] >= 512 and key not in seen:
            seen.add(key); candidates.append(row)
    required = {r['sample'][k] for r in candidates for k in ['rgb', 'semantic']}
    inventory = {r['path']: r for r in map(json.loads, (root / 'image_inventory.jsonl').read_text().splitlines()) if r['path'] in required}

    def checked(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha(path) != inventory[relative]['sha256']:
            raise ValueError('Audited image changed')
        with Image.open(path) as image:
            return np.array(image.convert('RGB'))

    def describe(row):
        s = row['sample']; rgb = checked(s['rgb']); mask = checked(s['semantic'])[..., 0] == 4
        pixels = rgb[mask].astype(np.float32)
        # Photometric QA strata only: bright neutral, dark neutral, warm colour,
        # other colour. Do not call any stratum concrete/wood/material GT.
        neutral = float(np.mean(pixels.max(1) - pixels.min(1) <= 30))
        brightness = float(pixels.mean())
        warm = float(np.mean((pixels[:, 0] > pixels[:, 1] * 1.15) & (pixels[:, 0] > pixels[:, 2] * 1.15)))
        kind = 'bright-neutral' if neutral >= .65 and brightness >= 100 else 'dark-neutral' if neutral >= .65 else 'warm-colour' if warm >= .5 else 'other-colour'
        return {**row, 'appearance_hint': kind, 'neutral_fraction': neutral, 'mean_brightness': brightness, 'warm_fraction': warm}

    with ThreadPoolExecutor(max_workers=4) as pool:
        described = list(pool.map(describe, candidates))
    selected = []; counts = {}
    for row in described:
        k = row['sample']['group_id'] + '/' + row['appearance_hint']
        counts[k] = counts.get(k, 0) + 1
    for group in sorted({r['sample']['group_id'] for r in described}):
        for kind in ['bright-neutral', 'dark-neutral', 'warm-colour', 'other-colour']:
            subset = [r for r in described if r['sample']['group_id'] == group and r['appearance_hint'] == kind]
            if subset:
                selected.append(max(subset, key=lambda r: (r['wall_pixels'], -r['index'])))
    args.output.mkdir(parents=True); pages = []
    for start in range(0, len(selected), 3):
        subset = selected[start:start+3]; canvas = Image.new('RGB', (1280, 384 * len(subset)))
        for j, row in enumerate(subset):
            s = row['sample']; rgb = checked(s['rgb']); mask = checked(s['semantic'])[..., 0] == 4
            overlay = rgb.copy(); overlay[mask] = [255, 0, 255]
            canvas.paste(Image.fromarray(rgb), (0, j*384+24)); canvas.paste(Image.fromarray(overlay), (640, j*384+24))
            ImageDraw.Draw(canvas).text((4, j*384+4), f"train index{row['index']} {s['group_id']}/{s['camera']} {row['appearance_hint']} WALL=magenta", fill='white')
        path = args.output / f'page-{start//3}.png'; canvas.save(path)
        pages.append({'path': path.name, 'sha256': sha(path)})
    report = {'status': 'completed_train_diagnostic_only', 'training_approved': False, 'main_audit_sha256': survey['main_audit_sha256'],
              'survey_records_sha256': survey['records_archive_sha256'], 'tool_sha256': sha(Path(__file__)), 'candidates': len(described),
              'counts_by_group_and_appearance_hint': counts, 'selected': selected, 'pages': pages,
              'limits': 'Colour strata are not material labels. Temporal candidates are correlated. No new examples, labels, independent groups, inference scores, or acceptance gates.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'candidates': len(described), 'selected': len(selected), 'pages': len(pages), 'counts': counts}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--survey', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
