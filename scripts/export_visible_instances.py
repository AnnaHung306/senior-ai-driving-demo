"""Apply a frozen policy to controlled engineering data, never approve training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json
from senior_ai.dataset import safe_path
from senior_ai.instance_labels import decode_instance_rgb
from senior_ai.visibility_policy import export_view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--policy', type=Path, default=ROOT / 'configs/visibility_policy_v1.json')
    parser.add_argument('--audit', default='audit-v2.json')
    args = parser.parse_args()
    root = args.root.resolve()
    audit_path = safe_path(root, args.audit)
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'passed':
        raise ValueError('Requires independent raw probe audit')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(root / 'image_inventory.jsonl') != audit['inventory_sha256']:
        raise ValueError('Inventory changed after audit')
    for name, expected in audit['metadata_sha256'].items():
        if digest(safe_path(root, name)) != expected:
            raise ValueError('Metadata changed after audit')
    inventory = {i['path']: i['sha256'] for i in [json.loads(s) for s in (root / 'image_inventory.jsonl').read_text().splitlines()]}
    destination = root / 'visible_instances'; destination.mkdir(exist_ok=False)
    policy = json.loads(args.policy.read_text())
    (destination / 'policy.json').write_bytes(args.policy.read_bytes())
    index = {x['case']: x for x in json.loads((root / 'index.json').read_text())}
    exported, positive, ignored, invisible, ignore_pixels = [], 0, 0, 0, 0
    reasons = Counter()
    for case in audit['cases']:
        label = json.loads(safe_path(root, index[case['case']]['label']).read_text())
        for name, view in case['views'].items():
            image = safe_path(root, label['cameras'][name]['modalities']['instance_segmentation']['image'])
            if digest(image) != inventory[image.relative_to(root).as_posix()]:
                raise ValueError('Instance image changed after audit')
            with Image.open(image) as png:
                classes, ids = decode_instance_rgb(np.array(png))
            result, mask = export_view(view['candidates'], classes, ids, policy)
            mask_name = f'{case["case"]}-{name}-ignore.png'
            Image.fromarray(mask).save(destination / mask_name)
            exported.append({**result, 'case': case['case'], 'camera': name, 'frame': case['frame'],
                'input_manifest': index[case['case']]['input'], 'ignore_mask': mask_name})
            positive += len(result['positive_objects']); ignored += len(result['ignored_objects'])
            invisible += result['not_visible_objects']; ignore_pixels += result['ignore_pixels']
            reasons.update(reason for obj in result['ignored_objects'] for reason in obj['reasons'])
    (destination / 'views.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in exported))
    summary = {'task': 'controlled-visible-instance-policy-export', 'status': 'passed', 'training_approved': False,
        'why_not_training': 'Controlled static engineering fixture; no natural episodes/group splits/coverage or ignore-aware detector integration',
        'views': len(exported), 'positive_object_views': positive, 'ignored_object_views': ignored,
        'not_visible_object_views': invisible, 'ignore_pixels': ignore_pixels, 'ignore_reasons': dict(reasons),
        'policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        'raw_audit_sha256': digest(audit_path)}
    write_json(destination / 'report.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
