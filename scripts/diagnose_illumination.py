"""No-learning paired RGB diagnostic of a frozen checkpoint, with GT only for scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from PIL import Image, ImageDraw
import torch
from senior_ai.core import write_json
from senior_ai.dataset import safe_path
from senior_ai.segmentation_data import load_inference_rgb
from senior_ai.segmentation_metrics import confusion, metrics
from senior_ai.segmentation_model import TinyUNet, colorize, map_target
from senior_ai.segmentation_training import configure_cuda, load_checkpoint, sha


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/illumination_diagnostic_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    roots = {key: safe_path(ROOT, value) for key, value in config['pairs'].items()}
    indexes, labels, audits = {}, {}, {}
    for key, root in roots.items():
        audits[key] = json.loads((root / config['audit']).read_text())
        if audits[key]['status'] != 'passed':
            raise ValueError('Raw data audit failed')
        for name, expected in audits[key]['metadata_sha256'].items():
            if sha(safe_path(root, name)) != expected:
                raise ValueError('Metadata changed after audit')
        inventory_path = root / 'image_inventory.jsonl'
        if sha(inventory_path) != audits[key]['inventory_sha256']:
            raise ValueError('Inventory changed')
        for entry in [json.loads(line) for line in inventory_path.read_text().splitlines()]:
            if sha(safe_path(root, entry['path'])) != entry['sha256']:
                raise ValueError('Image changed after audit')
        indexes[key] = {e['case']: e for e in json.loads((root / 'index.json').read_text())}
        labels[key] = {name: json.loads(safe_path(root, e['label']).read_text()) for name, e in indexes[key].items()}
    if set(indexes['noon']) != set(indexes['night']):
        raise ValueError('Cases differ')
    pose_error, mismatch, total = 0., 0, 0
    target_agreement = np.zeros((8, 8), np.int64)
    checkpoint = safe_path(ROOT, config['checkpoint'])
    if sha(checkpoint) != config['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    state = load_checkpoint(checkpoint); model_config = state['config']
    for case in indexes['noon']:
        a, b = labels['noon'][case], labels['night'][case]
        roles = lambda frame: {obj['role']: obj for obj in frame['objects']}
        aa, bb = roles(a), roles(b)
        if aa.keys() != bb.keys() or a['cameras'].keys() != b['cameras'].keys():
            raise ValueError('Actor/camera identities differ')
        for role in aa:
            if aa[role]['blueprint'] != bb[role]['blueprint']:
                raise ValueError('Actor asset differs')
            pose_error = max(pose_error, float(np.max(np.abs(np.array(aa[role]['T_world_from_box']) - bb[role]['T_world_from_box']))))
        for camera in a['cameras']:
            ca, cb = a['cameras'][camera], b['cameras'][camera]
            pose_error = max(pose_error, float(np.max(np.abs(np.array(ca['T_world_from_camera']) - cb['T_world_from_camera']))))
            if ca['K'] != cb['K']:
                raise ValueError('Camera intrinsics differ')
            target_a = read_rgb(safe_path(roots['noon'], ca['modalities']['semantic_segmentation']['image']))[..., 0]
            target_b = read_rgb(safe_path(roots['night'], cb['modalities']['semantic_segmentation']['image']))[..., 0]
            mismatch += int((target_a != target_b).sum()); total += target_a.size
            ta, tb = map_target(target_a, model_config), map_target(target_b, model_config)
            valid = (ta != 255) & (tb != 255)
            target_agreement += confusion(ta[valid], tb[valid])
    if pose_error > config['max_actor_pose_error'] or mismatch / total > config['max_global_semantic_target_mismatch_fraction']:
        raise ValueError('Paired geometry/target comparability gate failed')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_bytes(args.config.read_bytes())
    configure_cuda()
    model = TinyUNet(model_config['width']).cuda().eval(); model.load_state_dict(state['model'])
    output, predictions, frames = {}, {}, []
    for key, root in roots.items():
        matrix = np.zeros((8, 8), np.int64); predictions[key] = {}
        rgb_stats = []
        for case, entry in indexes[key].items():
            label = labels[key][case]
            for camera in label['cameras']:
                # RGB API alone supplies the model. GT opened separately below for scoring.
                rgb = load_inference_rgb(root, entry['input'], camera)
                x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).cuda().float() / 255
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    logits = model(x)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite predictions')
                pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                target = map_target(read_rgb(safe_path(root, label['cameras'][camera]['modalities']['semantic_segmentation']['image']))[..., 0], model_config)
                cm = confusion(target, pred); matrix += cm
                detail = metrics(cm, model_config['class_names'])
                frames.append({'weather_key': key, 'case': case, 'camera': camera, 'frame': label['frame'], **detail})
                rgb_stats.append({'case': case, 'camera': camera, 'mean_rgb_0_255': float(rgb.mean()),
                                  'near_black_pixel_fraction': float((rgb.max(2) < 5).mean())})
                if camera == 'front':
                    predictions[key][case] = (rgb, target, pred)
        output[key] = {**metrics(matrix, model_config['class_names']), 'rgb_profile': rgb_stats,
                       'raw_audit_sha256': sha(root / config['audit']), 'acquisition_report_sha256': sha(root / 'report.json')}
    panels = []
    for case in indexes['noon']:
        panel = Image.new('RGB', (640 * 3, 390 * 2), 'white')
        for row, key in enumerate(['noon', 'night']):
            rgb, target, pred = predictions[key][case]
            for col, (array, title) in enumerate([(rgb, 'RGB'), (colorize(target), 'GT: evaluation only'), (colorize(pred), 'Frozen learned prediction')]):
                panel.paste(Image.fromarray(array), (col * 640, row * 390 + 30))
                ImageDraw.Draw(panel).text((col * 640 + 5, row * 390 + 6), f'{case} / {key} / {title}', fill='black')
        name = case + '.png'; panel.save(args.output / name); panels.append(name)
    (args.output / 'frames.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
    receipt = {'task': 'frozen-model-paired-illumination-development-diagnostic', 'status': 'completed',
        'formal_acceptance': False, 'optimizer_updates': 0, 'checkpoint_sha256': sha(checkpoint),
        'checkpoint_unchanged': sha(checkpoint) == config['checkpoint_sha256'],
        'pose_error_max': pose_error, 'raw_semantic_target_mismatch_fraction': mismatch / total,
        'mapped_target_agreement': metrics(target_agreement, model_config['class_names']),
        'views_per_condition': len(frames) // 2, 'results': output, 'panels': panels,
        'source_sha256': sha(Path(__file__)), 'model_source_sha256': sha(ROOT / 'src/senior_ai/segmentation_model.py'),
        'config_sha256': sha(args.config), 'limits': config['limits'],
        'interpretation': 'Matched actor and camera geometry, fixed existing weights. Two full weather presets on a prior training road; not a new generalization score or rain-only causal effect.'}
    write_json(args.output / 'report.json', receipt)
    print(json.dumps({key: value['focus_macro_iou'] for key, value in output.items()}))


if __name__ == '__main__':
    main()
