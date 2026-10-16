"""Post-run engineering audit; never runs model selection or test inference."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from senior_ai.core import write_json
from senior_ai.segmentation_training import fingerprints, load_checkpoint, sha


def verify(run):
    read = lambda name: json.loads((run / name).read_text())
    receipt, config, history = read('report.json'), read('config.json'), read('history.json')
    root = ROOT / config['dataset']
    current = fingerprints(root, ROOT)
    acquisition = json.loads((root / 'report.json').read_text())
    source_configs = {'criteria': ROOT / 'configs/dataset_motion_criteria_v2.json',
                      'rig': ROOT / 'configs/camera_rig_v1.json'}
    checks = {'completed': receipt['status'] == 'completed',
              'all_images_and_metadata_unchanged': current['data_sha256'] == receipt['provenance']['data_sha256'],
              'numerical_source_unchanged': current['source_sha256'] == receipt['provenance']['source_sha256'],
              'config_bytes_unchanged': sha(run / 'config.json') == receipt['provenance']['config_sha256'],
              'acquisition_plan_bytes_match': sha(root / 'plan.json') == acquisition['plan_sha256']}
    for name, path in source_configs.items():
        checks[f'{name}_matches_frozen_source_bytes_and_copy_values'] = (
            sha(path) == acquisition[f'{name}_sha256'] and
            json.loads(path.read_text()) == json.loads((root / f'{name}.json').read_text()))
    # Dataset copies were serialized as JSON. Do not mistake whitespace differences
    # for changed criteria: retain original-byte hashes AND compare parsed copy values.
    best, last = load_checkpoint(run / 'best.pt'), load_checkpoint(run / 'last.pt')
    selected = max(history, key=lambda row: row['validation']['focus_macro_iou'])
    checks.update({'all_epochs_completed': [h['epoch'] for h in history] == list(range(1, config['epochs'] + 1)),
        'earliest_validation_best_selected': best['epoch'] == receipt['best_epoch'] == selected['epoch'],
        'last_checkpoint_complete': last['epoch'] == config['epochs'],
        'finite_training_metrics': all(math.isfinite(h['train_mean_batch_weighted_loss']) and
            math.isfinite(h['max_preclip_gradient_norm']) for h in history),
        'finite_model_weights': all(torch.isfinite(v).all().item() for v in best['model'].values()),
        'checkpoint_sha_matches_test_marker': sha(run / 'best.pt') == receipt['checkpoint_sha256'] == read('test_started.json')['checkpoint_sha256'],
        'checkpoint_provenance_matches_receipt': best['provenance'] == last['provenance'] == receipt['provenance'],
        'checkpoint_config_matches': best['config'] == last['config'] == config,
        'no_posthoc_quality_pass': receipt['quality_acceptance'] == 'not_defined'})
    stats = read('train_validation_statistics.json')
    groups = [set(stats[s]['independent_groups']) for s in ['train', 'validation']]
    groups.append(set(read('test_statistics.json')['independent_groups']))
    checks['disjoint_scene_groups'] = all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    output = {'task': 'post-training-engineering-audit', 'status': 'passed' if all(checks.values()) else 'failed',
        'checks': checks, 'verified_images_again': current['verified_images'],
        'optimizer_updates_last': sorted({int(s['step']) for s in last['optimizer']['state'].values()}),
        'amp_backoffs': sum(len(h['amp_overflow_backoffs']) for h in history),
        'note': 'Engineering/integrity audit only; does not imply perception quality or full Phase 3 acceptance.'}
    write_json(run / 'engineering_audit.json', output)
    if output['status'] != 'passed':
        raise ValueError(str(output))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    print(json.dumps(verify(parser.parse_args().run.resolve())))
