"""Package exact receipts and real QA images for the Phase 3 annotation iteration."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json
from senior_ai.segmentation_training import sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); target = args.output.resolve()
    roots = {'controlled': 'instance-probe-v1-001', 'matched_noon': 'instance-weather-noon-v1-001',
             'matched_night': 'instance-weather-night-v1-001', 'moving': 'motion-instance-probe-v1-001'}
    checks = {}
    for key, name in roots.items():
        root = ROOT / 'data' / name
        report = json.loads((root / 'report.json').read_text())
        audit_name = 'audit-instance-motion.json' if key == 'moving' else 'audit-v3.json'
        audit = json.loads((root / audit_name).read_text())
        if report['status'] != 'passed' or audit['status'] != 'passed' or audit['training_approved'] is not False:
            raise ValueError('Only passed non-training engineering probes are packaged')
        inventory = [json.loads(line) for line in (root / 'image_inventory.jsonl').read_text().splitlines()]
        for item in inventory:
            if sha(root / item['path']) != item['sha256']:
                raise ValueError('Raw image changed')
        checks[key] = {'data_path': root.relative_to(ROOT).as_posix(), 'report_sha256': sha(root / 'report.json'),
            'audit_sha256': sha(root / audit_name), 'images': len(inventory)}
    target.mkdir(parents=True, exist_ok=False)
    for key, name in roots.items():
        root = ROOT / 'data' / name; folder = target / key; folder.mkdir()
        audit_name = 'audit-instance-motion.json' if key == 'moving' else 'audit-v3.json'
        names = ['report.json', audit_name, 'image_inventory.jsonl', 'rig.json']
        names += ['criteria.json'] if key == 'moving' else ['config.json', 'index.json']
        if key != 'moving':
            names += ['visible_instances_v1_1/report.json', 'visible_instances_v1_1/policy.json']
        for path in names:
            destination = folder / path; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / path, destination)
    controlled = ROOT / 'data' / roots['controlled']
    for name in ['visual_review.json', 'qa-front-pairs.png', 'qa-other-visible.png']:
        shutil.copyfile(controlled / name, target / 'controlled' / name)
    shutil.copytree(ROOT / 'data' / roots['moving'] / 'qa-motion', target / 'moving/qa-motion')
    diagnostic = ROOT / 'artifacts/runs/illumination-diagnostic-v1-001'
    d = json.loads((diagnostic / 'report.json').read_text())
    if d['optimizer_updates'] != 0 or not d['checkpoint_unchanged'] or d['formal_acceptance'] is not False:
        raise ValueError('Diagnostic is not a frozen-model development-only experiment')
    folder = target / 'illumination'; folder.mkdir()
    for name in ['report.json', 'config.json', 'revealed.png']:
        shutil.copyfile(diagnostic / name, folder / name)
    for stem in ['phase0-before-instance-008', 'phase0-before-moving-instance-009']:
        shutil.copyfile(ROOT / 'artifacts/runs' / stem / 'report.json', target / (stem + '.json'))
    shutil.copyfile(ROOT / 'artifacts/reports/phase0-before-instance-007-sandbox-rpc.json', target / 'sandbox-rpc-timeout-007.json')
    shutil.copyfile(ROOT / 'artifacts/reports/phase3-instance-tests-final.log', target / 'unit-tests.log')
    source_paths = ['src/senior_ai/instance_labels.py', 'src/senior_ai/visibility_policy.py', 'src/senior_ai/frames.py',
        'scripts/probe_instances.py', 'scripts/audit_instance_probe.py', 'scripts/export_visible_instances.py',
        'scripts/collect_dataset.py', 'scripts/audit_motion_instances.py', 'scripts/diagnose_illumination.py',
        'scripts/summarize_visibility.py', 'configs/visibility_policy_v1_1.json', 'configs/illumination_diagnostic_v1.json']
    write_json(target / 'provenance.json', {'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {p: sha(ROOT / p) for p in source_paths}, 'receipts': checks,
        'original_model_checkpoint_sha256': d['checkpoint_sha256'],
        'note': 'Capture, audit, and policy source revisions differ deliberately; each original receipt retains its own source hashes. No raw image/weight changed.'})
    write_json(target / 'summary.json', {'status': 'completed-engineering-iteration', 'phase3_complete': False,
        'unique_raw_images_verified_again': sum(c['images'] for c in checks.values()), 'unit_tests_passed': 44,
        'training_approved': False, 'new_detector_trained': False,
        'paired_development_miou': {key: value['focus_macro_iou'] for key, value in d['results'].items()},
        'paired_geometry_pose_error': d['pose_error_max'], 'paired_semantic_target_mismatch_fraction': d['raw_semantic_target_mismatch_fraction']})
    print(target)


if __name__ == '__main__':
    main()
