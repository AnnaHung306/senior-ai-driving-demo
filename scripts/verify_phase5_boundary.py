"""Actual CPU CLI negative cases; no model load, simulator, or GPU inference."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import json
import hashlib
ROOT = Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    out = args.output.resolve()
    if out.exists():raise FileExistsError('New boundary proof required')
    out.mkdir(parents=True);cases = []
    for kind, expected in [('extra_GT_file', 'RGB staging contains extra/GT files'), ('GT_pose_in_sequence', 'GT fields forbidden in sequence')]:
        fixture = ROOT/'artifacts/runs/phase4-RGB-boundary-001'/kind;target = out/kind;target.mkdir()
        output = target/'must_not_exist'
        command = [sys.executable, '-X', 'utf8', str(ROOT/'scripts/run_phase5_world.py'), '--root', str(fixture/'root'),
                   '--input-list', str(fixture/'inputs.json'), '--inventory', str(fixture/'inventory.json'),
                   '--phase4', 'artifacts/runs/phase4-bev-pilot-002', '--phase4-integrity', 'artifacts/reports/phase4-artifact-integrity-001.json',
                   '--checkpoint', str(target/'nonexistent_checkpoint.pt'), '--config', 'configs/phase5_world_v1.json', '--output', str(output), '--device', 'cpu']
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, capture_output=True, text=True, timeout=60)
        (target/'stderr.log').write_text(result.stderr, encoding='utf-8')
        if result.returncode == 0 or expected not in result.stderr or output.exists():raise ValueError('Expected early rejection missing: '+kind)
        cases.append({'case': kind, 'return_code': result.returncode, 'expected': expected, 'output_not_created': True,
                      'nonexistent_checkpoint_not_opened': True, 'stderr_sha256': sha(target/'stderr.log'),
                      'fixture_inventory_sha256': sha(fixture/'inventory.json'), 'fixture_sequence_sha256': sha(fixture/'inputs.json')})
    report = {'status': 'passed_expected_rejections', 'cases': cases, 'inference_tool_sha256': sha(ROOT/'scripts/run_phase5_world.py'),
              'tool_sha256': sha(Path(__file__)), 'scope': 'CPU actual CLI rejects staged GT/sequence GT before opening a deliberately nonexistent checkpoint'}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8');print(report['status'])


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True);main(p.parse_args())
