"""Server readiness and concentrated acceptance, with no LLM in the tick loop."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-server-seconds', type=int, default=180)
    parser.add_argument('--run-id', default='phase0-001')
    args = parser.parse_args()
    import carla
    deadline = time.monotonic() + args.wait_server_seconds
    ready = False
    while time.monotonic() < deadline:
        try:
            # A client created before startup may retain a failed connection.
            # Rebuild the transport for each bounded readiness attempt.
            client = carla.Client('127.0.0.1', 2000)
            client.set_timeout(2)
            client.get_server_version()
            ready = True
            break
        except RuntimeError:
            time.sleep(2)
    if not ready:
        write_json(ROOT / 'artifacts/reports/phase0-pipeline.json',
                   {'status': 'failed', 'stage': 'server_start', 'error': 'RPC startup timeout'})
        return 1
    commands = [[sys.executable, 'scripts/run.py', 'doctor'],
                [sys.executable, 'scripts/run.py', 'smoke', '--output', f'artifacts/runs/{args.run_id}']]
    codes = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT)
        codes.append(result.returncode)
        if result.returncode:
            break
    report = {'status': 'passed' if len(codes) == 2 and not any(codes) else 'failed',
              'exit_codes': codes, 'run_id': args.run_id}
    write_json(ROOT / 'artifacts/reports/phase0-pipeline.json', report)
    print(json.dumps(report))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
