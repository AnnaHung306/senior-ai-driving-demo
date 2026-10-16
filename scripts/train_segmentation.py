"""Train a frozen experiment; logs and checkpoints are owned by one run directory."""
import argparse
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json
from senior_ai.segmentation_training import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        run_experiment(ROOT, args.config.resolve(), args.run.resolve(), args.resume)
    except Exception as exc:
        if args.run.is_dir():
            write_json(args.run / 'failure.json', {'status': 'failed', 'exception': type(exc).__name__,
                'message': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
