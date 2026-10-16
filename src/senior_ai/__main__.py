import argparse
from pathlib import Path
from . import doctor


def main():
    parser = argparse.ArgumentParser(description='Simulation-only autonomous driving research')
    commands = parser.add_subparsers(dest='command', required=True)
    env = commands.add_parser('doctor', help='Write environment readiness report')
    env.add_argument('--host', default='127.0.0.1')
    env.add_argument('--port', type=int, default=2000)
    env.add_argument('--output', type=Path, default=Path('artifacts/reports/environment.json'))
    env.set_defaults(run=doctor.run)
    smoke = commands.add_parser('smoke', help='Actual CARLA acceleration, braking and camera acceptance')
    smoke.add_argument('--host', default='127.0.0.1')
    smoke.add_argument('--port', type=int, default=2000)
    smoke.add_argument('--map', default='Town01')
    smoke.add_argument('--timeout', type=float, default=30)
    smoke.add_argument('--output', type=Path, required=True, help='New run directory; existing paths are refused')
    smoke.add_argument('--criteria', type=Path, default=Path('configs/phase0_acceptance.json'))
    def run_smoke(args):
        from .smoke import run
        return run(args)
    smoke.set_defaults(run=run_smoke)
    args = parser.parse_args()
    return args.run(args)


if __name__ == '__main__':
    raise SystemExit(main())
