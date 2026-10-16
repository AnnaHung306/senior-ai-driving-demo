"""Official CARLA archive download, bounded timeout, explicit opt-in resume."""
import argparse
import json
from pathlib import Path
import subprocess

URL = 'https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('artifacts/downloads'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    archive = args.directory / 'CARLA_0.9.16.zip'
    if archive.exists() and not args.resume:
        raise SystemExit('Archive exists. Use --resume or a new --directory; no data overwritten.')
    command = ['curl.exe', '--fail', '--location', '--silent', '--show-error',
               '--connect-timeout', '30', '--max-time', '1800',
               '--speed-limit', '1024', '--speed-time', '120',
               '--output', str(archive), URL]
    if args.resume:
        command += ['--continue-at', '-']
    result = subprocess.run(command)
    print(json.dumps({'status': 'downloaded' if result.returncode == 0 else 'failed',
                      'curl_exit_code': result.returncode, 'url': URL,
                      'archive': str(archive), 'note': 'Validate ZIP before installation.'}))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
