"""Validate and extract within the workspace. Optional non-LLM download wait."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ROOT / 'artifacts/downloads/CARLA_0.9.16.zip')
    parser.add_argument('--destination', type=Path, default=ROOT / 'artifacts/carla')
    parser.add_argument('--expected-size', type=int, default=7812252810,
                        help='Official HEAD size on 2026-10-16; not a publisher checksum')
    parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    report = {'status': 'failed', 'source': 'https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip',
              'archive': str(args.archive.resolve()), 'errors': [],
              'publisher_checksum_verified': False,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    destination = args.destination.resolve()
    last_size, last_progress = -1, time.monotonic()
    try:
        if not destination.is_relative_to(ROOT) or destination == ROOT:
            raise ValueError('Extraction destination must be inside this project')
        if destination.exists() and any(destination.iterdir()):
            raise ValueError('Destination not empty; existing installation preserved')
        while True:
            size = args.archive.stat().st_size if args.archive.exists() else 0
            if size == args.expected_size and zipfile.is_zipfile(args.archive):
                break
            if size != last_size:
                last_size, last_progress = size, time.monotonic()
            if size > args.expected_size:
                raise ValueError('Unexpected archive size; verify official metadata')
            if time.monotonic() >= deadline or time.monotonic() - last_progress > 180:
                raise TimeoutError(f'Archive incomplete or stalled: {size}/{args.expected_size} bytes')
            time.sleep(2)
        print(json.dumps({'stage': 'validating_archive', 'bytes': size}), flush=True)
        digest = hashlib.sha256()
        with args.archive.open('rb') as src:
            while block := src.read(4 * 1024 * 1024):
                digest.update(block)
        report['bytes'] = size
        report['sha256'] = digest.hexdigest()
        with zipfile.ZipFile(args.archive) as archive:
            for item in archive.infolist():
                target = (destination / item.filename).resolve()
                if not target.is_relative_to(destination) or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError(f'Unsafe archive member: {item.filename}')
            launchers = [i.filename for i in archive.infolist()
                         if Path(i.filename).name.lower() == 'carlaue4.exe']
            if len(launchers) != 1:
                raise ValueError(f'Expected one CarlaUE4.exe, found {len(launchers)}')
            print(json.dumps({'stage': 'extracting', 'files': len(archive.infolist())}), flush=True)
            archive.extractall(destination)  # zipfile checks CRC while extracting.
        report['carla_root'] = str((destination / launchers[0]).parent)
        report['status'] = 'passed'
        report['zip_crc_verified'] = True
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    write_json(ROOT / 'artifacts/reports/installation.json', report)
    print(json.dumps(report), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
