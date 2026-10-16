"""Prepare isolated official CPython ZIPs for a bounded runtime comparison.

Downloads are a separate explicit step. No system Python or user registry changes.
The embedded runtimes are diagnostics, not the maintained training environment.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = []
    for version, minor in [('3.12.10', '312'), ('3.11.9', '311'), ('3.10.11', '310')]:
        source = ROOT / f'artifacts/downloads/python-{version}-embed-amd64.zip'
        target = ROOT / f'artifacts/runtimes/python{minor}-official'
        with zipfile.ZipFile(source) as archive:
            archive.extractall(target)
        if minor == '312':
            packages = ROOT / '.venv/Lib/site-packages'
        else:
            packages = target / 'Lib/site-packages'
            packages.mkdir(parents=True, exist_ok=True)
            carla_version = '0.9.15' if minor == '310' else '0.9.16'
            wheel = next((ROOT / f'artifacts/downloads/api{minor}').glob(f'carla-{carla_version}-cp{minor}-*-win_amd64.whl'))
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(packages)
        (target / f'python{minor}._pth').write_text(
            f'python{minor}.zip\n.\n{packages}\n', encoding='utf-8')
        executable = target / 'python.exe'
        probe = subprocess.run([str(executable), '-c',
                                'import sys,carla,importlib.metadata,json; print(json.dumps({"python":sys.version,"carla":importlib.metadata.version("carla")}))'],
                               capture_output=True, text=True, check=True, timeout=15)
        report.append({'version': version, 'source': f'https://www.python.org/ftp/python/{version}/{source.name}',
                       'archive_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'runtime': str(executable), 'import_probe': json.loads(probe.stdout)})
    output = ROOT / 'artifacts/reports/diagnostic-runtimes.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
