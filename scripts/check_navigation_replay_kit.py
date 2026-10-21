"""Check a relocated replay kit over its actual loopback HTTP service."""
import hashlib
import json
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'artifacts/runs/phase89-portable-replay-v1-001'
BASE = 'http://127.0.0.1:8769'


def request(path, body=None, expected=200):
    req = urllib.request.Request(BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={} if body is None else {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            status, data, mime = r.status, r.read(), r.headers.get('Content-Type', '')
    except urllib.error.HTTPError as e:
        status, data, mime = e.code, e.read(), e.headers.get('Content-Type', '')
    assert status == expected, (path, status, data[:200])
    return json.loads(data) if mime.startswith('application/json') else data


def main():
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1', 2000)) != 0
    runs = request('/api/navigation/runs')
    assert len(runs) == 1 and runs[0]['frames'] == 64
    run_id = runs[0]['id']
    request('/api/navigation/replay', {'run_id': run_id, 'index': 18})
    state = request('/api/navigation/state')
    record = state['latest']
    assert state['replay_only'] and not state['worker_active'] and record['frame'] == 121465
    for camera, entry in record['RGB'].items():
        png = request(f'/api/navigation/rgb?camera={camera}&frame={record["frame"]}')
        assert hashlib.sha256(png).hexdigest() == entry['sha256']
    request('/api/navigation/start', {'token': 'disabled'}, expected=409)
    for path in ['/', '/navigation.js', '/vendor/three.module.js']:
        assert request(path)
    code = '''import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
import senior_ai.navigation_service as module
service=module.NavigationService(replay_only=True)
assert Path(module.__file__).is_relative_to(Path.cwd())
assert 'torch' not in sys.modules and 'carla' not in sys.modules
print(json.dumps({'module':module.__file__,'torch_imported':False,'carla_imported':False}))
'''
    imports = json.loads(subprocess.check_output([sys.executable, '-c', code], cwd=KIT, text=True))
    manifest = json.loads((KIT/'SHA256_MANIFEST.json').read_text(encoding='utf8'))
    for entry in manifest['files']:
        p = KIT/entry['path']
        assert p.stat().st_size == entry['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256']
    receipt = {'status': 'passed_relocated_CPU_replay_HTTP_and_manifest',
        'run_id': run_id, 'frames': 64, 'RGB_checked': 8, 'sample_frame': record['frame'],
        'CARLA_stopped': True, 'vehicle_start_rejected': True, 'manifest_files_checked': len(manifest['files']),
        'import_proof': imports, 'port': 8769, 'formal_quality_passed': False}
    (ROOT/'artifacts/reports/phase89-portable-smoke-final.json').write_text(
        json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    print(receipt)


if __name__ == '__main__':
    main()
