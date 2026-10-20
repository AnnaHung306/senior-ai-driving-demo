"""Fetch only pinned official Three.js browser files, checking npm SHA512."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://registry.npmjs.org/three/-/three-0.180.0.tgz'
INTEGRITY = 'sha512-o+qycAMZrh+TsE01GqWUxUIKR1AL0S8pq7zDkYOQw8GqfX8b8VoCKYUoHbhiX5j+7hr8XsuHDVU6+gkQJQKg9w=='
FILES = {'build/three.module.js': 'three.module.js', 'build/three.core.js': 'three.core.js',
         'examples/jsm/controls/OrbitControls.js': 'OrbitControls.js', 'LICENSE': 'LICENSE.three.txt'}


def main():
    out = ROOT/'web/world-viewer/vendor'
    if out.exists():raise FileExistsError('Vendor output already exists; preserve pinned bytes')
    with urllib.request.urlopen(URL, timeout=60) as response:data = response.read()
    if 'sha512-'+base64.b64encode(hashlib.sha512(data).digest()).decode() != INTEGRITY:
        raise ValueError('Official npm package integrity mismatch')
    cache = ROOT/'artifacts/dependencies/phase6';cache.mkdir(parents=True, exist_ok=True)
    (cache/'three-0.180.0.tgz').write_bytes(data);out.mkdir(parents=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        for source, target in FILES.items():
            raw = tar.extractfile('package/'+source).read();(out/target).write_bytes(raw)
            files[target] = {'package_member': source, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    (out/'provenance.json').write_text(json.dumps({'package': 'three', 'version': '0.180.0', 'url': URL,
        'npm_integrity_sha512_verified': INTEGRITY, 'tarball_sha256': hashlib.sha256(data).hexdigest(),
        'files': files, 'license': 'MIT; original LICENSE retained'}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'verified_official_vendor', 'version': '0.180.0', 'files': len(files)}))


if __name__ == '__main__':main()
