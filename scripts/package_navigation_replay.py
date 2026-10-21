"""Build a local CPU-only replay kit with one genuine eight-RGB driving run."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=ROOT/'artifacts/runs/phase89-portable-replay-v1-001';archive=ROOT/'artifacts/runs/phase89-portable-replay-v1-001.zip'
    if out.exists() or archive.exists():raise FileExistsError('New kit path required')
    out.mkdir(parents=True)
    def copy(p):
        target=out/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for n in ['__init__.py','core.py','frames.py','navigation.py','navigation_io.py','navigation_lock.py','navigation_runtime.py','navigation_service.py']:copy(ROOT/'src/senior_ai'/n)
    for n in ['navigation.html','navigation.css','navigation.js','navigation-contract.js']:copy(ROOT/'web/world-viewer'/n)
    for p in (ROOT/'web/world-viewer/vendor').glob('*'):
        if p.is_file():copy(p)
    for n in ['configs/phase89_navigation_v1.json','configs/phase89_routes_v1.json','scripts/serve_navigation.py','requirements-navigation-viewer.txt']:copy(ROOT/n)
    run=ROOT/'artifacts/runs/phase89-drive-pedestrian-probe-v1-001'
    for p in run.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:copy(p)
    (out/'README.md').write_text('''# CPU-only navigation replay kit

This kit contains a genuine CARLA eight-RGB pedestrian-yielding run (64 frames), not a synthetic video. It is recorded playback and cannot start a vehicle. No GPU, Torch, CARLA installation or model weights are needed.

1. Use Python 3.12, then `python -m pip install -r requirements-navigation-viewer.txt` in this folder.
2. Run `python scripts/serve_navigation.py --replay-only`.
3. Open http://127.0.0.1:8768/ . Load the recorded run and play or seek. Ctrl+C stops the server.

Copy this entire folder or extract the ZIP before running; keep PNG/JSON paths intact. The full project provides live driving separately. Map and ego localization are explicit priors; AI perception remains research quality. This kit is local and has not been uploaded or published. Three.js licensing/provenance is retained in web/world-viewer/vendor.
''',encoding='utf8')
    files=[p for p in sorted(out.rglob('*')) if p.is_file()]
    manifest=[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    (out/'SHA256_MANIFEST.json').write_text(json.dumps({'files':manifest,'manifest_self_excluded':True},indent=2)+'\n',encoding='utf8')
    with zipfile.ZipFile(archive,'w') as z:
        for p in files+[out/'SHA256_MANIFEST.json']:
            z.write(p,p.relative_to(out).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.png' else zipfile.ZIP_DEFLATED,compresslevel=None if p.suffix=='.png' else 9)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt={'status':'packaged_local_CPU_replay','run_id':run.name,'recorded_frames':64,'recorded_RGB_images':512,
             'archive':archive.relative_to(ROOT).as_posix(),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
             'kit_files':len(manifest)+1,'GPU_or_CARLA_required':False,'Torch_import_required':False,'uploaded':False,
             'source_sha256':sha(Path(__file__))}
    (ROOT/'artifacts/reports/phase89-portable-kit-final.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(receipt)
if __name__=='__main__':main()
