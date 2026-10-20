"""Exact-byte archive of this completed finite Phase5 baseline; no GPU work."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
ROOT = Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path, value):path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main(args):
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'evidence/phase5') or out.exists():raise ValueError('New evidence/phase5 output required')
    runs = ROOT/'artifacts/runs';reports = ROOT/'artifacts/reports'
    run = runs/'phase5-world-v1-002';evaluation = runs/'phase5-world-v1-002-evaluation';qa = runs/'phase5-QA-v1-002'
    training = runs/'phase5-cuboid-v1-001';crops = runs/'phase5-crops-v1-002'
    proof = read(reports/'phase5-integrity-v1-002.json');contract = read(run/'contract.json');review = read(qa/'visual_review.json')
    assert proof['status'] == 'passed_integrity_not_formal_quality'
    assert proof['run_contract_sha256'] == sha(run/'contract.json') == review['run_contract_sha256']
    assert proof['evaluation_report_sha256'] == sha(evaluation/'report.json') == review['evaluation_report_sha256']
    assert sha(ROOT/'scripts/verify_phase5_artifacts.py') == proof['tool_sha256']
    assert read(reports/'phase5-supplement-v1-001.json')['tool_sha256'] == sha(ROOT/'scripts/analyze_phase5_errors.py')
    assert sha(training/'best.pt') == contract['checkpoint_sha256']
    log = (reports/'phase5-regression-final-002.log').read_text(encoding='utf-8')
    assert 'Ran 223 tests' in log and log.rstrip().endswith('OK')
    for name, digest in proof['output_manifest'].items():assert sha(run/name) == digest
    out.mkdir(parents=True);copies = []

    def copy(source, name):
        target = out/name;target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():raise FileExistsError(target)
        data = source.read_bytes();target.write_bytes(data);assert target.read_bytes() == data
        copies.append({'archive': target.relative_to(out).as_posix(), 'source': source.relative_to(ROOT).as_posix(),
                       'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})

    def tree(source, prefix, excluded=()):
        for p in sorted(source.rglob('*')):
            if p.is_file() and not any(e in p.relative_to(source).parts for e in excluded):copy(p, Path(prefix)/p.relative_to(source))

    def archive_sources(source, name):
        pairs = [(p.relative_to(source).as_posix(), p) for p in source.rglob('*') if p.is_file()]
        bundle(pairs, name)

    def bundle(pairs, name):
        members = []
        with tarfile.open(out/name, 'w:gz') as tar:
            for relative, p in sorted(pairs):
                data = p.read_bytes();info = tarfile.TarInfo(relative);info.size = len(data);info.mode = 0o644
                tar.addfile(info, io.BytesIO(data));members.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest()})
        with tarfile.open(out/name) as tar:
            for row in members:assert hashlib.sha256(tar.extractfile(row['path']).read()).hexdigest() == row['sha256']
        write(out/(name+'.members.json'), members)

    copy(reports/'phase5-report-v1-001.md', 'README.md');copy(reports/'phase5-model-card-v1-001.md', 'MODEL_CARD.md')
    tree(run, 'run', ('source',));tree(evaluation, 'evaluation');tree(qa, 'visual_QA')
    for name in ['contract.json', 'config.json', 'state.json', 'history.json', 'conditional_validation.json', 'validation_predictions.npy']:copy(training/name, 'training/'+name)
    for name in ['contract.json', 'source_inventory.json', 'train_records.json', 'validation_records.json']:copy(crops/name, 'prepared_data/'+name)
    tree(runs/'phase5-boundary-v1-001', 'boundary')
    for name in ['phase5-integrity-v1-002.json', 'phase5-supplement-v1-001.json', 'phase5-regression-final-002.log',
                 'phase5-regression-final.log', 'phase5-regression-001.log']:copy(reports/name, 'checks/'+name)
    copy(runs/'phase5-crops-v1-001/failure.json', 'history/initial-crop-guard-failure.json')
    original = runs/'phase5-world-v1-001'
    for name in ['contract.json', 'config.json', 'state.json']:copy(original/name, 'history/initial-ray-world/'+name)
    original_ev = runs/'phase5-world-v1-001-evaluation'
    copy(original_ev/'report.json', 'history/initial-ray-evaluation/report.json')
    tree(original_ev/'source', 'history/initial-ray-evaluation/source')
    for source, name in [(run/'source', 'inference-final-source.tar.gz'), (original/'source', 'inference-initial-source.tar.gz'),
                         (training/'source', 'training-source.tar.gz'), (crops/'source', 'prepared-targets-source.tar.gz')]:archive_sources(source, name)
    tools = sorted(set(list((ROOT/'scripts').glob('*phase5*.py'))+list((ROOT/'tests').glob('test_world*.py'))+
                       [ROOT/'src/senior_ai/world_schema.py', ROOT/'PHASE5.md', ROOT/'requirements-phase5.txt']))
    bundle([(p.relative_to(ROOT).as_posix(), p) for p in tools], 'validation-and-tools-source.tar.gz')
    write(out/'registry_entry.json', {'model_id': 'phase5-cuboid-v1-001', 'task': 'conditional_RGB_3D_cuboid_head',
          'checkpoint_local_path': 'artifacts/runs/phase5-cuboid-v1-001/best.pt', 'checkpoint_sha256': contract['checkpoint_sha256'],
          'pretrained': False, 'parameters': 94424, 'train_crops': 659, 'validation_crops': 232,
          'status': 'research_baseline_not_formal_quality_pass', 'reliable_orientation_validated': False,
          'reliable_occupancy_validated': False, 'checkpoint_in_git': False, 'depth_auxiliary_adopted': False})
    write(out/'decision.json', {'status': 'completed_Phase5_engineering_baseline', 'formal_phase5_complete': False,
          'quality_passed': False, 'tests': {'total': 223, 'failed': 0, 'skipped': 0}, 'control_issued': False,
          'checkpoint_sha256': contract['checkpoint_sha256'], 'run_contract_sha256': sha(run/'contract.json'),
          'evaluation_report_sha256': sha(evaluation/'report.json'), 'visual_review_sha256': sha(qa/'visual_review.json'),
          'all32_3D_objects_and_tracking_exact_between001_002': True,
          'training_budget': 'One fixed12-epoch scratch run,132updates, no extra search/retraining',
          'remaining': ['Orientation symmetry and pedestrian geometry', 'Duplicate hypotheses and stable long-term ID',
                        'Sparse free evidence/unknown and fine geometry', 'Fresh predefined full routes and latency acceptance'],
          'local_only': ['Original corpus/RGB, resized crop NPZ packages, model/optimizer weights'],
          'original_pending_statuses_preserved': True})
    write(out/'copy_receipt.json', {'status': 'exact_raw_copies', 'files': copies})
    manifest = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'manifest.json', {'scope': 'Every archived file except this manifest; integrity only', 'files': manifest})
    for row in manifest:assert sha(out/row['path']) == row['sha256']
    print(json.dumps({'status': 'archived_exact_bytes', 'files': len(manifest)+1, 'MiB': sum(r['bytes'] for r in manifest)/1024**2, 'output': str(out)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True);main(p.parse_args())
