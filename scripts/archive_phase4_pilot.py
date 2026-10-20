"""Archive exact completed pilot artifacts, without inference or simulator work."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'evidence' / 'phase4'):
        raise ValueError('Archive must be inside evidence/phase4')
    if output.exists():
        raise FileExistsError('Use a new archive directory; no overwrite')
    runs = ROOT / 'artifacts/runs'
    run = runs / 'phase4-bev-pilot-002'
    evaluation = runs / 'phase4-bev-pilot-002-evaluation-final'
    qa = runs / 'phase4-visual-QA-001'
    selection = runs / 'phase4-pilot-selection-002'
    integrity = ROOT / 'artifacts/reports/phase4-artifact-integrity-001.json'
    review = read(qa / 'visual_review.json')
    report = read(evaluation / 'report.json')
    contract = read(run / 'contract.json')
    assert read(integrity)['status'] == 'passed_artifact_integrity_not_formal_quality'
    assert review['status'] == 'completed_assistant_visual_review_prototype_only'
    assert review['catalog_sha256'] == sha(qa / 'catalog.json')
    assert review['evaluation_report_sha256'] == sha(evaluation / 'report.json')
    assert review['prediction_contract_sha256'] == sha(run / 'contract.json')
    assert report['prediction_contract_sha256'] == sha(run / 'contract.json')
    assert report['evaluator_sha256'] == sha(ROOT / 'scripts/evaluate_phase4_pilot.py')
    assert report['GT_helper_sha256'] == sha(ROOT / 'scripts/select_phase4_pilot.py')
    for name, digest in contract['sources'].items():
        assert sha(run / 'source' / name) == digest
        assert sha(ROOT / name) == digest
    output.mkdir(parents=True)
    copied = []

    def copy(source, name):
        data = source.read_bytes()
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.write_bytes(data)
        assert target.read_bytes() == data
        copied.append({'archive': target.relative_to(output).as_posix(),
                       'source': source.relative_to(ROOT).as_posix(),
                       'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})

    def source_tar(name, pairs):
        members = []
        with tarfile.open(output / name, 'w:gz') as archive:
            for relative, source in sorted(pairs):
                data = source.read_bytes()
                info = tarfile.TarInfo(relative)
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
                members.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest()})
        with tarfile.open(output / name, 'r:gz') as archive:
            for member in members:
                assert hashlib.sha256(archive.extractfile(member['path']).read()).hexdigest() == member['sha256']
        write(output / (name + '.members.json'), members)

    copy(args.report.resolve(), 'README.md')
    for name in ['config.json', 'contract.json', 'progress.json', 'state.json']:
        copy(run / name, 'run/' + name)
    for row in report['records']:
        folder = Path(row['folder'])
        for name in ['frame.json', 'detections.json', 'panel.png']:
            copy(run / folder / name, Path('run') / folder / name)
    for source in sorted(evaluation.rglob('*')):
        if source.is_file():
            copy(source, Path('evaluation') / source.relative_to(evaluation))
    for source in sorted(qa.iterdir()):
        if source.is_file():
            copy(source, Path('visual_QA') / source.name)
    for name in ['selection.json', 'sequence_inputs.json', 'rgb_only_inventory.json']:
        copy(selection / name, 'selection/' + name)
    # Tiny manifests/calibration retained; original RGB and weights remain local.
    for source in sorted((selection / 'rgb_only').rglob('*.json')):
        copy(source, Path('selection/rgb_only_metadata') / source.relative_to(selection / 'rgb_only'))
    boundary = runs / 'phase4-RGB-boundary-001'
    for source in sorted(boundary.rglob('*')):
        if source.is_file() and source.suffix in {'.json', '.log', '.txt'}:
            copy(source, Path('boundary') / source.relative_to(boundary))
    for name in ['phase4-artifact-integrity-001.json', 'phase4-perception-unchanged-001.json',
                 'phase4-regression-final.log', 'phase4-path-refinement-CPU-001.json']:
        copy(ROOT / 'artifacts/reports' / name, 'checks/' + name)
    copy(runs / 'phase4-pilot-selection-001/failure.json', 'history/selector-001-failure.json')
    baseline = runs / 'phase4-bev-pilot-001'
    for name in ['config.json', 'contract.json', 'progress.json', 'state.json']:
        copy(baseline / name, 'history/paired-only-001/' + name)
    for name in ['report.json', 'evaluator.py', 'GT_helper.py', 'GT_input_inventory.json']:
        copy(runs / 'phase4-bev-pilot-001-evaluation' / name, 'history/initial-evaluation-001/' + name)
    for label, directory in [('inference-v1.1-source.tar.gz', run / 'source'),
                             ('paired-only-v1-source.tar.gz', baseline / 'source')]:
        source_tar(label, [(p.relative_to(directory).as_posix(), p) for p in directory.rglob('*') if p.is_file()])
    tool_names = ['scripts/archive_phase4_pilot.py', 'scripts/evaluate_phase4_pilot.py',
                  'scripts/make_phase4_review.py', 'scripts/prepare_phase4_inputs.py',
                  'scripts/run_phase4_pilot.py', 'scripts/select_phase4_pilot.py',
                  'scripts/verify_phase4_artifacts.py', 'scripts/verify_phase4_rgb_boundary.py',
                  'tests/test_bev_phase4.py', 'tests/test_bev_evaluation.py',
                  'PHASE4.md', 'requirements-phase4.txt']
    source_tar('validation-and-tools-source.tar.gz', [(n, ROOT / n) for n in tool_names])
    wheel = ROOT / 'artifacts/dependencies/phase4/opencv_python_headless-4.13.0.92-cp37-abi3-win_amd64.whl'
    wheel_sha = '77a82fe35ddcec0f62c15f2ba8a12ecc2ed4207c17b0902c7a3151ae29f37fb6'
    assert sha(wheel) == wheel_sha
    write(output / 'dependencies.json', {'python': '3.12.14', 'numpy': '2.3.5',
          'torch': '2.10.0+cu128', 'added_distribution': 'opencv-python-headless==4.13.0.92',
          'wheel': wheel.relative_to(ROOT).as_posix(), 'wheel_sha256': wheel_sha,
          'wheel_in_git': False, 'installation': 'Official PyPI pinned wheel, --no-deps'})
    write(output / 'decision.json', {'status': 'completed_geometric_BEV_prototype',
          'phase4_formal_complete': False, 'control_issued': False,
          'original_raw_statuses_preserved': True,
          'run_contract_sha256': sha(run / 'contract.json'),
          'evaluation_report_sha256': sha(evaluation / 'report.json'),
          'visual_review_sha256': sha(qa / 'visual_review.json'),
          'tests': {'total': 193, 'failures': 0, 'skipped': 0},
          'remaining': ['Fresh predefined complete sunny flat marked routes without nearby walls',
                        'Longer turns, missing observations and failure continuity evaluation',
                        'Computation latency and ground-contact object reliability'],
          'local_only': ['Original RGB/GT corpus, model weights, OpenCV wheel',
                         'Per-view observation and floating BEV arrays; output hashes in integrity report'],
          'limits': 'Known short train/validation pilot, no independent heldout, 3D box or live-driving acceptance'})
    write(output / 'copy_receipt.json', {'status': 'exact_raw_copies', 'files': copied})
    # The manifest excludes itself; it is a byte receipt, never a quality gate.
    manifest = [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(output.rglob('*')) if p.is_file()]
    write(output / 'manifest.json', {'scope': 'All archive files except this manifest', 'files': manifest})
    for row in manifest:
        assert sha(output / row['path']) == row['sha256']
    print(json.dumps({'status': 'archived_exact_bytes', 'files': len(manifest) + 1,
                      'MiB': sum(r['bytes'] for r in manifest) / 1024**2,
                      'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    main(parser.parse_args())
