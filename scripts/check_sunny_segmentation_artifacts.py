"""CPU/file-only check of a completed segmentation run; never evaluates heldout.

Exit 0 means stored artifacts are consistent, not that model quality passed.
The run and its frozen source are read-only. Output is replaced atomically.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(run):
    state = read(run / 'state.json')
    require(state['status'] == 'completed' and state['task'] == 'segmentation', 'Run is not completed segmentation')
    require(state['checkpoint_reload_identical'], 'Recorded checkpoint reload failed')
    provenance = read(run / 'provenance.json')
    config = read(run / 'config.json')
    hashes = {}
    for filename, expected in [
        ('best.pt', state['best_sha256']), ('last.pt', state['last_sha256']),
        ('validation.json', state['validation_sha256']),
        ('validation_frames.jsonl', state['validation_frames_sha256']),
        ('config.json', provenance['config_sha256']),
    ]:
        hashes[filename] = sha(run / filename)
        require(hashes[filename] == expected, 'SHA mismatch: ' + filename)
    source = provenance['source']['files']
    require(hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest() == provenance['source']['sha256'], 'Source manifest changed')
    for relative, expected in source.items():
        path = (run / 'source' / relative).resolve()
        require(path.is_relative_to((run / 'source').resolve()), 'Source path escapes snapshot')
        require(sha(path) == expected, 'Frozen source changed: ' + relative)

    result = read(run / 'validation.json')
    matrix = result['confusion_true_rows_predicted_columns']
    n = len(result['classes'])
    require(len(matrix) == n and all(len(row) == n for row in matrix), 'Invalid confusion shape')
    summed = [[0] * n for _ in range(n)]
    indices, groups = set(), set()
    with (run / 'validation_frames.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            frame = json.loads(line)
            require(frame['split'] == 'validation', 'Non-validation frame in report')
            require(frame['index'] not in indices, 'Duplicate validation index')
            indices.add(frame['index']); groups.add(frame['group_id'])
            m = frame['confusion_true_rows_predicted_columns']
            require(len(m) == n and all(len(row) == n for row in m), 'Invalid frame confusion shape')
            for i in range(n):
                for j in range(n):
                    require(type(m[i][j]) is int and m[i][j] >= 0, 'Invalid confusion count')
                    summed[i][j] += m[i][j]
    require(indices == set(range(len(indices))) and indices, 'Validation indices incomplete')
    require(summed == matrix, 'Per-frame confusion does not sum to aggregate')
    focus = []
    for i, row in enumerate(result['classes']):
        support = sum(matrix[i]); predicted = sum(r[i] for r in matrix)
        union = support + predicted - matrix[i][i]
        iou = matrix[i][i] / union if union else None
        require(row['support_pixels'] == support and row['predicted_pixels'] == predicted, 'Class support differs')
        require(row['iou'] == iou, 'Class IoU differs: ' + row['name'])
        if i and support:
            focus.append(iou)
    require(math.isclose(sum(focus) / len(focus), result['focus_macro_iou'], abs_tol=1e-12), 'Focus mIoU differs')
    epochs = [json.loads(line) for line in (run / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    require([r['epoch'] for r in epochs] == list(range(config['segmentation']['epochs'])), 'Epoch history incomplete')
    best = max(epochs, key=lambda row: row['validation_score'])
    require(best['epoch'] == state['best_epoch'] and best['validation_score'] == state['best_score'] == result['focus_macro_iou'], 'Best selection differs')
    require(epochs[-1]['updates'] == state['updates'], 'Update count differs')

    # These are diagnostics from the saved validation report, not a new evaluation.
    gates = config['quality_gates']
    checks = {'focus_mIoU': result['focus_macro_iou'] >= gates['segmentation_focus_mIoU'],
              'all_focus_present': len(focus) == n - 1 == result['focus_supported_classes'],
              'visible_margin_f1': result['visible_margin']['f1'] >= gates['minimum_visible_margin_f1']}
    checks.update({r['name'] + '/IoU': (r['iou'] or 0) >= gates['segmentation_per_focus_class_IoU'] for r in result['classes'][1:]})
    return {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'run': str(run),
            'checker_sha256': sha(Path(__file__)),
            'scope': 'Stored-artifact consistency only; no model inference, GPU access, raw dataset re-audit, or heldout evaluation',
            'artifact_integrity_passed': True, 'hashes': hashes,
            'source_snapshot_files_verified': len(source), 'data_signature_recorded_at_training': provenance['data'],
            'validation_frames': len(indices), 'validation_groups': sorted(groups),
            'updates': state['updates'], 'best_epoch_zero_based': state['best_epoch'],
            'focus_macro_iou': result['focus_macro_iou'], 'visible_margin_f1': result['visible_margin']['f1'],
            'classes': result['classes'], 'gates': gates, 'validation_quality_passed': all(checks.values()),
            'validation_checks': checks, 'failed_validation_checks': [k for k, v in checks.items() if not v],
            'lockbox_opened_recorded': state['lockbox_opened'],
            'lockbox_marker_exists': (run / 'lockbox_started.json').exists()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    require(not output.is_relative_to(run), 'Write the diagnostic outside the immutable run')
    report = check(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({k: report[k] for k in ['artifact_integrity_passed', 'validation_frames', 'validation_quality_passed', 'failed_validation_checks']}))
