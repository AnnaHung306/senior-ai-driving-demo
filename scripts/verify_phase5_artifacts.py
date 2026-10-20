"""CPU-only recount/provenance check. An integrity pass is not a quality pass."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
from phase5_gt import read, sha, audited_inventory
from train_phase5_cuboids import condition_metrics
from evaluate_phase5_world import occupancy_counts, occupancy_scores, metrics_objects, merge_audit
from senior_ai.world_schema import validate_world_frame
from senior_ai.world_tracking import WorldTracker
from senior_ai.core import write_json
from senior_ai.visibility_policy import ignore_background_proposals


def main(args):
    run = args.run;ev = args.evaluation;qa = args.review;training = args.training;data = args.crops;report = read(ev/'report.json');contract = read(run/'contract.json')
    state = read(run/'state.json');train_state = read(training/'state.json');train_contract = read(training/'contract.json');data_contract = read(data/'contract.json')
    assert train_state['status'] == 'completed_finite_baseline_training' and train_state['epochs'] == 12
    assert train_state['validation_reload_exact'] and min(train_state['first_gradient'].values()) > 0
    assert sha(training/'best.pt') == train_state['best_checkpoint_sha256'] == contract['checkpoint_sha256']
    assert train_contract == contract['training_contract']
    assert sha(data/'contract.json') == train_contract['data_contract_sha256']
    for parent, source in [(run, contract), (training, train_contract)]:
        for name, digest in source['sources'].items():assert sha(parent/'source'/name) == sha(ROOT/name) == digest, name
    for name, digest in data_contract['files'].items():assert sha(data/name) == digest, name
    root = Path(data_contract['dataset']);_, original = audited_inventory(root, ROOT)
    verified_sources = {}
    for inventory in [read(data/'source_inventory.json'), read(ev/'GT_input_inventory.json')]:
        for name, digest in inventory.items():
            if name not in verified_sources:
                assert sha(root/name) == digest == original[name], name;verified_sources[name] = digest
    with np.load(data/'validation.npz') as package:
        assert package['crops'].shape[0] == 232 and np.isfinite(package['targets']).all()
    predictions = np.load(training/'validation_predictions.npy')
    conditional = condition_metrics(predictions, read(data/'validation_records.json'), train_contract['class_size_priors'], read(run/'config.json')['center_residual_scale_m'])
    assert conditional == read(training/'conditional_validation.json')
    matrix = np.zeros((11, 12), np.int64);scenes = defaultdict(lambda: np.zeros((11, 12), np.int64));total = {n: Counter() for n in ['learned', 'prior']}
    split_counts = {s: {n: Counter() for n in total} for s in ['train', 'validation']};split_errors = {s: {n: [] for n in total} for s in split_counts}
    tracker = WorldTracker(read(run/'config.json')['tracker']);unchanged = [];all_timings = [];ignored_summary = Counter();world_sources = {}
    for record in report['records']:
        folder = Path(record['folder']);w = read(run/folder/'world.json')
        with np.load(run/folder/'occupancy.npz') as p:occupancy = p['semantic']
        assert validate_world_frame(w, occupancy)
        assert sha(run/folder/'occupancy.npz') == w['occupancy']['sha256']
        with np.load(ev/folder/'GT_reference.npz') as p:truth = p['semantic']
        counts = occupancy_counts(occupancy, truth);assert counts.tolist() == record['occupancy_confusion_11x12']
        matrix += counts;scenes[record['scene_id']] += counts
        split = next(s for s in split_counts if record['scene_id'] in data_contract['split_summaries'][s]['scenes'])
        visible = {camera: read(root/f"labels/visible/{record['scene_id']}/{camera}/{record['frame']:08}.json") for camera in ['front', 'front_right', 'right', 'rear_right', 'rear', 'rear_left', 'left', 'front_left']}
        for name, key in [('learned', 'objects'), ('prior', 'prior_baseline_objects')]:
            objects = [o for o in w[key] if np.linalg.norm(o['center_ego_m'][:2]) <= 25]
            metrics, errors, _ = metrics_objects(objects, record['GT_actors'])
            metrics.update(merge_audit(objects, {n: v['positive_objects'] for n, v in visible.items()}))
            assert metrics == record[name]['counts'] and errors == record[name]['matched_errors']
            total[name].update(metrics);split_counts[split][name].update(metrics);split_errors[split][name] += errors
            # Supplementary audit of the inherited 2D ignore policy. Do not alter
            # the original strict3D counts or retroactively make them pass.
            for obj in objects:
                statuses = []
                for source in obj['sources']:
                    label = visible[source['camera']];ignore = [r['bbox_xyxy'] for r in label['ignore_regions']]
                    statuses.append(ignore_background_proposals([source['bbox_xyxy']], [0], ignore)[0] == -1)
                ignored_summary[name+'_all_sources_in_2D_ignore_regions'] += all(statuses)
        old = read(args.original_run/folder/'world.json')
        assert all(w[k] == old[k] for k in ['objects', 'prior_baseline_objects', 'tracking', 'single_camera_cuboids', 'rejected'])
        unchanged.append(str(folder))
        parent = read(args.phase4/folder/'frame.json');assert sha(args.phase4/folder/'frame.json') == w['parent_frame_sha256']
        manifest = read(root/w['parent_manifest']);calibration_sha = sha(root/manifest['calibration'])
        tracked, replayed = tracker.update(w['objects'], w['scene_id'], w['timestamp_s'], calibration_sha, parent['motion'])
        assert tracked == w['objects'] and replayed == w['tracking']
        all_timings.append(sum(w['timing_ms'].values()))
        for p in (run/folder).iterdir():world_sources[str(p.relative_to(run))] = sha(p)
    assert matrix.tolist() == report['occupancy_confusion_11x12'] and occupancy_scores(matrix) == report['occupancy']
    assert {s: occupancy_scores(m) for s, m in scenes.items()} == report['occupancy_by_scene']
    for name in total:
        for key, value in total[name].items():assert value == report['objects'][name][key]
    review = read(qa/'visual_review.json');catalog = read(qa/'catalog.json')
    assert review['status'] == 'completed_actual_visual_review_research_baseline_only'
    assert review['catalog_sha256'] == sha(qa/'catalog.json') and review['evaluation_report_sha256'] == sha(ev/'report.json')
    assert report['run_contract_sha256'] == review['run_contract_sha256'] == sha(run/'contract.json')
    for page in catalog['pages']:
        assert sha(qa/page['path']) == page['sha256']
        for source in page['sources']:assert sha(Path(source['source'])) == source['sha256']
    for case in catalog['full_cases']:
        for name in ['prediction', 'evaluation']:assert sha(Path(case[name])) == case[name+'_sha256']
    assert read(args.boundary/'report.json')['status'] == 'passed_expected_rejections'
    assert report['evaluator_sha256'] == sha(ROOT/'scripts/evaluate_phase5_world.py')
    log = (ROOT/'artifacts/reports/phase5-regression-final.log').read_text(encoding='utf-8')
    assert 'Ran 220 tests' in log and log.rstrip().endswith('OK')
    def summary(counts, errors):
        return {**dict(counts), 'precision_at_2m': counts['TP']/(counts['TP']+counts['FP']), 'recall_at_2m': counts['TP']/(counts['TP']+counts['FN']),
                'precision_IoU3D_025': counts['TP_IoU3D_025']/(counts['TP_IoU3D_025']+counts['FP_IoU3D_025']),
                'recall_IoU3D_025': counts['TP_IoU3D_025']/(counts['TP_IoU3D_025']+counts['FN_IoU3D_025']),
                'errors_conditional_on_2m_matches': {key: np.asarray([r[key] for r in errors]).mean(0).tolist() for key in errors[0]} if errors else None}
    write_json(args.output, {'status': 'passed_integrity_not_formal_quality', 'tool_sha256': sha(Path(__file__)),
               'bundles_recounted': len(unchanged), 'sources_verified': len(contract['sources']), 'original_GT_RGB_files_reverified': len(verified_sources),
               'conditional232_validation_recount_exact': True, 'all32_3D_objects_and_tracking_exact_between001_002': True,
               'WorldFrame_contract_and_free_ground_slab_checked': True, 'occupancy_counts_all_scenes_exact': True,
               'split_results': {s: {n: summary(split_counts[s][n], split_errors[s][n]) for n in total} for s in split_counts},
               'supplementary_2D_ignore_audit': dict(ignored_summary), 'strict3D_counts_not_modified_by_audit': True,
               'compute_including_replayed_upstream_ms': {'p50': float(np.quantile(all_timings, .5)), 'p95': float(np.quantile(all_timings, .95)),
                                                        'limits': 'Stage-wise measured sums, not live capture/control wall-clock; excludes writes/startup'},
               'output_manifest': {p.relative_to(run).as_posix(): sha(p) for p in sorted(run.rglob('*')) if p.is_file()},
               'run_contract_sha256': sha(run/'contract.json'), 'evaluation_report_sha256': sha(ev/'report.json'), 'quality_passed': False})
    print(json.dumps({'integrity': 'passed', 'bundles': len(unchanged), 'reverified_original_files': len(verified_sources), 'quality_passed': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['run', 'original-run', 'phase4', 'training', 'crops', 'evaluation', 'review', 'boundary', 'output']:p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
