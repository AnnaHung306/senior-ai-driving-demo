"""Check exported show chapters against immutable local run records, read-only.

This optional provenance audit needs original artifacts. The portable checker
and Node tests do not. Use a new report name; never overwrite old evidence.
"""
import argparse
import json
from pathlib import Path
from build_static_demo import ROOT, read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'artifacts/reports'):
        raise ValueError('Use a new report inside artifacts/reports')
    rows = []
    for chapter in read(ROOT/'docs/data/showcases.json')['chapters']:
        bundle = read(ROOT/'docs'/chapter['file'])
        run = ROOT/'artifacts/runs'/bundle['run_id']
        if chapter['kind'] == 'perception':
            original = read(run/'bundle.json')
            assert sha(run/'bundle.json') == bundle['provenance']['source_bundle_sha256']
            entries = [r for r in original['frames'] if r['scene_id'] == bundle['scene_id']]
        else:
            original = read(run/'report.json');entries = original['records']
            assert sha(run/'report.json') == bundle['provenance']['source_report_sha256']
            assert sha(run/'contract.json') == bundle['provenance']['source_contract_sha256']
        assert len(entries) == len(bundle['records'])
        for entry, record, asset in zip(entries, bundle['records'], bundle['frame_assets']):
            assert record == read(run/entry['file'])
            assert sha(run/entry['file']) == entry['sha256'] == asset['record_source_sha256']
            cameras = record['cameras'] if chapter['kind'] == 'perception' else record['RGB']
            for name, camera in cameras.items():
                assert sha(run/camera['file']) == camera['sha256'] == asset['images'][name]['source_PNG_sha256']
        rows.append({'chapter': chapter['id'], 'records_equal_to_original': len(entries),
            'original_PNG_hashes_verified': len(entries)*8, 'kind': chapter['kind'],
            'source_bundle_sha256': chapter['sha256']})
    perception = read(ROOT/'docs/data/perception.json');frames = perception['records']
    together = [r['frame'] for r in frames if {1,2,4}.issubset({o['class_id'] for o in r['objects']})]
    signal = read(ROOT/'docs/data/signal-turn.json')['records']
    first_green = next(i for i,r in enumerate(signal) if any(s['phase']=='green' for s in r['relevant_RGB_signals']))
    release = next(i for i,r in enumerate(signal) if i>first_green and r['plan']['state']=='CRUISE')
    result = {'status': 'passed_unchanged_sources', 'chapters': rows,
        'perception': {'car_motorcycle_person_simultaneous_model_outputs': len(together),
            'domain': perception['provenance']['domain'], 'new_accuracy_claim': False,
            'control_issued': False, 'memory_max_coast_s': .65,
            'held_frames': sum(bool(r['dynamics']['held_unobserved_tracks']) for r in frames)},
        'signal': {'first_RGB_green_frame': signal[first_green]['frame'],
            'first_RGB_green_s': signal[first_green]['simulation_s'],
            'release_frame': signal[release]['frame'],
            'release_delay_after_first_RGB_green_s': signal[release]['simulation_s']-signal[first_green]['simulation_s'],
            'final_state': signal[-1]['plan']['state'],
            'ego_yaw_change_rad': signal[-1]['ego_pose_navigation_prior'][2]-signal[0]['ego_pose_navigation_prior'][2]},
        'limits': 'Original model mistakes are retained. Simultaneous predicted classes are not a new accuracy score or identity ground truth.'}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
