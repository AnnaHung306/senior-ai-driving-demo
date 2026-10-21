"""Extend the existing static demo with two audited, unaltered recordings.

No simulator, inference, GT geometry or training is used. Perception records
retain their own schema: they must never masquerade as a successful drive.
Requires Pillow only when rebuilding JPEG previews from local original PNGs.
"""
from pathlib import Path
from PIL import Image
from build_static_demo import ROOT, read, write, sha

SITE = ROOT/'docs'


def previews(run, record, cameras):
    images = {}
    for camera, rgb in cameras.items():
        original = run/rgb['file']
        assert sha(original) == rgb['sha256']
        target = SITE/f'images/{record["frame"]}/{camera}.jpg'
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(original) as image:
            assert image.size == (640, 360)
            image.convert('RGB').save(target, quality=86, optimize=True, subsampling=0)
        images[camera] = {'url': target.relative_to(SITE).as_posix(),
            'source_PNG_sha256': rgb['sha256'], 'web_JPEG_sha256': sha(target),
            'width': 640, 'height': 360}
    return images


def perception():
    run = ROOT/'artifacts/runs/phase7-viewer-bundle-v1-001'
    original = read(run/'bundle.json')
    records, assets = [], []
    # Use a whole contiguous real sequence. The bundle also contains synthetic
    # diagnostics, which are deliberately excluded by selecting its real index.
    selected = [e for e in original['frames'] if e['scene_id'] == 'town01-00']
    assert len(selected) == 60
    for entry in selected:
        source = run/entry['file']
        assert sha(source) == entry['sha256']
        record = read(source)
        assert record['GT_used_in_inference'] is False and record['control_issued'] is False
        assert record['future_observations_used'] is False
        records.append(record)
        assets.append({'frame': record['frame'], 'record_source_sha256': entry['sha256'],
            'images': previews(run, record, record['cameras'])})
    motion = read(ROOT/'configs/phase7_motion_v1_2.json')
    payload = {'schema_version': 1, 'mode': 'recorded_perception', 'kind': 'perception',
        'run_id': run.name, 'scene_id': 'town01-00', 'records': records, 'frame_assets': assets,
        'summary': {'frames': len(records), 'simulation_s': records[-1]['timestamp_s']-records[0]['timestamp_s'],
            'status': 'perception_only', 'formal_quality_passed': False},
        'memory_max_coast_s': .65,
        'provenance': {'not_live_inference': True, 'other_actor_GT_in_main_view': False,
            'model_outputs_unchanged': True, 'JPEG_is_lossy_preview_only': True,
            'source_bundle_sha256': sha(run/'bundle.json'),
            'motion_configuration_sha256': sha(ROOT/'configs/phase7_motion_v1_2.json'),
            'source_mode': original['source_mode'], 'motion_profile': original['motion_profile'],
            'domain': 'original_training_sequence_not_unseen_validation',
            'control_issued': False, 'future_observations_used': False,
            'checkpoint_sha256': original['checkpoint_sha256'],
            'held_tracks_used_for_occupancy': False}}
    assert motion['maximum_coast_s'] == payload['memory_max_coast_s']
    write(SITE/'data/perception.json', payload)


def signal_turn():
    run = ROOT/'artifacts/runs/phase10-drive-long-red-probe-v1-001'
    report = read(run/'report.json')
    assert report['engineering_route_passed'] and all(report['checks'].values())
    assert report['status'] == 'arrived' and report['frames'] == 236
    records, assets = [], []
    for entry in report['records']:
        source = run/entry['file']
        assert sha(source) == entry['sha256']
        record = read(source)
        assert record['other_actor_GT_in_planning'] is False
        records.append(record)
        assets.append({'frame': record['frame'], 'record_source_sha256': entry['sha256'],
            'images': previews(run, record, record['RGB'])})
    payload = {'schema_version': 1, 'mode': 'recorded_replay', 'kind': 'drive',
        'run_id': run.name, 'records': records, 'frame_assets': assets, 'route': report['route'],
        'summary': {k: report[k] for k in ['status', 'engineering_route_passed', 'frames', 'simulation_s',
            'wall_s', 'checks', 'cleanup_final_speed_mps', 'formal_quality_passed']},
        'provenance': {'source_report_sha256': sha(run/'report.json'),
            'source_contract_sha256': sha(run/'contract.json'),
            'models': read(run/'contract.json')['models'], 'model_outputs_unchanged': True,
            'not_live_inference': True, 'other_actor_GT_in_main_view': False,
            'JPEG_is_lossy_preview_only': True, 'map_and_ego_pose_are_explicit_priors': True,
            'only_this_named_route_verified': True,
            'scenario': 'Independent test director held lamps red then green; driver received RGB only.'}}
    write(SITE/'data/signal-turn.json', payload)


def main():
    perception()
    signal_turn()
    chapters = [
        {'id': 'perception', 'label': '01 · 車、機車、行人與短期記憶', 'file': 'data/perception.json',
         'camera': 'front_left', 'kind': 'perception',
         'description': '完整12秒感知序列 · 已知訓練場景 · 不含自主駕駛。彩色為當前模型觀測，金色線框為暫失記憶；物件由八鏡頭融合，未必全在所選鏡頭內。',
         'bookmarks': [{'index': 0, 'label': '同幀多物件'}, {'index': 2, 'label': '暫失 0.2 秒'},
            {'index': 4, 'label': '重現／記憶 0.6 秒'}, {'index': 5, 'label': '過期移除'}]},
        {'id': 'signal-turn', 'label': '02 · 紅燈停等 → 綠燈 → 左轉抵達', 'file': 'data/signal-turn.json',
         'camera': 'front', 'kind': 'drive',
         'description': '完整47.2秒真閉環紀錄 · 50.8m左轉路線。測試環境控制燈色，駕駛只讀RGB；保留綠燈後較慢起步的原始過程。',
         'bookmarks': [{'index': 20, 'label': '紅燈減速'}, {'index': 60, 'label': '靜止停等'},
            {'index': 97, 'label': 'RGB辨識綠燈'}, {'index': 150, 'label': '重新起步'},
            {'index': 210, 'label': '左轉中'}, {'index': 235, 'label': '抵達終點'}]},
        {'id': 'pedestrian', 'label': '03 · 原展示：行人避讓抵達', 'file': 'data/route.json',
         'camera': 'front', 'kind': 'drive',
         'description': '完整13.4秒真閉環紀錄 · 原28m行人避讓路線。',
         'bookmarks': [{'index': 0, 'label': '起步'}, {'index': 18, 'label': '行人避讓'},
            {'index': 66, 'label': '抵達終點'}]}
    ]
    for chapter in chapters:
        chapter['sha256'] = sha(SITE/chapter['file'])
    write(SITE/'data/showcases.json', {'schema_version': 1, 'chapters': chapters})
    print('Exported 60 perception + 236 signal/turn frames; original 67 preserved.')


if __name__ == '__main__':
    main()
