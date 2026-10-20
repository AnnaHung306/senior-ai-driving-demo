"""Export source-bound predicted WorldFrames and raw RGB for a local viewer."""
import argparse
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from senior_ai.world_display import sha, read, relative_path, display_payload
from senior_ai.core import write_json
from senior_ai.dataset import read_sensor_frame, safe_path


def main(args):
    run = args.run.resolve();parent = args.phase4.resolve();rgb = args.rgb_root.resolve();out = args.output.resolve()
    if out.exists():raise FileExistsError('New viewer bundle required')
    proof = read(args.integrity);parent_proof = read(args.phase4_integrity);state = read(run/'state.json')
    if proof['status'] != 'passed_integrity_not_formal_quality' or proof['run_contract_sha256'] != sha(run/'contract.json'):
        raise ValueError('Bound completed Phase5 integrity proof required')
    if proof['evaluation_report_sha256'] != sha(args.evaluation_report):raise ValueError('Evaluation source mismatch')
    for rel, digest in proof['output_manifest'].items():
        if sha(safe_path(run, rel)) != digest:raise ValueError('Phase5 output changed: '+rel)
    for rel, digest in parent_proof['output_manifest'].items():
        if sha(safe_path(parent, rel)) != digest:raise ValueError('Phase4 output changed: '+rel)
    inventory = read(args.rgb_inventory)
    actual = {p.relative_to(rgb).as_posix() for p in rgb.rglob('*') if p.is_file()}
    if actual != set(inventory) or any(not x.startswith(('inputs/', 'calibration/')) for x in actual):raise ValueError('RGB root contaminated')
    for rel, digest in inventory.items():
        if sha(safe_path(rgb, rel)) != digest:raise ValueError('RGB staging changed')
    contract = read(run/'contract.json')
    evaluation = read(args.evaluation_report);validation_quality = proof['split_results']['validation']['learned']
    if sha(parent/'contract.json') != contract['phase4_contract_sha256'] or sha(args.rgb_inventory) != contract['RGB_inventory_sha256']:
        raise ValueError('Parent/RGB provenance mismatch')
    out.mkdir(parents=True);files = {};frames = [];last = {};seen = set();bev_config = read(parent/'config.json')
    def save(relative, data):
        relative_path(relative);target = out/relative;target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():raise FileExistsError(target)
        target.write_bytes(data);files[relative] = sha(target);return relative
    for row in state['records']:
        folder = Path(row['folder']);world = read(run/folder/'world.json');key = world['scene_id']+'/'+str(world['frame'])
        if key in seen or world['timestamp_s'] <= last.get(world['scene_id'], -float('inf')):raise ValueError('Duplicate or reversed WorldFrame')
        seen.add(key);last[world['scene_id']] = world['timestamp_s']
        with np.load(run/folder/'occupancy.npz', allow_pickle=False) as a:occupancy = a['semantic']
        if sha(run/folder/'occupancy.npz') != world['occupancy']['sha256']:raise ValueError('Occupancy hash mismatch')
        if sha(parent/folder/'frame.json') != world['parent_frame_sha256']:raise ValueError('Phase4 frame mismatch')
        with np.load(parent/folder/'bev.npz', allow_pickle=False) as a:payload = display_payload(world, occupancy, a, bev_config)
        manifest = read_sensor_frame(rgb, world['parent_manifest']);cameras = {}
        for name, value in manifest['cameras'].items():
            relative = f"rgb/{world['scene_id']}/{name}/{world['frame']}.png"
            save(relative, safe_path(rgb, value['image']).read_bytes());cameras[name] = {'file': relative, 'sha256': files[relative]}
        binary = save(f'frames/{key}.bin', occupancy.tobytes(order='C'))
        payload.update({'occupancy': {'file': binary, 'sha256': files[binary], 'shape': list(occupancy.shape), 'dtype': 'uint8', 'order': 'C'},
                        'cameras': cameras, 'world_source_sha256': sha(run/folder/'world.json'), 'parent_bev_sha256': sha(parent/folder/'bev.npz')})
        relative = save(f'frames/{key}.json', (json.dumps(payload, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8'))
        frames.append({'key': key, 'scene_id': world['scene_id'], 'frame': world['frame'], 'timestamp_s': world['timestamp_s'],
                       'file': relative, 'sha256': files[relative], 'objects': len(payload['objects'])})
    scenes = []
    for name in dict.fromkeys(r['scene_id'] for r in frames):
        rows = [r for r in frames if r['scene_id'] == name]
        scenes.append({'id': name, 'frames': [r['key'] for r in rows], 'duration_s': rows[-1]['timestamp_s']-rows[0]['timestamp_s'],
                       'source_frequency_Hz': (len(rows)-1)/(rows[-1]['timestamp_s']-rows[0]['timestamp_s']) if len(rows)>1 else None})
    write_json(out/'bundle.json', {'schema_version': 1, 'source_mode': 'recorded_inference_replay', 'label': '晴天研究序列 · 已計算推論',
               'run_contract_sha256': sha(run/'contract.json'), 'checkpoint_sha256': contract['checkpoint_sha256'],
               'phase4_contract_sha256': sha(parent/'contract.json'), 'evaluation_report_sha256': sha(args.evaluation_report),
               'formal_model_quality_passed': False, 'frames': frames, 'scenes': scenes, 'file_inventory': files,
               'display_mapping': {'ego_to_three': '[y,z,-x]', 'yaw_y_three': '-yaw', 'size_three': '[width,height,length]'},
               'quality_context': {'scope': 'original_model_selection_validation_short_clips_not_heldout',
                                   'position_precision_at_2m': validation_quality['precision_at_2m'],
                                   'position_recall_at_2m': validation_quality['recall_at_2m'],
                                   'box_recall_IoU025': validation_quality['recall_IoU3D_025'],
                                   'ID_switches': evaluation['tracking']['ID_switches'],
                                   'matched_transitions': evaluation['tracking']['consecutive_matched_actor_transitions'],
                                   'quality_summary_source_sha256': sha(args.integrity)},
               'GT_geometry_exported': False, 'control_issued': False, 'exporter_sha256': sha(Path(__file__)),
               'adapter_sha256': sha(ROOT/'src/senior_ai/world_display.py')})
    print(json.dumps({'status': 'exported_verified_prediction_bundle', 'frames': len(frames), 'scenes': len(scenes), 'files': len(files), 'output': str(out)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['run', 'integrity', 'phase4', 'phase4-integrity', 'rgb-root', 'rgb-inventory', 'evaluation-report', 'output']:p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
