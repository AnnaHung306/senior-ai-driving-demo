"""Source and RGB provenance integrity seal for the longer frozen replay, not GT quality."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.dataset import read_sensor_frame
from senior_ai.core import write_json

def main(a):
    run=a.run;contract=read(run/'contract.json');state=read(run/'state.json');source=read(a.inputs/'sequence_inputs.json');inventory=read(a.inputs/'rgb_only_inventory.json')
    assert state['bundles']==240 and len(state['records'])==240 and state['source_integrity_exact'] is True
    assert contract['RGB_inventory_sha256']==sha(a.inputs/'rgb_only_inventory.json') and contract['input_list_sha256']==sha(a.inputs/'sequence_inputs.json')
    for rel,digest in contract['sources'].items():assert sha(run/'source'/rel)==digest
    for rel,digest in inventory.items():assert sha(a.inputs/'rgb_only'/rel)==digest
    if (run/'config.json').exists():config=read(run/'config.json')
    rows=[];expected=[p for s in source['sequences'] for p in s['manifests']]
    for record,path in zip(state['records'],expected,strict=True):
        f=read_sensor_frame(a.inputs/'rgb_only',path);folder=Path(record['folder'])
        if (run/folder/'world.json').exists():
            from senior_ai.world_display import strict_world
            w=read(run/folder/'world.json')
            with np.load(run/folder/'occupancy.npz',allow_pickle=False) as n:strict_world(w,n['semantic'])
        else:
            w=read(run/folder/'frame.json');assert w['manifest_sha256']==sha(a.inputs/'rgb_only'/path)
            assert w['motion']['motion_source']=='RGB_ground_features_only'
            with np.load(run/folder/'bev.npz',allow_pickle=False) as n:
                for key in ['single','front','temporal']:assert np.isfinite(n[key]).all() and (n[key]>=0).all() and (n[key]<=1.00001).all()
        assert w['scene_id']==f['scene_id'] and w['frame']==f['frame'] and w['timestamp_s']==f['timestamp_s'] and w['control_issued'] is False
        rows.append({'scene_id':f['scene_id'],'frame':f['frame']})
    manifest={p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file()}
    write_json(a.output,{'status':'passed_artifact_integrity_not_formal_quality','run_contract_sha256':sha(run/'contract.json'),'output_manifest':manifest,
        'frames':len(rows),'RGB_staging_files':len(inventory),'tool_sha256':sha(Path(__file__)),'GT_quality_evaluated':False,
        'scope':'Long Phase7 RGB replay identity/finite/provenance check only; no new BEV quality claim'})
    print({'status':'sealed','frames':len(rows),'files':len(manifest)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
