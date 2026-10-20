"""Recount saved BEV results and verify immutable artifacts; no model/GPU evaluation."""
import argparse,json,hashlib,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evaluate_phase4_pilot import road_counts,line_counts,finalize
from senior_ai.core import write_json
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(args):
    run=args.run;ev=args.evaluation;state=read(run/'state.json');report=read(ev/'report.json');contract=read(run/'contract.json');config=read(run/'config.json')
    if state['bundles']!=32 or state['sequences']!=4:raise ValueError('Expected fixed pilot')
    frozen_config=run/'source/configs/phase4_bev_v1.json'
    assert sha(frozen_config)==contract['config_sha256'] and config==read(frozen_config)
    assert sha(ROOT/'scripts/evaluate_phase4_pilot.py')==report['evaluator_sha256']
    for rel,digest in contract['sources'].items():assert sha(run/'source'/rel)==sha(ROOT/rel)==digest,rel
    for item in contract['models'].values():assert sha(ROOT/item['path'])==item['sha256']
    total={name:Counter() for name in ['front','single','temporal']};scenes=defaultdict(lambda:{n:Counter() for n in total});history={};path_modes=Counter();output_files={}
    for record in report['records']:
        folder=run/record['folder'];metadata=read(folder/'frame.json');b=np.load(folder/'bev.npz');gt=np.load(ev/record['folder']/'GT_reference.npz')
        assert metadata['control_issued'] is False and metadata['timestamp_s']>history.get(record['scene_id'],-float('inf'));history[record['scene_id']]=metadata['timestamp_s']
        assert metadata['motion']['motion_source']=='RGB_ground_features_only' and 0<=metadata['motion']['history_frames_used']<=2
        for name in total:
            assert np.isfinite(b[name]).all() and (b[name]>=0).all() and (b[name]<=1.000001).all()
            observed=b['front_observed'] if name=='front' else b['observed']
            c={**road_counts(b[name],observed,gt['road'],gt['visible_ground']),**line_counts(b[name],observed,gt['line'],gt['visible_ground'])};assert finalize(c)==record['metrics'][name]
            total[name].update(c);scenes[record['scene_id']][name].update(c)
        for obj in metadata['objects']:
            sources=obj['sources'];assert len({x['camera'] for x in sources})==len(sources);assert all(x['class_name']==obj['class_name'] for x in sources)
            assert obj['maximum_source_separation_m']<=config['object_merge_distance_m']+1e-8
            np.testing.assert_allclose(obj['xy_m'],np.average([x['xy_m'] for x in sources],axis=0,weights=[x['score'] for x in sources]),atol=1e-10)
        path=metadata['path'];assert path['control_issued'] is False and np.isfinite(np.asarray(path['points_ego_m'])).all();path_modes[path['source']]+=1
        if 'fixed_width_prior' in path['source']:assert path['width_prior_m']==3.5 and 'assumption' in path
    assert {n:finalize(v) for n,v in total.items()}==report['metrics']
    assert {s:{n:finalize(v) for n,v in modes.items()} for s,modes in scenes.items()}==report['by_scene']
    q=read(args.review/'catalog.json')
    assert q['run_contract_sha256']==sha(run/'contract.json') and q['evaluation_report_sha256']==sha(ev/'report.json')
    for page in q['pages']:
        assert sha(args.review/page['path'])==page['sha256']
        for source in page['sources']:assert sha(Path(source['source']))==source['sha256']
    for case in q['full_resolution_cases']:
        assert sha(Path(case['prediction_panel']))==case['prediction_sha256'];assert sha(Path(case['GT_panel']))==case['GT_sha256']
    assert read(args.boundary/'report.json')['status']=='passed_expected_rejections'
    for p in sorted(run.rglob('*')):
        if p.is_file():output_files[str(p.relative_to(run))]=sha(p)
    write_json(args.output,{'status':'passed_artifact_integrity_not_formal_quality','tool_sha256':sha(Path(__file__)),'bundles_recounted':32,'sources_verified':len(contract['sources']),'per_frame_full_scene_counts_exact':True,'serialized_run_config_semantically_exact_to_frozen_source':True,'path_modes':dict(path_modes),'contact_pages_verified':len(q['pages']),'models_unchanged':True,'output_manifest':output_files,'limits':__doc__})
    print(json.dumps({'bundles_recounted':32,'sources_verified':len(contract['sources']),'pages_verified':len(q['pages']),'integrity':'passed'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--boundary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
