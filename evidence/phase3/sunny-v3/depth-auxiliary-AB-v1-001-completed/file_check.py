"""CPU stored-artifact verification only, no model inference or heldout scoring."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.segmentation_metrics import metrics
from senior_ai.sunny_data import SEGMENTATION_NAMES
from senior_ai.road_boundaries import boundary_metrics
from senior_ai.perception_quality import quality
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
c=read(P/'contract.json');cfg=read(P/'config.json') if (P/'config.json').exists() else read(ROOT/'configs/depth_ablation_v1_1.json')
for name,digest in c['sources'].items():assert sha(ROOT/name)==digest,name
parent=ROOT/cfg['parent_run'];pc=read(parent/'config.json');parrows=rows(parent/'validation_frames.jsonl')
canonical=rows(ROOT/cfg['dataset']/'training/validation.jsonl')
assert len(canonical)==len(parrows)==3600
for filename,key in [('best.pt','parent_best_sha256'),('config.json','parent_config_sha256'),('state.json','parent_state_sha256'),('provenance.json','parent_provenance_sha256'),('validation.json','parent_validation_sha256'),('validation_frames.jsonl','parent_frames_sha256'),('dataset_statistics.json','parent_statistics_sha256')]:assert sha(parent/filename)==cfg[key],filename
summary={};digests=[]
for arm in ['control','depth']:
    d=P/arm;s=read(d/'state.json');r=read(d/'validation.json');frames=rows(d/'validation_frames.jsonl');ac=read(d/'config.json')
    assert s['status']=='completed_research_arm' and s['updates']==1712 and s['sampled_train_views']==13696
    assert not s['heldout_opened'] and s['checkpoint_reload_identical']
    assert read(d/'initial_GPU_replay.json')['exact_parent_full_validation']
    for filename,key in [('final.pt','final_checkpoint_sha256'),('validation.json','validation_sha256'),('validation_frames.jsonl','validation_frames_sha256')]:assert sha(d/filename)==s[key],filename
    assert ac['quality_gates']==pc['quality_gates'] and len(frames)==3600
    total=np.zeros((17,17),dtype=np.int64);by_scene={};border=Counter()
    for index,(f,original,row) in enumerate(zip(frames,parrows,canonical)):
        assert f['index']==index and all(f.get(k)==v for k,v in row.items())
        m=np.asarray(f['confusion_true_rows_predicted_columns']);assert m.dtype.kind in 'iu' and m.shape==(17,17) and (m>=0).all()
        assert np.array_equal(m.sum(1),np.asarray(original['confusion_true_rows_predicted_columns']).sum(1))
        assert metrics(m,SEGMENTATION_NAMES)=={k:f[k] for k in metrics(m,SEGMENTATION_NAMES)}
        total+=m;by_scene.setdefault(f['scene_id'],np.zeros_like(m))[:]+=m
        for key in ['gt_pixels','predicted_pixels','matched_gt_within_2px','matched_prediction_within_2px']:border[key]+=f['visible_margin'][key]
    recomputed=metrics(total,SEGMENTATION_NAMES);recomputed['visible_margin']=boundary_metrics(border)
    recomputed['by_scene']={k:metrics(v,SEGMENTATION_NAMES) for k,v in by_scene.items()}
    assert recomputed==r and quality('segmentation',r,pc['quality_gates'])==s['original_quality']
    digests.append(s['RGB_semantic_augmentation_sha256'])
    summary[arm]={'validation_frames':len(frames),'full_and_scene_confusion_metrics_exact':True,'boundary_counts_exact':True,'canonical_index_and_parent_GT_support_exact':True,'original_quality':s['original_quality'],'checkpoint_sha256':s['final_checkpoint_sha256']}
assert digests[0]==digests[1] and read(P/'comparison.json')['equal_augmentation_bytes']
g=P/'validation_comparison';catalog=read(g/'catalog.json')
for item in catalog['cases']+catalog['pages']:assert sha(g/item['path'])==item['sha256']
report={'status':'passed_artifact_consistency_not_quality','checked_utc':datetime.now(timezone.utc).isoformat(),'tool_sha256':sha(Path(__file__)),'sources_verified':len(c['sources']),'arms':summary,'equal_augmentation_sha256':digests[0],'QA_pages_verified':len(catalog['pages']),'limits':__doc__+' Uses stored per-frame counts and original parent GT supports; no independent full pixel-level prediction replay or new depth-quality evaluation.'}
(P/'file_integrity.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','sources_verified','QA_pages_verified']}))
