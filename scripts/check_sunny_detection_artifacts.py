"""CPU/file-only detector verification against canonical validation labels.

No model inference, raw-image re-audit, GPU, simulator or heldout evaluation.
Success means consistency, never automatic quality acceptance.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from check_sunny_segmentation_artifacts import selected_epoch,require,read,sha
from senior_ai.detection_metrics import evaluate_detection
from senior_ai.perception_quality import quality
from senior_ai.sunny_data import DETECTION_CLASSES
from senior_ai.dataset import safe_path


def expected_targets(labels):
    boxes=np.asarray([o['visible_bbox_xyxy'] for o in labels['positive_objects']],dtype=np.float32).reshape(-1,4).tolist()
    targets=[]
    for obj,box in zip(labels['positive_objects'],boxes):
        b=obj['visible_bbox_xyxy']
        targets.append({'box':box,'label':int(obj['category_id']),'visible_pixels':int(obj['visible_pixels']),
            'truncated':b[0]==0 or b[1]==0 or b[2]==640 or b[3]==360,
            'light_state_class':int(obj.get('light_state_class',-1))})
    ignored=np.asarray([o['bbox_xyxy'] for o in labels['ignore_regions']],dtype=np.float32).reshape(-1,4).tolist()
    return targets,ignored


def validate_predictions(predictions):
    require(len(predictions)<=100,'Prediction max100 exceeded')
    for p in predictions:
        b=p['box'];s=p['score'];label=p['label']
        require(type(label)is int and label in DETECTION_CLASSES.values(),'Invalid prediction label')
        require(isinstance(s,(int,float)) and not isinstance(s,bool) and math.isfinite(s) and 0<=s<=1,'Invalid prediction score')
        require(len(b)==4 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in b),'Invalid prediction coordinates')
        require(0<=b[0]<b[2]<=640 and 0<=b[1]<b[3]<=360,'Prediction outside original640x360 coordinates')


def check(run,root):
    state=read(run/'state.json');config=read(run/'config.json');provenance=read(run/'provenance.json')
    require(state['status']=='completed' and state['task']=='detection','Completed detector required')
    require(state['checkpoint_reload_identical'],'Checkpoint reload failed')
    hashes={}
    for name,expected in [('best.pt',state['best_sha256']),('last.pt',state['last_sha256']),('config.json',provenance['config_sha256']),
        ('validation.json',state['validation_sha256']),('validation_frames.jsonl',state['validation_frames_sha256'])]:
        hashes[name]=sha(run/name);require(hashes[name]==expected,'Run artifact changed: '+name)
    source=provenance['source']['files']
    require(hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()==provenance['source']['sha256'],'Source manifest differs')
    for name,digest in source.items():
        p=safe_path(run/'source',name);require(sha(p)==digest,'Frozen source differs: '+name)
    for name in ['src/senior_ai/detection_metrics.py','src/senior_ai/sunny_data.py','src/senior_ai/dataset.py']:
        require(sha(ROOT/name)==source[name],'Use matching frozen metric/data implementation: '+name)
    quality_source='src/senior_ai/perception_quality.py'
    if quality_source in source:require(sha(ROOT/quality_source)==source[quality_source],'Quality implementation differs')
    for name,key in [('audit.json','audit_sha256'),('metadata_inventory.json','metadata_inventory_sha256')]:
        require(sha(root/name)==provenance['data'][key],'Main data metadata differs: '+name)
    metadata={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
    require(sha(root/'training/validation.jsonl')==metadata['training/validation.jsonl'],'Canonical validation index changed')
    canonical=[json.loads(line) for line in (root/'training/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    frames=[json.loads(line) for line in (run/'validation_frames.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(frames)==len(canonical)>0,'Validation frame count differs')
    require([f['index'] for f in frames]==list(range(len(canonical))),'Validation index order/coverage differs')
    for frame,row in zip(frames,canonical):
        require(row['split']=='validation' and all(frame.get(k)==v for k,v in row.items()),'Canonical validation row identity differs')
        path=safe_path(root,row['detection']);require(sha(path)==metadata[row['detection']],'Canonical validation GT changed')
        targets,ignored=expected_targets(read(path))
        require(frame['targets']==targets,'Serialized GT/size/truncation/phase metadata differs')
        require(frame['ignore_boxes']==ignored,'Serialized ignore regions differ')
        validate_predictions(frame['predictions'])
    # This checks serialization and aggregation, using the exact frozen metric
    # implementation already covered by hand-computed metric regression tests.
    result=evaluate_detection(frames,DETECTION_CLASSES)
    result['by_scene']={name:evaluate_detection([f for f in frames if f['scene_id']==name],DETECTION_CLASSES)
        for name in sorted({f['scene_id'] for f in frames})}
    require(result==read(run/'validation.json'),'Recomputed full/split/strata AP/PR differs from saved report')
    epochs=[json.loads(line) for line in (run/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    require([r['epoch'] for r in epochs]==list(range(config['detection']['epochs'])),'Epoch history incomplete')
    chosen=selected_epoch(epochs,config.get('selection_policy','score_only'))
    require(chosen['epoch']==state['best_epoch'] and chosen['validation_score']==state['best_score']==result['mAP50_95'],'Selected checkpoint epoch/score differs')
    stats=read(run/'dataset_statistics.json');steps=math.ceil(stats['train_views']/config['detection']['batch_size']);last=0
    for row in epochs:
        require(row['updates']-last==steps-row['skipped_fully_ignored_batches'],'Optimizer update history differs')
        require(math.isfinite(row['mean_loss']) and math.isfinite(row['max_gradient_norm']),'Nonfinite epoch diagnostic')
        last=row['updates']
    require(last==state['updates'],'Final update count differs')
    require(stats['validation_views']==len(frames),'Validation sample count differs from training stats')
    q=quality('detection',result,config['quality_gates'])
    return {'artifact_integrity_passed':True,'checked_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),
        'checker_sha256':sha(Path(__file__)),'selection_helper_sha256':sha(ROOT/'scripts/check_sunny_segmentation_artifacts.py'),
        'quality_source_sha256':sha(ROOT/quality_source),'quality_scope':'Recheck saved config gates; older runs predate the extracted quality module',
        'scope':__doc__,'hashes':hashes,'source_snapshot_files_verified':len(source),'canonical_validation_frames_verified':len(frames),
        'all_validation_GT_and_ignore_bytes_verified':True,'full_and_by_scene_metrics_recomputed_exact':True,
        'updates':last,'best_epoch_zero_based':state['best_epoch'],'validation_quality_passed':q['passed'],
        'failed_validation_checks':[k for k,v in q['checks'].items() if not v],
        'per_class':result['per_class'],'mAP50':result['mAP50'],'mAP50_95':result['mAP50_95'],
        'lockbox_marker_exists':(run/'lockbox_started.json').exists()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    run,root,out=args.run.resolve(),args.root.resolve(),args.output.resolve()
    require(not out.is_relative_to(run) and not out.is_relative_to(root),'Write outside immutable run/data')
    report=check(run,root);out.parent.mkdir(parents=True,exist_ok=True)
    pending=out.with_name(out.name+'.tmp');pending.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');pending.replace(out)
    print(json.dumps({k:report[k] for k in ['artifact_integrity_passed','canonical_validation_frames_verified','validation_quality_passed','failed_validation_checks']}))
