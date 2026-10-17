"""Chunked CPU train/validation diagnostic, separate from official GPU gates.

All cameras at every Nth saved bundle per scene; no heldout indexes. Training
metrics describe seen data. CPU FP32 validation can also be compared with the
original GPU BF16 metrics on exactly the same selected images.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from check_sunny_segmentation_artifacts import check
from predict_sunny import load_model
from survey_sunny_materials import write
from senior_ai.dataset import safe_path
from senior_ai.sunny_data import sha,png,segmentation_target,SEGMENTATION_NAMES
from senior_ai.segmentation_metrics import confusion,metrics
from senior_ai.road_boundaries import boundary_counts,boundary_metrics


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def summarize(records,identity,expected,status):
    groups=defaultdict(list)
    for record in records:
        groups[record['split']].append(record)
        groups[record['split']+'/'+record['group_id']].append(record)
    reports={}
    for name,rows in groups.items():
        matrix=sum((np.array(r['confusion'],dtype=np.int64) for r in rows),np.zeros((17,17),dtype=np.int64))
        margin={key:sum(r['margin_counts'][key] for r in rows) for key in rows[0]['margin_counts']}
        result={**metrics(matrix,SEGMENTATION_NAMES),'visible_margin':boundary_metrics(margin),'completed_views':len(rows)}
        if name=='validation':
            gpu=sum((np.array(r['GPU_reference_confusion'],dtype=np.int64) for r in rows),np.zeros((17,17),dtype=np.int64))
            result['original_GPU_on_same_images']=metrics(gpu,SEGMENTATION_NAMES)
        reports[name]=result
    return {'status':status,'identity':identity,'completed_views':len(records),'expected_views':expected,
            'scope':'CPU FP32 diagnostic only. Seen train and selected validation images; not official acceptance, unseen-test performance, or independent-trial generalization. No model/gate changes or GPU access.',
            'results':reports}


def main(args):
    run,root,out=args.run.resolve(),args.root.resolve(),args.output.resolve()
    if args.bundle_stride<1 or out.is_relative_to(run) or out.is_relative_to(root):raise ValueError('Positive stride and separate output required')
    health=check(run);provenance=read(run/'provenance.json')
    for filename,key in [('audit.json','audit_sha256'),('visual_review.json','review_sha256'),('image_inventory.jsonl','image_inventory_sha256'),('metadata_inventory.json','metadata_inventory_sha256')]:
        if sha(root/filename)!=provenance['data'][key]:raise ValueError('Bound inventory changed')
    for relative,digest in provenance['source']['files'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Runtime differs from frozen source')
    metadata={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
    selected=[]
    for split in ['train','validation']:
        filename=f'training/{split}.jsonl'
        if sha(root/filename)!=metadata[filename]:raise ValueError('Canonical index changed')
        rows=[json.loads(line) for line in (root/filename).read_text(encoding='utf-8').splitlines()]
        if any(r['split']!=split for r in rows):raise ValueError('Train/validation only')
        scene_frames={scene:sorted({r['frame'] for r in rows if r['scene_id']==scene})[::args.bundle_stride] for scene in {r['scene_id'] for r in rows}}
        selected.extend({**r,'split_index':i} for i,r in enumerate(rows) if r['frame'] in scene_frames[r['scene_id']])
    needed={r[key] for r in selected for key in ['rgb','semantic','instance']}
    inventory={}
    with (root/'image_inventory.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line)
            if item['path'] in needed:inventory[item['path']]=item
    if set(inventory)!=needed:raise ValueError('Missing selected raw inventory')
    gpu_frames={r['index']:r for r in [json.loads(line) for line in (run/'validation_frames.jsonl').read_text(encoding='utf-8').splitlines()]}
    identity={'tool_sha256':sha(Path(__file__)),'io_helper_sha256':sha(ROOT/'scripts/survey_sunny_materials.py'),
              'checkpoint_sha256':health['hashes']['best.pt'],'data_audit_sha256':provenance['data']['audit_sha256'],
              'bundle_stride':args.bundle_stride,'chunk_size':20,'CPU_threads':4,
              'selection_sha256':hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),
              'GPU_validation_frames_sha256':health['hashes']['validation_frames.jsonl']}
    out.mkdir(parents=True,exist_ok=True);chunks=out/'chunks';chunks.mkdir(exist_ok=True)
    report_path=out/'report.json'
    if report_path.exists() and read(report_path)['identity']!=identity:raise ValueError('Resume identity differs')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model,_,_=load_model(run/'best.pt','cpu');records=[];new_chunks=0
    for start in range(0,len(selected),20):
        batch=selected[start:start+20];path=chunks/f'{start//20:04d}.json'
        cached=read(path) if path.exists() else None
        if cached and (cached['identity']!=identity or cached['samples']!=batch):raise ValueError('Cached chunk identity changed')
        # Verify bytes even when resuming saved results, without recomputing logits.
        for sample in batch:
            for key in ['rgb','semantic','instance']:
                source=safe_path(root,sample[key]);item=inventory[sample[key]]
                if sha(source)!=item['sha256'] or source.stat().st_size!=item['bytes']:raise ValueError('Selected raw bytes changed')
        if cached:
            records.extend(cached['records']);continue
        computed=[]
        for sample in batch:
            rgb=png(safe_path(root,sample['rgb']))
            target=segmentation_target(png(safe_path(root,sample['semantic']))[...,0],png(safe_path(root,sample['instance'])),sample['verified_obstacle_actor_ids'])
            with torch.inference_mode():
                prediction=model(torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255)[0].argmax(0).numpy().astype(np.uint8)
            matrix=confusion(target,prediction,17)
            record={key:sample[key] for key in ['split','group_id','scene_id','camera','frame','rgb','split_index']}
            record.update(confusion=matrix.tolist(),margin_counts=boundary_counts(target,prediction))
            if sample['split']=='validation':
                gpu=gpu_frames[sample['split_index']]
                if gpu['rgb']!=sample['rgb'] or not np.array_equal(matrix.sum(1),np.array(gpu['confusion_true_rows_predicted_columns']).sum(1)):
                    raise ValueError('CPU target differs from original GPU target')
                record['GPU_reference_confusion']=gpu['confusion_true_rows_predicted_columns']
            computed.append(record)
        write(path,{'identity':identity,'samples':batch,'records':computed});records.extend(computed);new_chunks+=1
        write(report_path,summarize(records,identity,len(selected),'in_progress'))
        print(json.dumps({'saved_views':len(records),'total':len(selected),'new_chunks':new_chunks}),flush=True)
        if args.max_new_chunks and new_chunks>=args.max_new_chunks:
            write(report_path,summarize(records,identity,len(selected),'stopped_at_chunk_budget'));return
    write(report_path,summarize(records,identity,len(selected),'completed_diagnostic'))
    print(json.dumps({'status':'completed_diagnostic','views':len(records)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bundle-stride',type=int,default=10)
    p.add_argument('--max-new-chunks',type=int,default=0);main(p.parse_args())
