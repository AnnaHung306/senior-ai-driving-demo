"""Chunk-resumable CPU RPN coverage on every original val bicycle-positive view.

Measures proposal coverage only, not final detection accuracy or acceptance.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import torchvision
from torchvision.models.detection.rpn import RegionProposalNetwork,concat_box_prediction_layers
from predict_sunny import load_model
from senior_ai.sunny_data import sha,png
from senior_ai.detection_metrics import iou
from senior_ai.core import write_json

BUDGETS=[(500,150),(1000,500),(2000,1000)]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def summarize(records):
    results={}
    for pre,post in BUDGETS:
        name=f'{pre}_{post}';groups={}
        for frame in records:
            for obj in frame['targets']:
                for key in ['all',obj['size'],'truncated' if obj['truncated'] else 'not_truncated','scene/'+frame['scene_id']]:
                    c=groups.setdefault(key,Counter());c['GT_targets']+=1
                    for threshold in [.5,.7]:c[f'covered_IoU{threshold}']+=int(obj['max_proposal_IoU'][name]>=threshold)
        results[name]={'pre_per_level':pre,'post_total':post,'groups':{key:{**counts,
            'coverage_IoU50':counts['covered_IoU0.5']/counts['GT_targets'],'coverage_IoU70':counts['covered_IoU0.7']/counts['GT_targets']} for key,counts in groups.items()}}
    return results


@torch.inference_mode()
def proposal_grid(model,tensor,check_reference=False):
    images,_=model.transform([tensor],None);features=model.backbone(images.tensors);maps=list(features.values())
    objectness,deltas=model.rpn.head(maps);anchors=model.rpn.anchor_generator(images,maps)
    per_level=[level[0].numel() for level in objectness]
    objectness,deltas=concat_box_prediction_layers(objectness,deltas)
    decoded=model.rpn.box_coder.decode(deltas.detach(),anchors).view(1,-1,4);out={}
    for pre,post in BUDGETS:
        model.rpn._pre_nms_top_n['testing']=pre;model.rpn._post_nms_top_n['testing']=post
        proposals,_=model.rpn.filter_proposals(decoded,objectness,images.image_sizes,per_level)
        out[f'{pre}_{post}']=proposals[0]
    if check_reference:
        model.rpn._pre_nms_top_n['testing']=500;model.rpn._post_nms_top_n['testing']=150
        original,_=model.rpn(images,features,None)
        if not torch.equal(out['500_150'],original[0]):raise ValueError('Manual component orchestration differs from real RPN forward')
    return images.image_sizes[0],out


def main(args):
    root=args.root.resolve();out=args.output.resolve()
    if out.is_relative_to(root) or out.is_relative_to(args.checkpoint.resolve().parent):raise ValueError('Write outside immutable data/snapshot')
    if out.exists() and not args.resume:raise FileExistsError('Exact-contract --resume or new output required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);model,task,saved=load_model(args.checkpoint,'cpu')
    if task!='detection':raise ValueError('Detector required')
    for rel,digest in saved['provenance']['source']['files'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Model source differs')
    for file,key in [('audit.json','audit_sha256'),('metadata_inventory.json','metadata_inventory_sha256'),('image_inventory.jsonl','image_inventory_sha256')]:
        if sha(root/file)!=saved['provenance']['data'][key]:raise ValueError('Main signature differs')
    metadata={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
    if sha(root/'training/validation.jsonl')!=metadata['training/validation.jsonl']:raise ValueError('Validation index changed')
    rows=[json.loads(line) for line in (root/'training/validation.jsonl').read_text().splitlines()]
    if any(r['split']!='validation' for r in rows):raise ValueError('Original validation only')
    selected=[]
    for index,row in enumerate(rows):
        path=(root/row['detection']).resolve()
        if not path.is_relative_to(root) or sha(path)!=metadata[row['detection']]:raise ValueError('Audited GT changed')
        objects=[]
        for obj in read(path)['positive_objects']:
            if obj['category_id']!=3:continue
            b=obj['visible_bbox_xyxy'];area=(b[2]-b[0])*(b[3]-b[1])
            objects.append({'box':b,'visible_pixels':obj['visible_pixels'],'truncated':b[0]==0 or b[1]==0 or b[2]==640 or b[3]==360,
                'size':'small_lt_32_squared' if area<1024 else 'medium_32_to_96_squared' if area<9216 else 'large_ge_96_squared'})
        if objects:selected.append({'index':index,'sample':row,'targets':objects})
    reference=args.checkpoint.parent/'best_validation.json';receipt=read(args.checkpoint.parent/'snapshot_receipt.json')
    if receipt['best_sha256']!=sha(args.checkpoint) or receipt['copied_metadata']['best_validation.json']!=sha(reference):raise ValueError('Snapshot reference binding differs')
    expected=read(reference)['per_class']['bicycle']['ground_truth_objects']
    if sum(len(r['targets']) for r in selected)!=expected:raise ValueError('All positive GT count differs from canonical full validation')
    inventory={r['path']:r['sha256'] for r in map(json.loads,(root/'image_inventory.jsonl').read_text().splitlines())}
    contract={'checkpoint_sha256':sha(args.checkpoint),'reference_sha256':sha(reference),'tool_sha256':sha(Path(__file__)),
        'RPN_dependency_sha256':sha(Path(inspect.getsourcefile(RegionProposalNetwork))),'main_audit_sha256':saved['provenance']['data']['audit_sha256'],
        'selection_sha256':hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),'budgets':BUDGETS,
        'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'CPU_threads':2,'GT_targets':expected,'views':len(selected),
        'scope':'All original validation bicycle-positive images only, CPU FP32 RPN coverage. No bicycle-negative images, final detector PR, model/gate changes, or heldout.'}
    # JSON roundtrip normalizes tuple budgets for an exact-contract resume.
    contract=json.loads(json.dumps(contract));out.mkdir(parents=True,exist_ok=True)
    if (out/'contract.json').exists():
        if read(out/'contract.json')!=contract:raise ValueError('Resume contract changed')
    else:write_json(out/'contract.json',contract)
    signature=sha(out/'contract.json');chunks=out/'chunks';chunks.mkdir(exist_ok=True);records=[];baseline_verified=False
    for start in range(0,len(selected),20):
        subset=selected[start:start+20];path=chunks/f'{start//20:04d}.json';indices=[r['index'] for r in subset]
        for row in subset:
            s=row['sample'];rgb_path=(root/s['rgb']).resolve();gt_path=(root/s['detection']).resolve()
            if not rgb_path.is_relative_to(root) or sha(rgb_path)!=inventory[s['rgb']] or sha(gt_path)!=metadata[s['detection']]:raise ValueError('Selected bytes changed')
        if path.exists():
            cached=read(path)
            if cached['contract_sha256']!=signature or cached['indices']!=indices:raise ValueError('Cached chunk changed')
            records.extend(cached['records']);baseline_verified|=cached['real_RPN_forward_exact_verified'];continue
        computed=[]
        for row in subset:
            s=row['sample'];rgb=png(root/s['rgb'])
            size,proposals=proposal_grid(model,torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255,not baseline_verified)
            baseline_verified=True;h,w=size;objects=[]
            for obj in row['targets']:
                b=obj['box'];scaled=[b[0]*w/640,b[1]*h/360,b[2]*w/640,b[3]*h/360]
                objects.append({**obj,'max_proposal_IoU':{name:float(max(iou(scaled,p.tolist()),default=0)) for name,p in proposals.items()}})
            computed.append({'index':row['index'],'scene_id':s['scene_id'],'group_id':s['group_id'],'camera':s['camera'],'frame':s['frame'],'RGB':s['rgb'],'targets':objects})
        write_json(path,{'contract_sha256':signature,'indices':indices,'real_RPN_forward_exact_verified':baseline_verified,'records':computed});records.extend(computed)
        write_json(out/'progress.json',{'saved_views':len(records),'total_views':len(selected),'results':summarize(records)})
        print(json.dumps({'saved_bicycle_positive_views':len(records),'total':len(selected)}),flush=True)
    for name,value in model.state_dict().items():
        if not torch.equal(value.cpu(),saved['model'][name].cpu()):raise ValueError('Learned tensor changed')
    write_json(out/'report.json',{'status':'completed_all_bicycle_positive_validation_CPU_RPN_survey','views':len(records),'GT_targets':expected,
        'results':summarize(records),'real_RPN_forward_exact_verified':baseline_verified,'learned_tensors_identical':True,
        'limits':'Proposal coverage uses per-target independent IoU existence, not one-to-one detector AP/PR. Positive-only validation, not unseen data or full-system recall. Higher proposal coverage does not guarantee ROI classification/regression success.'})
    print(json.dumps({'status':'completed_CPU_component_diagnostic_only','views':len(records),'GT_targets':expected,'all':{k:v['groups']['all'] for k,v in summarize(records).items()}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');main(p.parse_args())
