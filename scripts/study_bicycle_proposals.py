"""CPU-only fixed RPN-budget study on previously GT-selected validation views.

No training, checkpoint rewrite, deployment threshold change or heldout.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from predict_sunny import load_model
from senior_ai.sunny_data import sha,png
from senior_ai.detection_metrics import iou
from senior_ai.core import write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    if args.output.exists():raise FileExistsError('New study version required')
    root=args.root.resolve()
    if args.output.resolve().is_relative_to(root) or args.output.resolve().is_relative_to(args.checkpoint.resolve().parent):raise ValueError('Write outside data/checkpoint snapshot')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,task,saved=load_model(args.checkpoint,'cpu')
    if task!='detection':raise ValueError('Detector required')
    for rel,digest in saved['provenance']['source']['files'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen source differs')
    for file,key in [('audit.json','audit_sha256'),('metadata_inventory.json','metadata_inventory_sha256'),('image_inventory.jsonl','image_inventory_sha256')]:
        if sha(root/file)!=saved['provenance']['data'][key]:raise ValueError('Main signature differs')
    metadata={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
    if sha(root/'training/validation.jsonl')!=metadata['training/validation.jsonl']:raise ValueError('Canonical validation changed')
    canonical=[json.loads(line) for line in (root/'training/validation.jsonl').read_text().splitlines()]
    selection=read(args.selection);rows=[r for r in selection['records'] if r['sample']['split']=='validation']
    if selection['status']!='completed_selected_CPU_diagnostic_only' or selection['checkpoint_sha256']!=sha(args.checkpoint) or not rows:raise ValueError('Bound completed CPU selection required')
    for row in rows:
        s=row['sample']
        if {k:v for k,v in s.items() if k!='split_index'}!=canonical[s['split_index']]:raise ValueError('Selection differs from canonical validation')
    inventory={r['path']:r['sha256'] for r in map(json.loads,(root/'image_inventory.jsonl').read_text().splitlines())}
    budgets=[{'pre_per_level':500,'post_total':150},{'pre_per_level':1000,'post_total':500},{'pre_per_level':2000,'post_total':1000}]
    if model.rpn._pre_nms_top_n['testing']!=500 or model.rpn._post_nms_top_n['testing']!=150 or model.roi_heads.detections_per_img!=100:raise ValueError('Unexpected parent proposal/score protocol')
    args.output.mkdir(parents=True)
    write_json(args.output/'contract.json',{'checkpoint_sha256':sha(args.checkpoint),'selection_sha256':sha(args.selection),'tool_sha256':sha(Path(__file__)),
        'fixed_budgets_before_inference':budgets,'record_indices':[r['sample']['split_index'] for r in rows],
        'scope':'Selected validation CPU only. Original trained weights, score=.5 and max100 final detections unchanged; these runtime proposal budgets are exploratory and not adopted.'})
    captured={}
    def hook(module,inputs,output):
        captured['proposals']=output[0][0].detach().cpu();captured['size']=inputs[0].image_sizes[0]
    handle=model.rpn.register_forward_hook(hook);results=[]
    try:
        for budget in budgets:
            model.rpn._pre_nms_top_n['testing']=budget['pre_per_level'];model.rpn._post_nms_top_n['testing']=budget['post_total'];cases=[]
            for selected in rows:
                s=selected['sample'];path=(root/s['rgb']).resolve();gt=(root/s['detection']).resolve()
                if not path.is_relative_to(root) or sha(path)!=inventory[s['rgb']] or not gt.is_relative_to(root) or sha(gt)!=metadata[s['detection']]:raise ValueError('Selected bytes changed')
                box=selected['selected_bicycle_box']
                if not any(o['category_id']==3 and o['visible_bbox_xyxy']==box for o in read(gt)['positive_objects']):raise ValueError('Selected target changed')
                rgb=png(path);tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255
                with torch.inference_mode():p=model([tensor])[0]
                predicted=[{'box':b,'label':int(label),'score':float(score)} for b,label,score in zip(p['boxes'].tolist(),p['labels'].tolist(),p['scores'].tolist())]
                h,w=captured['size'];scaled=[box[0]*w/640,box[1]*h/360,box[2]*w/640,box[3]*h/360]
                proposal_iou=float(max(iou(scaled,captured['proposals'].tolist()),default=0))
                matching=[v for v in predicted if v['label']==3 and iou(v['box'],[box])[0]>=.5]
                score=max((v['score'] for v in matching),default=None);detected=score is not None and score>=.5
                original_detected=selected['selected_bicycle_diagnostic']=='detected'
                if budget==budgets[0] and detected!=original_detected:raise ValueError('Baseline bicycle decision differs from previous CPU diagnostic')
                cases.append({'sample':s,'selected_GT_box':box,'RPN_proposals':len(captured['proposals']),'max_RPN_proposal_IoU':proposal_iou,
                    'RPN_has_IoU50_proposal':proposal_iou>=.5,'highest_matching_bicycle_score':score,'bicycle_detected_at_original_score_half':detected,
                    'CPU_baseline_predictions_identical':predicted==selected['evaluation_frame']['predictions'] if budget==budgets[0] else None,
                    'wrong_class_matches_at_score_half':[v for v in predicted if v['label']!=3 and v['score']>=.5 and iou(v['box'],[box])[0]>=.5]})
            result={'budget':budget,'cases':cases,'selected_targets':len(cases),'detected':sum(c['bicycle_detected_at_original_score_half'] for c in cases),
                'RPN_IoU50_covered':sum(c['RPN_has_IoU50_proposal'] for c in cases)}
            results.append(result);write_json(args.output/'progress.json',{'completed_budgets':results})
            print(json.dumps({'budget':budget,'selected_targets':len(cases),'detected':result['detected'],'RPN_IoU50_covered':result['RPN_IoU50_covered']}),flush=True)
    finally:handle.remove()
    for name,value in model.state_dict().items():
        if not torch.equal(value.cpu(),saved['model'][name].cpu()):raise ValueError('Exploration changed learned tensor: '+name)
    write_json(args.output/'report.json',{'status':'completed_selected_CPU_proposal_study_only','results':results,'weights_identical':True,'no_trained_model_or_gate_changed':True,
        'limits':'Six biased GT-selected validation contexts, not full validation, GPU acceptance, physical occlusion rates or unseen-test performance. Proposal IoU50 alone is not a causal proof about all ROI classification/regression failures.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
