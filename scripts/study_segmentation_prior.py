"""CPU validation-only study of train-frequency logit prior correction.

Fixed alpha grid, unchanged checkpoint; never an official GPU acceptance run.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from predict_sunny import load_model
from senior_ai.sunny_data import sha, png, segmentation_target, SEGMENTATION_NAMES
from senior_ai.segmentation_metrics import confusion, metrics
from senior_ai.road_boundaries import boundary_counts, boundary_metrics
from senior_ai.core import write_json
from senior_ai.perception_quality import quality
from collections import Counter


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def correction_bias(support,wall_only=False):
    support=np.asarray(support,dtype=np.float64)
    if support.shape!=(len(SEGMENTATION_NAMES),) or not np.isfinite(support).all() or (support<0).any() or support.max()<=0:
        raise ValueError('Finite nonnegative17-class actual train support required')
    weights=np.clip(np.sqrt(max(1,support.max())/np.maximum(1,support)),1,20)
    bias=torch.tensor(np.log(weights),dtype=torch.float32)[:,None,None]
    if wall_only:
        selected=torch.zeros_like(bias);selected[12]=bias[12];bias=selected
    return bias,weights


def main(args):
    if args.output.exists():raise FileExistsError('New study version required')
    parent=args.checkpoint.resolve().parent
    if args.wall_only:
        state=read(parent/'state.json');config=read(parent/'config.json');gate=quality('segmentation',read(parent/'validation.json'),config['quality_gates'])
        if state['status']!='completed' or sha(args.checkpoint)!=state['best_sha256'] or sha(parent/'validation.json')!=state['validation_sha256']:
            raise ValueError('Wall-only diagnostic requires a completed verified parent')
        if {k for k,v in gate['checks'].items() if not v}!={'wall/IoU'}:raise ValueError('Wall must be the sole original failed validation gate')
        if state.get('lockbox_opened') or (parent/'lockbox_started.json').exists():raise ValueError('No new study after heldout opening')
    root=args.root.resolve();args.output.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    checkpoint_sha=sha(args.checkpoint);model,task,saved=load_model(args.checkpoint,'cpu')
    if task!='segmentation':raise ValueError('Semantic checkpoint required')
    for path,digest in saved['provenance']['source']['files'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Checkpoint source differs')
    bound=saved['provenance']['data']
    for path,key in [('audit.json','audit_sha256'),('image_inventory.jsonl','image_inventory_sha256'),('metadata_inventory.json','metadata_inventory_sha256')]:
        if sha(root/path)!=bound[key]:raise ValueError('Data binding differs')
    if sha(args.statistics)!=args.statistics_sha256:raise ValueError('Frozen train statistics changed')
    support=np.asarray(read(args.statistics)['train_segmentation_pixels'])
    bias,weights=correction_bias(support,args.wall_only)
    rows=[json.loads(line) for line in (root/'training/validation.jsonl').read_text().splitlines()]
    meta={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
    if sha(root/'training/validation.jsonl')!=meta['training/validation.jsonl'] or any(r['split']!='validation' for r in rows):raise ValueError('Canonical validation only')
    inventory={r['path']:r['sha256'] for r in map(json.loads,(root/'image_inventory.jsonl').read_text().splitlines())}
    grid=[i/8 for i in range(9)] if args.wall_only else [0.,.25,.5,.75,1.]
    write_json(args.output/'contract.json',{'checkpoint_sha256':checkpoint_sha,'epoch':saved['epoch'],'updates':saved['updates'],
        'tool_sha256':sha(Path(__file__)),'source':saved['provenance']['source'],'main_audit_sha256':bound['audit_sha256'],
        'train_statistics_sha256':args.statistics_sha256,'train_weights':weights.tolist(),'alpha_grid':grid,'validation_index_sha256':meta['training/validation.jsonl'],
        'corrected_class_indices':[12] if args.wall_only else list(range(17)),
        'parent_state_sha256':sha(parent/'state.json') if args.wall_only else None,
        'formula':'logits_corrected = logits - alpha * log(inverse_sqrt_train_frequency_weights), masked to recorded class indices',
        'scope':'CPU FP32 exploratory validation only. No parameter updates, model-selection claim, final GPU gate, or heldout data. Full grid fixed before inference.'})
    (args.output/'tool_used.py').write_bytes(Path(__file__).read_bytes())
    if args.wall_only:(args.output/'parent_GPU_validation.json').write_bytes((parent/'validation.json').read_bytes())
    matrices={str(a):np.zeros((17,17),dtype=np.int64) for a in grid};borders={str(a):Counter() for a in grid}
    def checked(rel):
        path=(root/rel).resolve()
        if not path.is_relative_to(root) or sha(path)!=inventory[rel]:raise ValueError('Validation bytes differ')
        return png(path)
    for i,row in enumerate(rows):
        rgb=checked(row['rgb']);truth=segmentation_target(checked(row['semantic'])[...,0],checked(row['instance']),row['verified_obstacle_actor_ids'])
        with torch.inference_mode():logits=model(torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255)[0]
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite CPU inference')
        for a in grid:
            prediction=(logits-a*bias).argmax(0).numpy()
            matrices[str(a)]+=confusion(truth,prediction,17);borders[str(a)].update(boundary_counts(truth,prediction))
        if (i+1)%100==0:
            result={k:{**metrics(m,SEGMENTATION_NAMES),'visible_margin':boundary_metrics(borders[k])} for k,m in matrices.items()}
            write_json(args.output/'progress.json',{'views':i+1,'total':len(rows),'diagnostic_only':True,'grid':result})
            print(json.dumps({'cpu_diagnostic_views':i+1,'total':len(rows)}),flush=True)
    result={k:{**metrics(m,SEGMENTATION_NAMES),'visible_margin':boundary_metrics(borders[k])} for k,m in matrices.items()}
    if sha(args.checkpoint)!=checkpoint_sha:raise ValueError('Parent checkpoint changed during CPU study')
    write_json(args.output/'report.json',{'status':'completed_CPU_validation_diagnostic_only','checkpoint_sha256':checkpoint_sha,'views':len(rows),'grid':result,
        'corrected_class_indices':[12] if args.wall_only else list(range(17)),
        'CPU_only_quality_diagnostic':{k:quality('segmentation',v,saved['config']['quality_gates']) for k,v in result.items()}})
    print(json.dumps({'status':'completed_CPU_validation_diagnostic_only','views':len(rows),'summary':{k:{'mIoU':v['focus_macro_iou'],'wall':v['classes'][12]['iou'],'margin_f1':v['visible_margin']['f1']} for k,v in result.items()}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--statistics',type=Path,required=True);p.add_argument('--statistics-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wall-only',action='store_true');main(p.parse_args())
