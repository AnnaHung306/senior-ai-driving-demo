"""Bound post-training semantic calibration, full GPU validation, no heldout.

The only changed parameter is the existing classifier bias. Inherited training
and this calibration operation are recorded separately, without resume support.
"""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time
import torch
from train_sunny import (ROOT, build_model, configure_cuda, evaluate, inputs,
                         require_idle_gpu, save_checkpoint, source_fingerprint)
from senior_ai.sunny_data import SunnySamples, sha, verify
from senior_ai.segmentation_supplement import append_scenery, verify_scenery
from senior_ai.perception_quality import quality, selection_key
from senior_ai.core import write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def validate_grid(grid):
    if not isinstance(grid,list) or not grid or grid[0]!=0 or len(set(grid))!=len(grid):
        raise ValueError('Unique fixed alpha grid must start with unmodified alpha0')
    if any(type(a) not in (int,float) or not math.isfinite(a) or not 0<=a<=1 for a in grid):
        raise ValueError('Prior correction alpha must be finite in [0,1]')
    return [float(a) for a in grid]


def corrected_bias(original,support,alpha):
    support=torch.as_tensor(support,dtype=torch.float64,device='cpu')
    if support.shape!=original.shape or bool((support<0).any()) or not bool(torch.isfinite(support).all()):
        raise ValueError('Finite nonnegative train support must match classifier')
    weights=(max(1.,float(support.max()))/support.clamp_min(1)).sqrt().clamp(1,20)
    return original.detach().cpu()-float(alpha)*weights.log().to(original.dtype)


def main(args):
    parent=args.parent.resolve();run=args.run.resolve();root=args.root.resolve()
    if run.exists():raise FileExistsError('Calibration requires a new run; partial runs remain preserved')
    design=read(args.design);grid=validate_grid(design['alpha_grid'])
    if design['method']!='classifier_bias_train_prior_correction' or design['selection_policy']!='qualified_first_then_score':
        raise ValueError('Explicit prior correction and original qualified-first selection required')
    state=read(parent/'state.json');prior=read(parent/'provenance.json');config=read(parent/'config.json')
    if state['status']!='completed' or state['task']!='segmentation' or not state['checkpoint_reload_identical']:
        raise ValueError('Completed reloaded segmentation parent required')
    if (parent/'lockbox_started.json').exists() or state.get('lockbox_opened'):
        raise ValueError('Cannot calibrate a candidate after its heldout was opened')
    for filename,key in [('best.pt','best_sha256'),('validation.json','validation_sha256'),('validation_frames.jsonl','validation_frames_sha256')]:
        if sha(parent/filename)!=state[key]:raise ValueError('Parent result bytes changed')
    if quality('segmentation',read(parent/'validation.json'),config['quality_gates'])['passed']:
        raise ValueError('Parent already passes original validation; no recovery calibration needed')
    if sha(parent/'config.json')!=prior['config_sha256'] or source_fingerprint()!=prior['source']:
        raise ValueError('Parent config/source differs; use frozen parent sources')
    require_idle_gpu();configure_cuda();started=time.monotonic()
    data=verify(root)
    if data!=prior['data']:raise ValueError('Main audit/review/inventories differ')
    scenery=config['segmentation'].get('training_scenery');supplement=None
    if scenery:
        supplement=verify_scenery(ROOT/scenery['root'],data['audit_sha256'],scenery['plan_sha256'])
        supplement['sampling_repeats']=scenery['sampling_repeats']
        if supplement!=prior['training_supplement']:raise ValueError('Parent supplement differs')
    train=SunnySamples(root,'train','segmentation',config['cache_budget_gib_per_split'])
    if scenery:append_scenery(train,ROOT/scenery['root'],scenery['sampling_repeats'],config['cache_budget_gib_per_split'])
    support=train.support.copy()
    statistics=read(parent/'dataset_statistics.json')
    if support.tolist()!=statistics['train_segmentation_pixels'] or len(train)!=statistics['train_views']:
        raise ValueError('Independently recomputed train support differs from parent statistics')
    del train;gc.collect()
    run.mkdir(parents=True)
    contract={'method':design['method'],'design_sha256':sha(args.design),'tool_sha256':sha(Path(__file__)),
        'parent_run':str(parent),'parent_best_sha256':state['best_sha256'],'parent_state_sha256':sha(parent/'state.json'),
        'parent_config_sha256':sha(parent/'config.json'),'main_data':data,'training_supplement':supplement,
        'recomputed_train_support':support.tolist(),'alpha_grid':grid,'selection_policy':design['selection_policy'],
        'scope':'Post-training parameter calibration only. Original class mapping, data boundaries and quality gates unchanged. Full GPU BF16 validation, no heldout.'}
    write_json(run/'calibration_contract.json',contract)
    (run/'calibration_design.json').write_bytes(args.design.read_bytes())
    validation=SunnySamples(root,'validation','segmentation',config['cache_budget_gib_per_split'])
    saved=torch.load(parent/'best.pt',map_location='cpu',weights_only=False)
    model=build_model('segmentation',config,pretrained=False).cuda();model.load_state_dict(saved['model'])
    original=model.head.bias.detach().cpu().clone();size=config['segmentation']['batch_size'];winner=None;records=[]
    for alpha in grid:
        with torch.no_grad():model.head.bias.copy_(corrected_bias(original,support,alpha).to('cuda'))
        result,_=evaluate(model,'segmentation',validation,size)
        if alpha==0 and result!=read(parent/'validation.json'):
            raise ValueError('Unmodified candidate does not reproduce exact parent full GPU validation')
        gates=quality('segmentation',result,config['quality_gates'])
        record={'alpha':alpha,'score':result['focus_macro_iou'],'quality':gates,'result':result}
        records.append(record);write_json(run/'candidates.json',records)
        key=selection_key(record['score'],gates['passed'],design['selection_policy'])
        if winner is None or key>winner['key']:winner={'key':key,'alpha':alpha,'result':result,'quality':gates}
        print(json.dumps({'alpha':alpha,'focus_mIoU':record['score'],'wall_IoU':result['classes'][12]['iou'],
            'visible_margin_f1':result['visible_margin']['f1'],'validation_qualified':gates['passed']}),flush=True)
    alpha=winner['alpha']
    with torch.no_grad():model.head.bias.copy_(corrected_bias(original,support,alpha).to('cuda'))
    model.eval();images=inputs([validation[0][0]])
    with torch.inference_mode():before=model(images).clone()
    # Freeze final model metadata after the predeclared grid selected alpha.
    config=copy.deepcopy(config);config['version']=design['output_version']
    calibration={**contract,'selected_alpha':alpha,'parent_training_updates':state['updates'],'calibration_optimizer_updates':0,
        'allowed_changed_parameters':['head.bias'],'formula':'head.bias_new = head.bias_parent - alpha*log(clamp(sqrt(max(train_support)/train_support),1,20))'}
    config['post_training_calibration']=calibration;write_json(run/'config.json',config)
    provenance=copy.deepcopy(prior);provenance['config_sha256']=sha(run/'config.json');provenance['post_training_calibration']=calibration
    write_json(run/'provenance.json',provenance);write_json(run/'dataset_statistics.json',statistics)
    checkpoint={'task':'segmentation','model':model.state_dict(),'epoch':saved['epoch'],'updates':0,
        'inherited_training_updates':state['updates'],'config':config,'provenance':provenance,'post_training_calibration':calibration}
    save_checkpoint(run/'best.pt',checkpoint)
    reloaded=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(reloaded['model'])
    with torch.inference_mode():identical=torch.equal(before,model(images))
    if not identical:raise RuntimeError('Calibrated checkpoint reload differs')
    for name,value in reloaded['model'].items():
        if name!='head.bias' and not torch.equal(value.cpu(),saved['model'][name].cpu()):raise ValueError('Calibration changed non-bias tensor: '+name)
    result,frames=evaluate(model,'segmentation',validation,size)
    if result!=winner['result']:raise ValueError('Selected checkpoint validation differs from candidate')
    write_json(run/'validation.json',result);write_json(run/'best_validation.json',result);write_json(run/'validation_quality.json',winner['quality'])
    (run/'validation_frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')
    for relative in provenance['source']['files']:
        target=run/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
    tool=run/'calibration_tool.py';tool.write_bytes(Path(__file__).read_bytes())
    receipt={'status':'completed','operation':'post_training_classifier_bias_calibration','task':'segmentation','best_epoch':saved['epoch'],
        'updates':0,'inherited_training_updates':state['updates'],'best_score':result['focus_macro_iou'],'best_sha256':sha(run/'best.pt'),
        'checkpoint_reload_identical':identical,'non_bias_tensors_identical_to_parent':True,'selected_validation_qualified':winner['quality']['passed'],
        'selection_policy':design['selection_policy'],'validation_sha256':sha(run/'validation.json'),'validation_frames_sha256':sha(run/'validation_frames.jsonl'),
        'training_wall_s':0,'calibration_wall_s':time.monotonic()-started,'parameter_count':sum(p.numel() for p in model.parameters()),
        'max_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'max_cuda_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
        'lockbox_opened':False,'calibration_contract_sha256':sha(run/'calibration_contract.json'),
        'candidates_sha256':sha(run/'candidates.json'),'selected_alpha':alpha,'calibration_tool_sha256':sha(tool)}
    write_json(run/'state.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--design',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    try:main(args)
    except Exception as exc:
        if args.run.exists():write_json(args.run/'failure.json',{'error':type(exc).__name__,'message':str(exc)})
        raise
