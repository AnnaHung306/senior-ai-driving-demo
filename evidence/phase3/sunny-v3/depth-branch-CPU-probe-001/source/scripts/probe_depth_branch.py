"""One bounded CPU-only real-train depth experiment, not production training."""
import argparse,copy,json,math,os
from pathlib import Path
import sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from experimental_depth_branch import ExperimentalDepthBranch,decode_depth_rgb,depth_target,masked_depth_loss
from predict_sunny import load_model
from senior_ai.mobile_segmentation import MobileUNet
from senior_ai.sunny_data import sha,png,segmentation_target,SEGMENTATION_NAMES
from senior_ai.sunny_losses import segmentation_loss
from senior_ai.segmentation_metrics import confusion,metrics
from senior_ai.core import write_json

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def run(args):
    out=args.output.resolve();parent=ROOT/'artifacts/runs/sunny-segmentation-v6-001';root=ROOT/'data/sunny-cross-town-v3-007'
    if out.exists() or not out.is_relative_to(ROOT/'artifacts/runs'):raise ValueError('New isolated artifacts/runs output required')
    state=read(parent/'state.json');prior=read(parent/'provenance.json')
    if state['status']!='completed' or state['lockbox_opened'] or sha(parent/'best.pt')!=state['best_sha256']:raise ValueError('Completed intact unevaluated parent required')
    for relative,digest in prior['source']['files'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Parent source changed: '+relative)
    for filename,key in [('audit.json','audit_sha256'),('visual_review.json','review_sha256'),('image_inventory.jsonl','image_inventory_sha256'),('metadata_inventory.json','metadata_inventory_sha256')]:
        if sha(root/filename)!=prior['data'][key]:raise ValueError('Main data contract changed')
    meta={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
    if sha(root/'training/train.jsonl')!=meta['training/train.jsonl']:raise ValueError('Train index changed')
    inventory={r['path']:r for r in map(json.loads,(root/'image_inventory.jsonl').read_text().splitlines())}
    rows=[json.loads(line) for line in (root/'training/train.jsonl').read_text().splitlines()]
    selected=[]
    for role,scenes in [('fit',['town01-00','town03-00']),('unused_train_probe',['town01-01','town03-01'])]:
        for scene in scenes:
            for camera in ['front','front_left']:
                row=next(r for r in rows if r['scene_id']==scene and r['camera']==camera)
                if row['split']!='train':raise ValueError('Train-only bounded prototype')
                selected.append({'role':role,'sample':row})
    out.mkdir(parents=True);args._created=True
    settings={'device':'cpu','threads':2,'seed':20261018,'updates':24,'batch_size':1,'shared_LR':1e-5,'depth_head_LR':.003,'depth_loss_coefficient':.2,
        'minimum_depth_m':.1,'maximum_depth_m':120.,'target':'log1p(depth_m)/log1p(120)','depth_loss':'valid per-image smoothL1 beta.05',
        'augmentation':'none; original aligned 640x360 only','max_training_seconds':300,'head_fit_sanity_ratio_max':.9}
    tool_paths=['scripts/experimental_depth_branch.py','scripts/probe_depth_branch.py','tests/test_experimental_depth_branch.py']
    contract={'scope':'Isolated CPU FP32 train-only 24-update feasibility probe. Four fit and four unused TRAIN views, no official validation/heldout, no production checkpoint mutation, no claimed Wall-IoU improvement or GPU capacity.',
        'settings':settings,'parent_best_sha256':state['best_sha256'],'parent_state_sha256':sha(parent/'state.json'),'parent_source':prior['source'],
        'parent_statistics_sha256':sha(parent/'dataset_statistics.json'),'main_data':prior['data'],'train_index_sha256':meta['training/train.jsonl'],
        'selected':selected,'prototype_sources':{p:sha(ROOT/p) for p in tool_paths},'network_forward_inputs':['RGB']}
    write_json(out/'contract.json',contract)
    for relative in tool_paths:
        p=out/'source'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/relative).read_bytes())
    checked={};samples=[]
    def image(relative):
        p=(root/relative).resolve()
        if not p.is_relative_to(root) or sha(p)!=inventory[relative]['sha256']:raise ValueError('Selected raw bytes differ')
        checked[relative]=inventory[relative]['sha256'];x=png(p)
        if x.shape!=(360,640,3):raise ValueError('Original rig size required')
        return x
    for item in selected:
        row=item['sample'];manifest=row['input_manifest']
        if sha(root/manifest)!=meta[manifest]:raise ValueError('Selected RGB manifest differs')
        m=read(root/manifest)
        if m['frame']!=row['frame'] or m['scene_id']!=row['scene_id'] or sha(root/m['calibration'])!=meta[m['calibration']]:raise ValueError('Selected frame/calibration differs')
        rgb=image(row['rgb']);semantic=image(row['semantic']);instance=image(row['instance'])
        depth_rel=row['semantic'].replace('labels/semantic_segmentation/','labels/depth/')
        depth=torch.from_numpy(decode_depth_rgb(image(depth_rel)).astype(np.float32))[None]
        target,valid=depth_target(depth,torch.from_numpy(semantic[...,0].copy())[None],settings['minimum_depth_m'],settings['maximum_depth_m'])
        truth=segmentation_target(semantic[...,0],instance,row['verified_obstacle_actor_ids'])
        samples.append({**item,'RGB':rgb,'rgb':torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255,
            'truth':torch.from_numpy(truth.copy()).long()[None],'depth':depth,'target':target,'valid':valid})
    write_json(out/'selected_bytes.json',checked)
    torch.set_num_threads(settings['threads']);torch.use_deterministic_algorithms(True);torch.manual_seed(settings['seed'])
    base,task,saved=load_model(parent/'best.pt','cpu')
    if task!='segmentation':raise ValueError('Wrong parent task')
    with torch.inference_mode():original=[base(s['rgb']) for s in samples]
    wrapper=ExperimentalDepthBranch(base,settings['seed']).cpu().eval()
    weights=np.asarray(read(parent/'dataset_statistics.json')['train_segmentation_pixels']);weights=torch.tensor(np.clip(np.sqrt(weights.max()/np.maximum(1,weights)),1,20),dtype=torch.float32)
    initial={};initial_depth=[]
    def evaluate(save_depth=False):
        wrapper.eval();reports={};pictures=[]
        for role in ['fit','unused_train_probe']:
            losses=[];relative=[];squared=[];matrix=np.zeros((17,17),dtype=np.int64);count=0
            for i,s in enumerate(samples):
                if s['role']!=role:continue
                with torch.inference_mode():logits,prediction=wrapper(s['rgb']);loss=masked_depth_loss(prediction,s['target'],s['valid'])
                if save_depth and not torch.equal(logits,original[i]):raise ValueError('Adding depth changed original RGB segmentation before training')
                full=F.interpolate(prediction,size=(360,640),mode='bilinear',align_corners=False)[:,0]
                depth_m=torch.expm1(full*math.log1p(settings['maximum_depth_m']));delta=(depth_m-s['depth'])[s['valid']]
                count+=delta.numel();relative.append(float((delta.abs()/s['depth'][s['valid']]).sum()));squared.append(float(delta.square().sum()));losses.append(float(loss))
                matrix+=confusion(s['truth'][0].numpy(),logits[0].argmax(0).numpy(),17)
                pictures.append((i,full[0].numpy().copy()))
            reports[role]={'views':len(losses),'valid_depth_pixels':count,'mean_normalized_log_depth_loss':float(np.mean(losses)),
                'depth_AbsRel':sum(relative)/count,'depth_RMSE_m':math.sqrt(sum(squared)/count),
                'biased_selected_train_segmentation':metrics(matrix,SEGMENTATION_NAMES)}
        return reports,dict(pictures)
    initial,initial_depth=evaluate(True);del original
    write_json(out/'initial.json',initial)
    wrapper.train();_,prediction=wrapper(samples[0]['rgb']);aux=masked_depth_loss(prediction,samples[0]['target'],samples[0]['valid'])
    gradient=torch.autograd.grad(aux,[wrapper.depth_head[2].weight,base.encoder[0][0].weight])
    grad_norms=[float(g.norm()) for g in gradient]
    if not all(math.isfinite(v) and v>0 for v in grad_norms):raise ValueError('Auxiliary loss does not reach head and shared encoder')
    optimizer=torch.optim.AdamW([{'params':base.parameters(),'lr':settings['shared_LR']},{'params':wrapper.depth_head.parameters(),'lr':settings['depth_head_LR']}],weight_decay=.0001)
    fit=[s for s in samples if s['role']=='fit'];started=time.monotonic();records=[]
    write_json(out/'state.json',{'status':'testing_CPU_prototype','total_updates':24,'updates':0})
    for update in range(settings['updates']):
        if time.monotonic()-started>settings['max_training_seconds']:raise TimeoutError('Bounded CPU training time exceeded; do not expand the experiment automatically')
        s=fit[update%len(fit)];wrapper.train();optimizer.zero_grad(set_to_none=True)
        logits,prediction=wrapper(s['rgb']);semantic=segmentation_loss(logits,s['truth'],weights);depth=masked_depth_loss(prediction,s['target'],s['valid'])
        total=semantic+settings['depth_loss_coefficient']*depth
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite toy loss')
        total.backward();norm=torch.nn.utils.clip_grad_norm_(wrapper.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        record={'update':update+1,'segmentation_loss':float(semantic.detach()),'depth_loss':float(depth.detach()),'total_loss':float(total.detach()),'gradient_norm':float(norm),'wall_s':time.monotonic()-started}
        records.append(record)
        with (out/'steps.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        if (update+1)%4==0:
            write_json(out/'state.json',{'status':'testing_CPU_prototype','updates':update+1,'total_updates':24});print(json.dumps(record),flush=True)
    final,final_depth=evaluate();write_json(out/'final.json',final)
    with torch.inference_mode():before=wrapper(samples[0]['rgb']);bypass=wrapper(samples[0]['rgb'],with_depth=False)
    if not torch.equal(before[0],bypass):raise ValueError('RGB segmentation bypass differs')
    checkpoint={'operation':'isolated_CPU_depth_auxiliary_feasibility','state_dict':wrapper.state_dict(),'optimizer_updates':24,'contract_sha256':sha(out/'contract.json')}
    torch.save(checkpoint,out/'prototype.pt')
    reloaded=ExperimentalDepthBranch(MobileUNet(precision='bfloat16'),settings['seed']).cpu().eval()
    reloaded.load_state_dict(torch.load(out/'prototype.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True)
    with torch.inference_mode():after=reloaded(samples[0]['rgb'])
    reload_identical=all(torch.equal(a,b) for a,b in zip(before,after))
    if not reload_identical:raise ValueError('Toy checkpoint reload differs')
    pages=[]
    def color(norm,valid):
        x=np.clip(norm,0,1);c=np.stack([x,1-np.abs(x*2-1),1-x],axis=-1);c[~valid]=0
        return (c*255).astype(np.uint8)
    for page_start in [0,4]:
        chosen=[page_start,page_start+2];page=Image.new('RGB',(1536,480));draw=ImageDraw.Draw(page)
        for j,index in enumerate(chosen):
            s=samples[index];valid=s['valid'][0].numpy();panels=[s['RGB'],color(s['target'][0].numpy(),valid),color(initial_depth[index],valid),color(final_depth[index],valid)]
            for k,pixels in enumerate(panels):page.paste(Image.fromarray(pixels).resize((384,216)),(384*k,240*j+24))
            draw.text((4,j*240+4),f"{s['role']} {s['sample']['scene_id']}/{s['sample']['camera']} | RGB / GT log-depth / INITIAL / AFTER24 CPU",fill='white')
        name=f'depth-page-{page_start//4}.png';page.save(out/name);pages.append({'path':name,'sha256':sha(out/name)})
    if sha(parent/'best.pt')!=state['best_sha256']:raise ValueError('Parent checkpoint mutated')
    if sha(parent/'dataset_statistics.json')!=contract['parent_statistics_sha256']:raise ValueError('Parent statistics changed')
    for relative,digest in {**prior['source']['files'],**contract['prototype_sources']}.items():
        if sha(ROOT/relative)!=digest:raise ValueError('Frozen source changed during prototype')
    ratio=final['fit']['mean_normalized_log_depth_loss']/initial['fit']['mean_normalized_log_depth_loss']
    result={'status':'completed_CPU_depth_branch_feasibility_only','mechanism_checks_passed':True,'small_fit_sanity_passed':ratio<=settings['head_fit_sanity_ratio_max'],
        'fit_depth_loss_ratio':ratio,'updates':24,'training_wall_s':records[-1]['wall_s'],'base_parameters':sum(p.numel() for p in base.parameters()),
        'additional_depth_parameters':sum(p.numel() for p in wrapper.depth_head.parameters()),'initial_RGB_segmentation_exact_parent':True,
        'depth_only_gradient_norms_head_and_encoder':grad_norms,'checkpoint_reload_identical':reload_identical,'segmentation_bypass_identical':True,
        'parent_checkpoint_unchanged':True,'network_forward_inputs':['RGB'],'device':'CPU FP32','GPU_capacity_tested':False,
        'prototype_checkpoint_sha256':sha(out/'prototype.pt'),'initial':initial,'final':final,'pages':pages,'contract_sha256':sha(out/'contract.json'),
        'limits':'Only four train fit views and four unused train views, 24 updates without augmentation. No validation/heldout, no production integration, no evidence of improved cross-town Wall IoU or CUDA/BF16 capacity. Assistant visual review pending.'}
    write_json(out/'report.json',result);write_json(out/'state.json',{k:v for k,v in result.items() if k not in ['initial','final']})
    print(json.dumps({k:v for k,v in result.items() if k not in ['initial','final']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args._created=False
    try:run(args)
    except Exception as exc:
        if args._created:write_json(args.output/'failure.json',{'error':type(exc).__name__,'message':str(exc),'note':'Preserve this bounded prototype; no automatic retry or increased training budget.'})
        raise
