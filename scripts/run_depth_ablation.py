"""One-shot GPU preflight then two one-epoch research arms; no heldout access."""
import argparse,copy,gc,hashlib,json,math,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,value):
    p=Path(p);temporary=p.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(p)

def decide(control,depth,minimum_gain=.002):
    from depth_ablation_components import nonwall_iou
    gain=nonwall_iou(depth)-nonwall_iou(control)
    margin=depth['visible_margin']['f1']-control['visible_margin']['f1']
    regressions=[b['name'] for a,b in zip(control['classes'],depth['classes']) if a['name'] not in {'wall','other_annotated'} and a['iou']>=.2 and b['iou']<.2]
    return {'nonwall_mIoU_gain':gain,'visible_margin_f1_change':margin,'new_nonwall_class_gate_regressions':regressions,
        'positive_signal_for_assistant_review':gain>=minimum_gain and margin>=-.01 and not regressions,
        'automatic_extra_epochs':False,'limits':'Predeclared one-run screening rule, not statistical significance or changed original quality gates.'}

def frozen_sources(config):
    parent=ROOT/config['parent_run'];prior=read(parent/'provenance.json')
    files=dict(prior['source']['files'])
    for name in ['experimental_depth_branch.py','depth_ablation_components.py','run_depth_ablation.py','hold_awake_for_training.py']:
        files['scripts/'+name]=sha(ROOT/'scripts'/name)
    files['tests/test_depth_ablation_components.py']=sha(ROOT/'tests/test_depth_ablation_components.py')
    return files

def verify_sources(files):
    for relative,digest in files.items():
        if sha(ROOT/relative)!=digest:raise ValueError('Frozen source changed: '+relative)

def await_previous(args,contract):
    batch=args.wait_for_batch;qa=args.wait_for_qa;deadline=time.monotonic()+6*3600
    write(args.output/'state.json',{'status':'waiting_for_existing_batch_and_QA','GPU_work_started':False})
    while True:
        if (args.output/'CANCEL').exists():raise RuntimeError('Explicit local cancellation marker; no new GPU work started')
        for root in [batch,qa]:
            if (root/'failure.json').exists():raise RuntimeError('Existing job failed; inspect '+str(root/'failure.json'))
        try:a=read(batch/'state.json');b=read(qa/'state.json')
        except (OSError,json.JSONDecodeError):a={};b={}
        if a.get('status') in {'quality_failed','phase3_metrics_passed_pending_visual_review'} and b.get('status')=='completed_artifacts_pending_assistant_visual_review':break
        if time.monotonic()>deadline:raise TimeoutError('Six-hour current-session wait ended; no automatic retry')
        time.sleep(10)
    if sha(batch/'contract.json')!=contract['previous_batch_contract_sha256'] or sha(qa/'contract.json')!=contract['previous_QA_contract_sha256']:
        raise ValueError('Previous source contract changed')
    if sha(batch/'export/summary.json')!=a['summary_sha256']:raise ValueError('Previous batch terminal receipt changed')
    previous=read(batch/'contract.json')
    for relative,digest in previous['sources'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Previous frozen implementation changed')
    for name,digest in read(qa/'contract.json')['tools'].items():
        if sha(ROOT/'scripts'/name)!=digest:raise ValueError('Previous QA implementation changed')
    for path,key in [('segmentation/catalog.json','segmentation_catalog_sha256'),('detection/catalog.json','detection_catalog_sha256'),('phase/receipt.json','phase_receipt_sha256')]:
        if sha(qa/path)!=b[key]:raise ValueError('Previous QA artifact changed')
    verify_sources(contract['sources'])
    write(args.output/'previous_jobs_completed.json',{'batch_state_sha256':sha(batch/'state.json'),'QA_state_sha256':sha(qa/'state.json'),'scope':'GPU work finished; prior visual artifacts still require assistant review, not a Phase3 quality pass.'})

def execute(config,out,contract):
    import random,platform
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image,ImageDraw
    from train_sunny import (configure_cuda,require_idle_gpu,batches,inputs,evaluate,make_optimizer,save_checkpoint,seed_all)
    from predict_sunny import load_model,PALETTE
    from experimental_depth_branch import ExperimentalDepthBranch
    from depth_ablation_components import DepthStream,paired_depth_geometry,nearest_depth_loss,nonwall_iou,update_augmentation_digest
    from senior_ai.sunny_data import SunnySamples,verify
    from senior_ai.segmentation_supplement import verify_scenery,append_scenery
    from senior_ai.segmentation_expansion import verify_expansion,append_expansion
    from senior_ai.sunny_losses import segmentation_loss
    from senior_ai.segmentation_metrics import confusion
    from senior_ai.road_boundaries import boundary_counts,boundary_metrics
    from senior_ai.perception_quality import quality
    require_idle_gpu();configure_cuda();parent=ROOT/config['parent_run'];root=ROOT/config['dataset'];pc=read(parent/'config.json');prior=read(parent/'provenance.json');ps=read(parent/'state.json')
    if ps['status']!='completed' or ps['lockbox_opened'] or sha(parent/'best.pt')!=config['parent_best_sha256'] or sha(parent/'config.json')!=config['parent_config_sha256']:
        raise ValueError('Parent no longer matches fixed design')
    for filename,key in [('provenance.json','parent_provenance_sha256'),('state.json','parent_state_sha256'),('validation.json','parent_validation_sha256'),('validation_frames.jsonl','parent_frames_sha256'),('dataset_statistics.json','parent_statistics_sha256')]:
        if sha(parent/filename)!=config[key]:raise ValueError('Frozen parent artifact changed: '+filename)
    write(out/'state.json',{'status':'verifying_and_loading_data','GPU_work_started':True})
    signature=verify(root)
    if signature!=prior['data']:raise ValueError('Parent data binding differs')
    c=pc['segmentation'];extra=c.get('training_scenery');expansions=c.get('training_expansions',[])
    if extra:
        found=verify_scenery(ROOT/extra['root'],signature['audit_sha256'],extra['plan_sha256']);found['sampling_repeats']=extra['sampling_repeats']
        if found!=prior['training_supplement']:raise ValueError('Original scenery binding differs')
    found=[{**verify_expansion(ROOT/e['root'],root,e['plan_sha256'],e['audit_sha256'],e['review_sha256']),'sampling_repeats':e['sampling_repeats']} for e in expansions]
    if found!=prior.get('training_expansions',[]):raise ValueError('Reserved scenery binding differs')
    train=SunnySamples(root,'train','segmentation',pc['cache_budget_gib_per_split']);val=SunnySamples(root,'validation','segmentation',pc['cache_budget_gib_per_split'])
    if extra:append_scenery(train,ROOT/extra['root'],extra['sampling_repeats'],pc['cache_budget_gib_per_split'])
    for e in expansions:append_expansion(train,ROOT/e['root'],e['sampling_repeats'],pc['cache_budget_gib_per_split'])
    statistics=read(parent/'dataset_statistics.json')
    if train.support.tolist()!=statistics['train_segmentation_pixels'] or len(train)!=statistics['train_views'] or len(val)!=statistics['validation_views']:
        raise ValueError('Loaded counts/support differ from parent')
    stream=DepthStream(root,pc,capacity=128,minimum=config['minimum_depth_m'],maximum=config['maximum_depth_m'])
    weights=torch.tensor(np.clip(np.sqrt(max(1,train.support.max())/np.maximum(1,train.support)),1,20),device='cuda',dtype=torch.float32)
    size=config['batch_size'];steps=math.ceil(len(train)/size)
    write(out/'runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'python':platform.python_version(),
        'train_sampled_views':len(train),'validation_views':len(val),'updates_per_arm':steps,'training_support':train.support.tolist(),
        'depth_cache_entries':128,'depth_loss_resampling':'nearest, to avoid CUDA bilinear backward nondeterminism; differs from prior CPU prototype display/loss interpolation'})
    def fresh():
        base,task,_=load_model(parent/'best.pt','cuda')
        if task!='segmentation':raise ValueError('Wrong initializer task')
        return base
    def auxiliary(base):
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
        # The original CPU prototype seeds all devices inside a CPU-only fork.
        # Enclose it in an all-device fork for this CUDA experiment.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            wrapper=ExperimentalDepthBranch(base,config['head_seed']).cuda()
        if not torch.equal(cpu,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())):
            raise ValueError('Auxiliary head construction changed CPU/CUDA RNG')
        return wrapper
    def geometry(rgb,target,indices):
        z,valid=zip(*(stream.get(train.samples[int(i)]) for i in indices))
        images=inputs(rgb,augment=True,photometric_range=c['photometric_range'])
        return paired_depth_geometry(images,torch.stack(target).cuda(),torch.stack(z).cuda(),torch.stack(valid).cuda(),**c['paired_geometry'])
    def optimizer_for(base,wrapper):
        optimizer=make_optimizer(base,c)
        if wrapper is not None:optimizer.add_param_group({'params':wrapper.depth_head.parameters(),'lr':config['depth_head_learning_rate']})
        return optimizer
    # One fixed batch8 check using real views from two original dynamic train groups.
    first_groups=[]
    for row in train.samples[:9600]:
        if row['group_id'] not in first_groups:first_groups.append(row['group_id'])
    chosen=[]
    for group in [first_groups[0],first_groups[-1]]:
        chosen.extend([i for i,row in enumerate(train.samples[:9600]) if row['group_id']==group][:size//2])
    if len(chosen)!=size or size!=8:raise ValueError('Predeclared two-group batch8 capacity design required')
    seed_all(config['seed']);base=fresh();wrapper=auxiliary(base);optimizer=optimizer_for(base,wrapper)
    rgb=[train[i][0] for i in chosen];target=[train[i][1] for i in chosen];losses=[];feature_dtypes=set();convolution_dtypes=set()
    handles=[base.dec4.register_forward_hook(lambda module,args,output:feature_dtypes.add(str(output.dtype))),
        base.dec4[0].register_forward_hook(lambda module,args,output:convolution_dtypes.add(str(output.dtype)))]
    write(out/'state.json',{'status':'GPU_BF16_32_update_preflight','GPU_work_started':True});torch.cuda.reset_peak_memory_stats()
    for update in range(config['preflight_updates']):
        images,ids,z,valid,_=geometry(rgb,target,chosen);optimizer.zero_grad(set_to_none=True);wrapper.train()
        logits,prediction=wrapper(images);sl=segmentation_loss(logits,ids,weights);dl=nearest_depth_loss(prediction,z,valid);loss=sl+config['depth_loss_coefficient']*dl
        if not torch.isfinite(loss):raise FloatingPointError('GPU preflight loss not finite')
        if update==0:
            depth_only=torch.autograd.grad(dl,[wrapper.depth_head[2].weight,base.encoder[0][0].weight],retain_graph=True)
            gradients={'depth_head':float(depth_only[0].norm()),'shared_encoder':float(depth_only[1].norm())}
            if not all(math.isfinite(v) and v>0 for v in gradients.values()):raise ValueError('Missing head/shared gradient')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(wrapper.parameters(),5.,error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()))
    for handle in handles:handle.remove()
    wrapper.eval();images=inputs(rgb)
    with torch.inference_mode():before=tuple(t.detach().cpu() for t in wrapper(images))
    preflight_state={k:v.detach().cpu().clone() for k,v in wrapper.state_dict().items()}
    capacity={'status':'passed','updates':len(losses),'preflight_shared_state_discarded_before_arms':True,'head_initialization_CPU_CUDA_RNG_preserved':True,'selected_train_indices':chosen,
        'original_train_groups':[first_groups[0],first_groups[-1]],'shared_feature_dtypes':sorted(feature_dtypes),'autocast_convolution_dtypes':sorted(convolution_dtypes),'depth_only_gradient_norms':gradients,
        'losses':losses,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
    if 'torch.bfloat16' not in convolution_dtypes or capacity['peak_allocated_mib']>config['maximum_allocated_mib']:raise ValueError('BF16/capacity requirement failed')
    del wrapper,base,optimizer,images,logits,prediction,loss,sl,dl;gc.collect();torch.cuda.empty_cache()
    base=fresh();wrapper=auxiliary(base).eval();wrapper.load_state_dict(preflight_state)
    with torch.inference_mode():after=tuple(t.detach().cpu() for t in wrapper(inputs(rgb)))
    capacity['reload_identical']=all(torch.equal(a,b) for a,b in zip(before,after))
    if not capacity['reload_identical']:raise ValueError('BF16 full-batch reload differs')
    write(out/'GPU_preflight.json',capacity)
    del wrapper,base,before,after,preflight_state;gc.collect();torch.cuda.empty_cache()
    results={};frames={};hashes={}
    # Each arm is initialized afresh: none of the32 preflight updates are retained.
    for arm in ['control','depth']:
        arm_root=out/arm;arm_root.mkdir();verify_sources(contract['sources']);seed_all(config['seed']);base=fresh()
        wrapper=auxiliary(base) if arm=='depth' else None
        initial,_=evaluate(base,'segmentation',val,size)
        if initial!=read(parent/'validation.json'):raise ValueError('Arm initialization does not exactly replay full parent GPU validation')
        write(arm_root/'initial_GPU_replay.json',{'exact_parent_full_validation':True,'parent_validation_sha256':sha(parent/'validation.json')})
        seed_all(config['seed']);optimizer=optimizer_for(base,wrapper)
        schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1,(s+1)/max(1,c['warmup_steps']))*.5*(1+math.cos(math.pi*min(s,steps)/steps)))
        started=time.monotonic();digest=hashlib.sha256();losses=[];depth_losses=[];max_norm=0.;torch.cuda.reset_peak_memory_stats()
        write(out/'state.json',{'status':'training_one_epoch','arm':arm,'updates':0,'total_updates':steps,'GPU_work_started':True})
        for update,(rgb,target,indices) in enumerate(batches(train,size,shuffle=True),1):
            if (out/'CANCEL').exists():raise RuntimeError('Explicit local cancellation; partial research preserved')
            images,ids,z,valid,ledger=geometry(rgb,target,indices);update_augmentation_digest(digest,indices,images,ids,ledger)
            optimizer.zero_grad(set_to_none=True);base.train()
            if wrapper is not None:
                wrapper.train();logits,prediction=wrapper(images);dl=nearest_depth_loss(prediction,z,valid)
            else:logits=base(images);dl=logits.sum()*0
            sl=segmentation_loss(logits,ids,weights);loss=sl+config['depth_loss_coefficient']*dl
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite A/B loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_((wrapper or base).parameters(),5.,error_if_nonfinite=True);optimizer.step();schedule.step()
            losses.append(float(sl.detach()));depth_losses.append(float(dl.detach()));max_norm=max(max_norm,float(norm))
            if update%100==0 or update==steps:
                progress={'status':'training_one_epoch','arm':arm,'updates':update,'total_updates':steps,'mean_segmentation_loss':float(np.mean(losses)),
                    'mean_depth_loss':float(np.mean(depth_losses)),'wall_s':time.monotonic()-started,'GPU_work_started':True}
                with (arm_root/'train_progress.jsonl').open('a',encoding='utf-8') as stream_out:stream_out.write(json.dumps(progress)+'\n')
                write(out/'state.json',progress);print(json.dumps(progress),flush=True)
        if update!=steps:raise ValueError('Incomplete one-epoch arm')
        result,rows=evaluate(base,'segmentation',val,size);results[arm]=result;frames[arm]=rows;hashes[arm]=digest.hexdigest()
        write(arm_root/'validation.json',result)
        (arm_root/'validation_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        arm_config=copy.deepcopy(pc);arm_config['version']='depth-ablation-v1-'+arm;arm_config['segmentation']['epochs']=1
        arm_config['segmentation']['initial_checkpoint_path']=str(parent/'best.pt');arm_config['segmentation']['initial_checkpoint_sha256']=config['parent_best_sha256']
        arm_config['bounded_research']={'arm':arm,'experiment_contract_sha256':sha(out/'contract.json'),'depth_loss_coefficient':config['depth_loss_coefficient'] if arm=='depth' else 0,'formal_model_adoption':False}
        write(arm_root/'config.json',arm_config)
        checkpoint={'task':'segmentation','model':base.state_dict(),'auxiliary_depth_head':wrapper.depth_head.state_dict() if wrapper else None,
            'config':arm_config,'provenance':{**prior,'config_sha256':sha(arm_root/'config.json'),'bounded_research_sources':contract['sources']},'epoch':0,'updates':steps,
            'optimizer':optimizer.state_dict(),'scheduler':schedule.state_dict(),'rng':{'python':random.getstate(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
        save_checkpoint(arm_root/'final.pt',checkpoint)
        with torch.inference_mode():before=base(inputs([val[0][0]])).detach().cpu()
        reload_model,_,_=load_model(arm_root/'final.pt','cpu');del wrapper,base,optimizer,schedule,images,logits,loss,sl,dl
        if arm=='depth':del prediction
        gc.collect();torch.cuda.empty_cache();reload_model.cuda().eval()
        with torch.inference_mode():after=reload_model(inputs([val[0][0]])).detach().cpu()
        if not torch.equal(before,after):raise ValueError('Arm segmentation checkpoint reload differs')
        arm_state={'status':'completed_research_arm','operation':'bounded_one_epoch_depth_ablation','arm':arm,'updates':steps,'sampled_train_views':len(train),'nonwall_focus_mIoU':nonwall_iou(result),
            'original_quality':quality('segmentation',result,pc['quality_gates']),'checkpoint_reload_identical':True,'final_checkpoint_sha256':sha(arm_root/'final.pt'),
            'validation_sha256':sha(arm_root/'validation.json'),'validation_frames_sha256':sha(arm_root/'validation_frames.jsonl'),
            'RGB_semantic_augmentation_sha256':digest.hexdigest(),'training_wall_s':time.monotonic()-started,'max_gradient_norm':max_norm,
            'mean_segmentation_loss':float(np.mean(losses)),'mean_depth_loss':float(np.mean(depth_losses)),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'heldout_opened':False}
        write(arm_root/'state.json',arm_state);print(json.dumps(arm_state),flush=True)
        del reload_model,before,after,checkpoint;gc.collect();torch.cuda.empty_cache()
    if hashes['control']!=hashes['depth']:raise ValueError('A/B RGB/semantic/order/geometry bytes differ; comparison invalid')
    comparison=decide(results['control'],results['depth'],config['minimum_nonwall_mIoU_gain'])
    comparison.update(parent_nonwall_focus_mIoU=nonwall_iou(read(parent/'validation.json')),control_nonwall_focus_mIoU=nonwall_iou(results['control']),depth_nonwall_focus_mIoU=nonwall_iou(results['depth']),
        per_class=[{'name':a['name'],'control_IoU':a['iou'],'depth_IoU':b['iou'],'difference':b['iou']-a['iou']} for a,b in zip(results['control']['classes'],results['depth']['classes'])],
        equal_augmentation_bytes=True,updates_per_arm=steps,total_research_epochs=2,new_heldout_opened=False)
    write(out/'comparison.json',comparison)
    # Six validation examples: largest non-Wall gain and loss within every scene.
    selected={}
    for scene in sorted({r['scene_id'] for r in frames['control']}):
        ids=[i for i,row in enumerate(frames['control']) if row['scene_id']==scene]
        def delta(i):
            def score(row):
                values=[v['iou'] for v in row['classes'] if v['name'] not in {'other_annotated','wall'} and v['support_pixels']>0]
                return float(np.mean(values))
            return score(frames['depth'][i])-score(frames['control'][i])
        for reason,index in [('largest_nonwall_gain',max(ids,key=lambda i:(delta(i),-i))),('largest_nonwall_loss',min(ids,key=lambda i:(delta(i),i)))]:
            selected.setdefault(index,[]).append(reason)
    predictions={}
    for arm in ['control','depth']:
        model,_,_=load_model(out/arm/'final.pt','cuda');model.eval();predictions[arm]={}
        for start in sorted({i//size*size for i in selected}):
            stop=min(start+size,len(val))
            with torch.inference_mode():masks=model(inputs([val[i][0] for i in range(start,stop)])).argmax(1).cpu().numpy()
            for i in range(start,stop):
                if i not in selected:continue
                m=confusion(val[i][1].numpy(),masks[i-start],17);boundary=boundary_metrics(boundary_counts(val[i][1].numpy(),masks[i-start]))
                if m.tolist()!=frames[arm][i]['confusion_true_rows_predicted_columns'] or boundary!=frames[arm][i]['visible_margin']:
                    raise ValueError('Selected QA does not exactly replay original GPU batch')
                predictions[arm][i]=masks[i-start]
        del model;gc.collect();torch.cuda.empty_cache()
    gallery=out/'validation_comparison';gallery.mkdir();pages=[];cases=[]
    for number,(index,reasons) in enumerate(sorted(selected.items())):
        row=val.samples[index];rgb=val.rgb[index].permute(1,2,0).numpy();truth=val.targets[index].numpy();gt=np.zeros_like(rgb);valid=truth!=255;gt[valid]=PALETTE[truth[valid]]
        panel=Image.new('RGB',(1280,768));draw=ImageDraw.Draw(panel)
        for k,(pixels,label) in enumerate([(rgb,'RGB'),(gt,'GROUND TRUTH'),(PALETTE[predictions['control'][index]],'CONTROL one epoch'),(PALETTE[predictions['depth'][index]],'DEPTH AUX one epoch')]):
            x=(k%2)*640;y=(k//2)*384;panel.paste(Image.fromarray(pixels),(x,y+24));draw.text((x+4,y+4),label,fill='white')
        name=f'case-{number:02}.png';panel.save(gallery/name);cases.append({'index':index,'sample':row,'reasons':reasons,'path':name,'sha256':sha(gallery/name)})
    for start in range(0,len(cases),2):
        page=Image.new('RGB',(1280,768*len(cases[start:start+2])))
        for j,item in enumerate(cases[start:start+2]):
            with Image.open(gallery/item['path']) as im:page.paste(im,(0,j*768))
        name=f'page-{start//2}.png';page.save(gallery/name);pages.append({'path':name,'sha256':sha(gallery/name)})
    write(gallery/'catalog.json',{'status':'GPU_same_batch_replay_exact_pending_visual_review','cases':cases,'pages':pages,'scope':'Full validation used for comparison; per-scene extreme improvements and regressions, not independent trials. No heldout.'})
    verify_sources(contract['sources'])
    for filename,key in [('best.pt','parent_best_sha256'),('config.json','parent_config_sha256'),('state.json','parent_state_sha256'),('provenance.json','parent_provenance_sha256'),('validation.json','parent_validation_sha256'),('validation_frames.jsonl','parent_frames_sha256'),('dataset_statistics.json','parent_statistics_sha256')]:
        if sha(parent/filename)!=config[key]:raise ValueError('Frozen parent changed during research: '+filename)
    result={'status':'completed_first_epoch_AB_pending_assistant_review','GPU_preflight_sha256':sha(out/'GPU_preflight.json'),'comparison_sha256':sha(out/'comparison.json'),
        'QA_catalog_sha256':sha(gallery/'catalog.json'),'automatic_second_epoch':False,'heldout_opened':False,'positive_signal':comparison['positive_signal_for_assistant_review']}
    write(out/'state.json',result);print(json.dumps(result),flush=True)

def main(args):
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    config=read(args.config)
    if config['epochs_per_arm']!=1 or config['preflight_updates']!=32 or config['batch_size']!=8 or config['automatic_extra_epochs']:
        raise ValueError('Only predeclared32-step and one-epoch-per-arm budget allowed')
    if args.output.exists() or not args.output.is_relative_to(ROOT/'artifacts/runs'):raise ValueError('New isolated research output required')
    args.output.mkdir(parents=True);args._created=True
    contract={'config_sha256':sha(args.config),'sources':frozen_sources(config),'previous_batch_contract_sha256':sha(args.wait_for_batch/'contract.json'),
        'previous_QA_contract_sha256':sha(args.wait_for_qa/'contract.json'),'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'One current-session finite wait for existing GPU jobs, then32 discarded preflight updates and two one-epoch research arms. No scheduling, heldout, production adoption or automatic extra trials.'}
    verify_sources(contract['sources'])
    write(args.output/'contract.json',contract);(args.output/'design.json').write_bytes(args.config.read_bytes())
    for relative in contract['sources']:
        path=args.output/'source'/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/relative).read_bytes())
    if os.name=='nt':
        from hold_awake_for_training import creation_filetime
        with (args.output/'awake.log').open('w',encoding='utf-8') as log:
            subprocess.Popen([sys.executable,str(ROOT/'scripts/hold_awake_for_training.py'),'--pid',str(os.getpid()),'--expected-start-filetime',str(creation_filetime(os.getpid())),
                '--max-hours','6','--output',str(args.output/'awake.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    await_previous(args,contract)
    if sha(args.config)!=contract['config_sha256']:raise ValueError('Design changed while waiting')
    execute(config,args.output,contract)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-for-batch',type=Path,required=True);parser.add_argument('--wait-for-qa',type=Path,required=True);args=parser.parse_args();args._created=False
    try:main(args)
    except Exception as exc:
        if args._created:write(args.output/'failure.json',{'error':type(exc).__name__,'message':str(exc),'note':'Preserve partial research. No automatic retry, CPU fallback, reduced batch size or larger trial budget.'})
        raise
