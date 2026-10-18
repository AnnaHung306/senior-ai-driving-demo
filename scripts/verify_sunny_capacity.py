"""GPU capacity/finite update check for a frozen detector experiment, not accuracy."""
import argparse
import json
from pathlib import Path
import time
from train_sunny import ROOT,build_model,make_optimizer,configure_cuda,seed_all,require_idle_gpu
import torch
from senior_ai.core import write_json
from senior_ai.sunny_data import sha
from predict_sunny import load_model


def main(args):
    require_idle_gpu();configure_cuda();config=json.loads(args.config.read_text(encoding='utf-8'));seed_all(config['seed']);c=config['detection']
    torch.cuda.reset_peak_memory_stats();model=build_model('detection',config).cuda();optimizer=make_optimizer(model,c)
    images=[torch.rand(3,360,640,device='cuda') for _ in range(c['batch_size'])]
    targets=[{'boxes':torch.tensor([[120.,60,126,74],[300.,80,340,190]],device='cuda'),
        'labels':torch.tensor([7,3],device='cuda'),'ignore_boxes':torch.tensor([[20.,200,50,220]],device='cuda')} for _ in images]
    steps=[]
    for i in range(3):
        model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        parts=model(images,targets);loss=sum(parts.values());loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite capacity loss')
        optimizer.step();torch.cuda.synchronize();steps.append({'step':i,'loss':float(loss.detach()),'gradient_norm':float(norm),'wall_ms':(time.perf_counter()-start)*1000})
    model.eval()
    with torch.inference_mode():before=model(images)
    # Exercise the production loader, including non-tensor resize configuration.
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    checkpoint=args.output.with_suffix('.synthetic.pt')
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'task':'detection','config':config,'model':state},checkpoint)
    reloaded,_,_=load_model(checkpoint,'cuda')
    with torch.inference_mode():after=reloaded(images)
    identical=all(torch.equal(a[key],b[key]) for a,b in zip(before,after) for key in a)
    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();total=torch.cuda.get_device_properties(0).total_memory
    result={'status':'passed' if identical and allocated<total*.92 else 'failed','scope':'Synthetic engineering capacity only, no quality metric or training update retained',
        'config_sha256':sha(args.config),'source_sha256':sha(Path(__file__)),'device':torch.cuda.get_device_name(),'batch_size':c['batch_size'],
        'internal_min_size':c.get('min_size',360),'internal_max_size':c.get('max_size',640),'input_wh':[640,360],
        'strict_deterministic':torch.are_deterministic_algorithms_enabled(),'steps':steps,'checkpoint_reload_identical':identical,
        'synthetic_checkpoint_sha256':sha(checkpoint),'synthetic_checkpoint_scope':'Capacity-only dummy updates, never a selected trained model',
        'max_cuda_allocated_mib':allocated/1024**2,'max_cuda_reserved_mib':reserved/1024**2,'device_total_mib':total/1024**2}
    write_json(args.output,result);print(json.dumps(result),flush=True);return 0 if result['status']=='passed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);raise SystemExit(main(p.parse_args()))
