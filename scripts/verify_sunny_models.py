"""Finite forward/backward and RGB-only inference, with synthetic inputs only."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from senior_ai.core import write_json
from senior_ai.detection_model import make_detector
from senior_ai.segmentation_model import TinyUNet
from senior_ai.light_model import LightStateNet
from senior_ai.segmentation_training import weighted_loss,configure_cuda
from senior_ai.sunny_data import sha,SEGMENTATION_NAMES
from senior_ai.sunny_losses import segmentation_loss
from train_sunny import require_idle_gpu


def main(args):
    if args.device=='cuda': require_idle_gpu();configure_cuda()
    else: torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.manual_seed(19);device=torch.device(args.device);results=[]
    for task in ['detection','segmentation','light_state']:
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        if task=='detection':
            model=make_detector(weights=args.pretrained).to(device).train();x=[torch.rand(3,360,640,device=device) for _ in range(2)]
            targets=[{'boxes':torch.tensor([[20.,20,100,180],[400.,100,408,112]],device=device),
                      'labels':torch.tensor([2,7],device=device),'ignore_boxes':torch.tensor([[200.,100,320,200]],device=device)},
                     {'boxes':torch.empty(0,4,device=device),'labels':torch.empty(0,dtype=torch.long,device=device),'ignore_boxes':torch.tensor([[20.,20,80,80]],device=device)}]
            parts=model(x,targets);loss=sum(parts.values())
        elif task=='segmentation':
            model=TinyUNet(classes=len(SEGMENTATION_NAMES)).to(device).train();x=torch.rand(2,3,360,640,device=device)
            target=torch.randint(0,len(SEGMENTATION_NAMES),(2,360,640),device=device);target[:,:5,:5]=255
            loss=segmentation_loss(model(x),target,torch.ones(len(SEGMENTATION_NAMES),device=device));parts={'weighted_ce_plus_foreground_dice':loss}
        else:
            model=LightStateNet().to(device).train();x=torch.rand(2,3,64,32,device=device)
            loss=torch.nn.functional.cross_entropy(model(x),torch.tensor([1,3],device=device));parts={'ce':loss}
        loss.backward();finite=bool(torch.isfinite(loss).item()) and all(bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None)
        row={'task':task,'finite_loss_and_gradients':finite,'losses':{k:float(v.detach()) for k,v in parts.items()}}
        if task=='detection':row.update({'rpn_ignore_calls':model.rpn.ignore_assignment_calls,'roi_ignore_calls':model.roi_heads.ignore_assignment_calls,
                                         'pyramid_strides':[4,8,16,32,64],'roi_impl':type(model.roi_heads.box_roi_pool).__name__})
        model.eval()
        with torch.inference_mode():prediction=model(x)
        if args.device=='cuda':row['max_cuda_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2;row['max_cuda_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
        results.append(row);del prediction,loss,parts,model,x;gc.collect()
        if args.device=='cuda':torch.cuda.empty_cache()
    result={'status':'passed' if all(r['finite_loss_and_gradients'] for r in results) else 'failed','device':args.device,
            'strict_deterministic':True,'scope':'Synthetic engineering check only; no perception accuracy or training approval',
            'source':{p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'src/senior_ai/detection_model.py',ROOT/'src/senior_ai/segmentation_model.py',ROOT/'src/senior_ai/light_model.py',ROOT/'src/senior_ai/sunny_losses.py']},'results':results}
    write_json(args.output,result);print(json.dumps(result));return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--pretrained',type=Path);p.add_argument('--output',type=Path,required=True);sys.exit(main(p.parse_args()))
