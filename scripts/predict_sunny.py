"""Learned per-camera inference from RGB manifest only; no labels or CARLA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.core import write_json
from senior_ai.detection_model import make_detector
from senior_ai.segmentation_model import TinyUNet
from senior_ai.mobile_segmentation import MobileUNet
from senior_ai.segmentation_data import load_inference_rgb
from senior_ai.sunny_data import SEGMENTATION_NAMES,DETECTION_CLASSES,sha
from senior_ai.light_model import LightStateNet,crop_rgb
from senior_ai.light_labels import LIGHT_STATES
from senior_ai.road_boundaries import visible_margin

PALETTE=np.array([(70,70,70),(128,64,128),(244,35,232),(220,20,60),(255,0,0),(0,0,142),
    (0,0,70),(0,60,100),(0,0,230),(119,11,32),(255,255,0),(250,170,30),(102,102,156),
    (190,153,153),(153,153,153),(255,90,0),(152,251,152)],dtype=np.uint8)


def load_model(checkpoint,device):
    if str(device).startswith('cuda'):
        from senior_ai.segmentation_training import configure_cuda
        configure_cuda()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False); task=saved['task']
    if task not in ['detection','segmentation']:raise ValueError('Expected detector or segmentation checkpoint')
    if task=='detection':model=make_detector()
    else:
        c=saved['config']['segmentation']
        model=MobileUNet(classes=len(SEGMENTATION_NAMES),precision=c['precision']) if c.get('architecture')=='mobilenet_unet' else TinyUNet(width=c['width'],classes=len(SEGMENTATION_NAMES))
    model.load_state_dict(saved['model']);model.to(device).eval()
    return model,task,saved


def main(args):
    args.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(4)
    model,task,saved=load_model(args.checkpoint,args.device)
    light_model=None
    if args.light_checkpoint:
        if task!='detection': raise ValueError('Light-state head requires a detector')
        phase=torch.load(args.light_checkpoint,map_location='cpu',weights_only=False)
        if phase['task']!='light_state': raise ValueError('Wrong phase checkpoint task')
        light_model=LightStateNet().to(args.device).eval();light_model.load_state_dict(phase['model'])
    rgb=load_inference_rgb(args.root,args.manifest,args.camera)
    if rgb.shape!=(360,640,3):raise ValueError('Inference requires the trained 640x360 RGB rig resolution')
    tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).to(args.device).float()/255
    if args.device=='cuda': torch.cuda.synchronize()
    started=time.perf_counter()
    with torch.inference_mode(): result=model([tensor])[0] if task=='detection' else model(tensor[None])[0]
    if args.device=='cuda': torch.cuda.synchronize()
    elapsed=(time.perf_counter()-started)*1000
    metadata={'source':'learned_prediction','task':task,'RGB_only':True,'camera':args.camera,
              'checkpoint_sha256':sha(args.checkpoint),'selected_epoch':saved['epoch'],'network_first_forward_ms':elapsed,
              'timing_note':'includes cold model kernels; not steady-state or eight-camera real-time claim'}
    if task=='detection':
        names={v:k for k,v in DETECTION_CLASSES.items()}; overlay=Image.fromarray(rgb);draw=ImageDraw.Draw(overlay);objects=[]
        for box,label,score in zip(result['boxes'].cpu().tolist(),result['labels'].cpu().tolist(),result['scores'].cpu().tolist()):
            record={'bbox_xyxy':box,'class_id':label,'class_name':names[label],'score':score}
            phase_name=None
            if label==7 and score>=.5 and light_model is not None:
                crop=crop_rgb(rgb,box).to(args.device).float()/255
                with torch.inference_mode(): probs=light_model(crop[None]).softmax(1)[0]
                confidence,state=probs.max(0);phase_name=LIGHT_STATES[int(state)]
                record.update({'light_state':phase_name,'phase_softmax_uncalibrated':float(confidence)})
            objects.append(record)
            if score>=.5:
                draw.rectangle(box,outline='yellow',width=2);draw.text((box[0],max(0,box[1]-12)),f'{names[label]} {score:.2f}'+(' '+phase_name if phase_name else ''),fill='yellow')
        draw.text((5,5),'LEARNED PREDICTION',fill='red');overlay.save(args.output/'prediction.png');metadata['objects']=objects
        if light_model is not None: metadata['light_checkpoint_sha256']=sha(args.light_checkpoint)
    else:
        confidence,prediction=result.softmax(0).max(0); ids=prediction.cpu().numpy().astype(np.uint8)
        Image.fromarray(ids).save(args.output/'class_ids.png');np.save(args.output/'softmax_max_uncalibrated.npy',confidence.cpu().numpy())
        Image.fromarray((visible_margin(ids)*255).astype(np.uint8)).save(args.output/'visible_road_margin.png')
        metadata['visible_margin_definition']='Derived RGB prediction interface: road/road line adjoining sidewalk or terrain; no hidden curb or free-space inference'
        color=PALETTE[ids]; panel=Image.new('RGB',(1280,360));panel.paste(Image.fromarray(rgb),(0,0));panel.paste(Image.fromarray(color),(640,0))
        ImageDraw.Draw(panel).text((645,5),'LEARNED PREDICTION',fill='white');panel.save(args.output/'prediction.png');metadata['classes']=SEGMENTATION_NAMES
    write_json(args.output/'prediction.json',metadata);print(json.dumps(metadata if task!='detection' else {k:v for k,v in metadata.items() if k!='objects'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--camera',default='front');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--light-checkpoint',type=Path);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');main(p.parse_args())
