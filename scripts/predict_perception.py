"""Eight-view RGB-only detector + semantic/margin + optional phase inference.

The output contains learned predictions. No simulator, labels, depth or world pose
is read. This is a perception artifact and does not issue vehicle controls.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from predict_sunny import load_model,PALETTE
from senior_ai.core import write_json
from senior_ai.dataset import read_sensor_frame
from senior_ai.segmentation_data import load_inference_rgb
from senior_ai.sunny_data import SEGMENTATION_NAMES,DETECTION_CLASSES,sha
from senior_ai.light_model import LightStateNet,crop_rgb
from senior_ai.light_labels import LIGHT_STATES
from senior_ai.road_boundaries import visible_margin


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    detector,kind,_=load_model(args.detector,args.device)
    if kind!='detection':raise ValueError('Detector checkpoint task mismatch')
    segmentation,kind,_=load_model(args.segmentation,args.device)
    if kind!='segmentation':raise ValueError('Segmentation checkpoint task mismatch')
    phase=None
    if args.light_checkpoint:
        saved=torch.load(args.light_checkpoint,map_location='cpu',weights_only=False)
        if saved['task']!='light_state':raise ValueError('Phase checkpoint task mismatch')
        phase=LightStateNet().to(args.device).eval();phase.load_state_dict(saved['model'])
    frame=read_sensor_frame(args.root,args.manifest);names={v:k for k,v in DETECTION_CLASSES.items()}
    result={'schema_version':1,'source':'learned_prediction','model_inputs':'RGB_only','frame':frame['frame'],
            'timestamp_s':frame['timestamp_s'],'detector_sha256':sha(args.detector),'segmentation_sha256':sha(args.segmentation),
            'phase_sha256':sha(args.light_checkpoint) if args.light_checkpoint else None,
            'segmentation_classes':SEGMENTATION_NAMES,'display_detection_score_threshold':.5,
            'confidence_definition':'Detector score and phase/semantic maximum softmax are uncalibrated','cameras':{},
            'source_sha256':sha(Path(__file__))}
    grid=Image.new('RGB',(1280,1440));started=time.perf_counter()
    with torch.inference_mode():
        for index,camera in enumerate(frame['cameras']):
            rgb=load_inference_rgb(args.root,args.manifest,camera)
            if rgb.shape!=(360,640,3):raise ValueError('Expected trained rig 640x360 RGB')
            tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).to(args.device).float()/255
            detected=detector([tensor])[0];logits=segmentation(tensor[None])[0]
            if not torch.isfinite(logits).all() or not torch.isfinite(detected['boxes']).all() or not torch.isfinite(detected['scores']).all():raise FloatingPointError('Nonfinite perception output')
            confidence,ids=logits.softmax(0).max(0);ids=ids.cpu().numpy().astype(np.uint8);margin=visible_margin(ids)
            folder=args.output/camera;folder.mkdir();Image.fromarray(ids).save(folder/'semantic_ids.png')
            Image.fromarray((margin*255).astype(np.uint8)).save(folder/'visible_margin.png')
            np.save(folder/'semantic_confidence_uncalibrated.npy',confidence.cpu().numpy())
            overlay=np.array(Image.blend(Image.fromarray(rgb),Image.fromarray(PALETTE[ids]),.25));overlay[margin]=[0,255,255]
            image=Image.fromarray(overlay);draw=ImageDraw.Draw(image);objects=[]
            for box,label,score in zip(detected['boxes'].cpu().tolist(),detected['labels'].cpu().tolist(),detected['scores'].cpu().tolist()):
                record={'bbox_xyxy':box,'class_id':label,'class_name':names[label],'score':score};phase_name=None
                if label==7 and score>=.5 and phase is not None:
                    crop=crop_rgb(rgb,box).to(args.device).float()/255;probs=phase(crop[None]).softmax(1)[0]
                    if not torch.isfinite(probs).all():raise FloatingPointError('Nonfinite phase output')
                    certainty,state=probs.max(0);phase_name=LIGHT_STATES[int(state)];record.update({'light_state':phase_name,'phase_softmax_uncalibrated':float(certainty)})
                objects.append(record)
                if score>=.5:
                    draw.rectangle(box,outline='yellow',width=2);draw.text((box[0],max(15,box[1]-12)),f'{names[label]} {score:.2f}'+(' '+phase_name if phase_name else ''),fill='yellow')
            draw.text((4,4),'LEARNED '+camera+' / cyan=visible road margin',fill='red');image.save(folder/'prediction.png');grid.paste(image,((index%2)*640,(index//2)*360))
            result['cameras'][camera]={'objects':objects,'semantic_ids':camera+'/semantic_ids.png','visible_margin':camera+'/visible_margin.png',
                                       'confidence':camera+'/semantic_confidence_uncalibrated.npy'}
    if args.device=='cuda':torch.cuda.synchronize()
    result['artifact_generation_wall_ms']=(time.perf_counter()-started)*1000
    result['timing_note']='Cold first inference plus PNG/npy writing; use benchmark_sunny.py for warmed timing. Excludes simulator sensors and controls.'
    grid.save(args.output/'surround.png');write_json(args.output/'perception.json',result)
    print(json.dumps({'source':result['source'],'frame':result['frame'],'cameras':len(result['cameras']),'output':str(args.output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--detector',type=Path,required=True);p.add_argument('--segmentation',type=Path,required=True);p.add_argument('--light-checkpoint',type=Path)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');main(p.parse_args())
