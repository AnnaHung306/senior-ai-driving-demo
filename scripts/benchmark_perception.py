"""Warm complete eight-view RGB-only perception timings, including phase crops."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from predict_sunny import load_model
from senior_ai.core import write_json
from senior_ai.dataset import read_sensor_frame
from senior_ai.segmentation_data import load_inference_rgb
from senior_ai.sunny_data import sha
from senior_ai.light_model import LightStateNet,crop_rgb
from senior_ai.road_boundaries import visible_margin
from train_sunny import require_idle_gpu
from perception_inputs import fingerprint_rgb_bundle


def main(args):
    require_idle_gpu();detector,task,_=load_model(args.detector,'cuda')
    if task!='detection':raise ValueError('Detector required')
    segmentation,task,_=load_model(args.segmentation,'cuda')
    if task!='segmentation':raise ValueError('Segmentation required')
    state=torch.load(args.light_checkpoint,map_location='cpu',weights_only=False)
    if state['task']!='light_state':raise ValueError('Phase classifier required')
    phase=LightStateNet().cuda().eval();phase.load_state_dict(state['model'])
    frame=read_sensor_frame(args.root,args.manifest);cameras=list(frame['cameras'])
    inputs=fingerprint_rgb_bundle(args.root,args.manifest)
    def forward():
        results=[]
        for camera in cameras:
            rgb=load_inference_rgb(args.root,args.manifest,camera);tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).cuda().float()/255
            detection=detector([tensor])[0];segmentation_ids=segmentation(tensor[None])[0].argmax(0).cpu().numpy();margin=visible_margin(segmentation_ids)
            objects=[]
            for box,label,score in zip(detection['boxes'].cpu().tolist(),detection['labels'].cpu().tolist(),detection['scores'].cpu().tolist()):
                if score<.5:continue
                record={'box':box,'label':label,'score':score}
                if label==7:record['phase']=int(phase(crop_rgb(rgb,box).cuda()[None].float()/255).argmax(1)[0])
                objects.append(record)
            results.append((objects,segmentation_ids,margin))
        return results
    with torch.inference_mode():
        for _ in range(3):forward()
        times=[]
        for _ in range(10):
            torch.cuda.synchronize();started=time.perf_counter();results=forward();torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
    result={**inputs,'detector_sha256':sha(args.detector),'segmentation_sha256':sha(args.segmentation),'phase_sha256':sha(args.light_checkpoint),
        'source_sha256':sha(Path(__file__)),'frame':frame['frame'],'cameras':len(cameras),'warmup_bundles':3,'measured_bundles':10,
        'eight_view_RGB_decode_to_CPU_perception_ms':{'p50':float(np.percentile(times,50)),'p95':float(np.percentile(times,95)),'samples':times},
        'max_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'max_cuda_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
        'limits':'Warm filesystem and repeated one bundle, sequential camera processing. Includes image decode, GPU copies, detector, semantic argmax, visible margin and phase crops; excludes simulator/render/sensor delivery, PNG writes and vehicle control. No 20Hz claim.'}
    write_json(args.output,result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--detector',type=Path,required=True);p.add_argument('--segmentation',type=Path,required=True);p.add_argument('--light-checkpoint',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
