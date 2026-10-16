"""RGB-only synchronized inference timing; never extrapolate to driving rate."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from predict_sunny import load_model,load_inference_rgb,sha
from senior_ai.core import write_json


def main(args):
    torch.set_num_threads(4);model,task,saved=load_model(args.checkpoint,'cuda')
    cameras=list(json.loads((args.root/args.manifest).read_text(encoding='utf-8'))['cameras'])
    rgb=load_inference_rgb(args.root,args.manifest,cameras[0]); tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).cuda().float()/255
    def forward(t):
        result=model([t])[0] if task=='detection' else model(t[None])[0]
        if task=='detection': return {k:v.cpu() for k,v in result.items()}
        return result.argmax(0).cpu()
    with torch.inference_mode():
        for _ in range(10): forward(tensor)
        network=[]; end_to_end=[]
        for _ in range(50):
            torch.cuda.synchronize();start=time.perf_counter();forward(tensor);torch.cuda.synchronize();network.append((time.perf_counter()-start)*1000)
        for _ in range(50):
            start=time.perf_counter();rgb=load_inference_rgb(args.root,args.manifest,cameras[0]);t=torch.from_numpy(rgb.transpose(2,0,1).copy()).cuda().float()/255;forward(t);torch.cuda.synchronize();end_to_end.append((time.perf_counter()-start)*1000)
        start=time.perf_counter()
        for camera in cameras:
            rgb=load_inference_rgb(args.root,args.manifest,camera);t=torch.from_numpy(rgb.transpose(2,0,1).copy()).cuda().float()/255;forward(t)
        torch.cuda.synchronize();surround=(time.perf_counter()-start)*1000
    result={'task':task,'checkpoint_sha256':sha(args.checkpoint),'source_sha256':sha(Path(__file__)),
            'samples':50,'warmup':10,'network_and_result_to_cpu_ms':{'p50':float(np.percentile(network,50)),'p95':float(np.percentile(network,95))},
            'warm_filesystem_RGB_manifest_to_result_ms':{'p50':float(np.percentile(end_to_end,50)),'p95':float(np.percentile(end_to_end,95))},
            'one_eight_camera_sequential_measurement_ms':surround,
            'limits':'Excludes simulator/render/sensor delivery, output encoding/writes and control. Warm filesystem, one bundle for surround; not a 20Hz or driving guarantee.'}
    write_json(args.output,result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
