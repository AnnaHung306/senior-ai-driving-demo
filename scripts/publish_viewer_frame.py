"""Publish a verified recorded prediction pointer; no simulator/inference claims."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_viewer_service import ViewerBundle,publish_existing_frame

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    preferred=ROOT/'artifacts/runs/phase7-viewer-bundle-v1-001'
    p.add_argument('--bundle',type=Path,default=preferred if (preferred/'bundle.json').is_file() else ROOT/'artifacts/runs/phase6-viewer-bundle-002')
    p.add_argument('--feed',type=Path,default=ROOT/'artifacts/runs/phase7-feed')
    p.add_argument('--key',required=True);p.add_argument('--heartbeat-seconds',type=float,default=0)
    a=p.parse_args()
    if not 0<=a.heartbeat_seconds<=60:p.error('Bounded diagnostic heartbeat must be 0..60 seconds')
    bundle=ViewerBundle(a.bundle);end=time.monotonic()+a.heartbeat_seconds;count=0
    while True:
        value=publish_existing_frame(bundle,a.feed,a.key);count+=1
        if time.monotonic()>=end:break
        time.sleep(min(.25,max(0,end-time.monotonic())))
    print({'published':a.key,'heartbeat_writes':count,'source_mode':'recorded_inference_replay'},flush=True)
