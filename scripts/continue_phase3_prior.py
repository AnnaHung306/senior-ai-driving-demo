"""Same-session continuation: wait for the owned batch, then one bounded study.

This is a native process, not a scheduler. It never opens heldout or starts a
second training batch, and clears its awake request when it exits.
"""
import argparse
import ctypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from hold_awake_for_training import creation_filetime,kernel,process_creation
from train_sunny import ROOT,source_fingerprint
from senior_ai.sunny_data import sha
from senior_ai.core import write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def alive(pid,expected):
    api=kernel();handle=api.OpenProcess(0x1000|0x100000,False,pid)
    if not handle:return False
    try:
        if process_creation(api,handle)!=expected:return False
        result=api.WaitForSingleObject(handle,0)
        if result==258:return True
        if result==0:return False
        raise ctypes.WinError(ctypes.get_last_error())
    finally:api.CloseHandle(handle)


def main(args):
    if args.output.exists():raise FileExistsError('One continuation per output; preserve old receipt')
    if not alive(args.batch_pid,args.batch_start_filetime):raise ValueError('Exact owned batch is not running')
    args.output.mkdir(parents=True)
    tool=ROOT/'scripts/calibrate_segmentation_prior.py';design=ROOT/'configs/segmentation_prior_calibration_v1.json'
    expected={'calibrator':sha(tool),'design':sha(design),'source':source_fingerprint(),'self':sha(Path(__file__))}
    write_json(args.output/'contract.json',{'batch_pid':args.batch_pid,'batch_creation_filetime':args.batch_start_filetime,
        'batch_contract_sha256':sha(args.batch/'contract.json'),'expected':expected,'max_hours':6,
        'scope':'Wait for exact same-session batch to exit; if completed with failing seg validation, perform one separately recorded GPU calibration. No heldout, no new training batch.'})
    log=(args.output/'awake.log').open('a',encoding='utf-8')
    subprocess.Popen([sys.executable,str(ROOT/'scripts/hold_awake_for_training.py'),'--pid',str(os.getpid()),
        '--expected-start-filetime',str(creation_filetime(os.getpid())),'--max-hours','6','--output',str(args.output/'awake.json')],
        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    log.close();start=time.monotonic()
    while alive(args.batch_pid,args.batch_start_filetime):
        if time.monotonic()-start>6*3600:raise TimeoutError('Bounded six-hour wait elapsed; no continuation launched')
        write_json(args.output/'state.json',{'status':'waiting_for_owned_batch','sampled_utc':datetime.now(timezone.utc).isoformat()})
        time.sleep(10)
    state_path=args.batch/'state.json'
    if not state_path.exists():raise ValueError('Batch exited without completed result; inspect original failure, do not start recovery blindly')
    batch_state=read(state_path)
    if batch_state['status']=='phase3_metrics_passed_pending_visual_review':
        write_json(args.output/'state.json',{'status':'skipped_original_metrics_passed_pending_visual_review'});return
    if batch_state['status']!='quality_failed':raise ValueError('Unexpected batch status')
    if sha(tool)!=expected['calibrator'] or sha(design)!=expected['design'] or source_fingerprint()!=expected['source']:
        raise ValueError('Continuation source/design changed during wait')
    if read(args.parent/'state.json')['selected_validation_qualified']:
        write_json(args.output/'state.json',{'status':'skipped_segmentation_already_qualified_other_tasks_failed'});return
    command=[sys.executable,str(tool),'--parent',str(args.parent),'--root',str(args.root),'--design',str(design),'--run',str(args.run)]
    write_json(args.output/'state.json',{'status':'calibrating_after_owned_batch_exit','command':command})
    with (args.output/'calibration.log').open('w',encoding='utf-8') as output:
        process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        write_json(args.output/'process.json',{'pid':process.pid,'creation_filetime':creation_filetime(process.pid),'command':command})
        code=process.wait()
    if code:raise RuntimeError('Calibration failed; inspect preserved log, never overwrite partial run')
    result=read(args.run/'state.json')
    write_json(args.output/'state.json',{'status':'segmentation_validation_recovered_pending_unified_verification' if result['selected_validation_qualified'] else 'calibration_completed_segmentation_quality_failed',
        'calibration_state_sha256':sha(args.run/'state.json'),'selected_alpha':result['selected_alpha'],'no_heldout_opened_by_this_program':True,
        'phase3_complete':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--batch-pid',type=int,required=True);p.add_argument('--batch-start-filetime',type=int,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:main(args)
    except Exception as exc:
        if args.output.exists():write_json(args.output/'failure.json',{'error':type(exc).__name__,'message':str(exc)})
        raise
