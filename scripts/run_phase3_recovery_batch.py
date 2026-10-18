"""Sequential GPU recovery and unified validation; never open heldout on val failure."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from train_sunny import ROOT,source_fingerprint,require_idle_gpu
from evaluate_sunny import quality
from senior_ai.core import write_json
from senior_ai.sunny_data import sha
from audit_light_supplement import verify_supplement
from perception_inputs import fingerprint_rgb_bundle
from export_sunny_results import final_receipt


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def acquire_batch_lock(path):
    handle=path.open('a+b')
    if path.stat().st_size==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('Another process owns this recovery batch')
    return handle


def qualified(run):
    state=read(run/'state.json')
    if state['status']!='completed' or not state['checkpoint_reload_identical'] or sha(run/'best.pt')!=state['best_sha256']:raise ValueError('Incomplete/changed model: '+run.name)
    for file,key in [('validation.json','validation_sha256'),('validation_frames.jsonl','validation_frames_sha256')]:
        if sha(run/file)!=state[key]:raise ValueError('Validation bytes changed: '+run.name)
    result=quality(state['task'],read(run/'validation.json'),read(run/'config.json')['quality_gates'])
    write_json(run/'validation_quality.json',result);return result


def main(args):
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    require_idle_gpu()
    if args.output.exists() and not args.resume:raise FileExistsError('Use --resume for this batch or a new output version')
    args.output.mkdir(parents=True,exist_ok=True)
    batch_lock=acquire_batch_lock(args.output/'batch.lock')  # OS releases on exit, even after interruption.
    sources={**source_fingerprint()['files']}
    for name in ['train_light_state.py','validate_light_joint.py','evaluate_light_state.py','predict_perception.py','benchmark_sunny.py','benchmark_perception.py','perception_inputs.py','export_sunny_results.py','verify_sunny_capacity.py','run_phase3_recovery_batch.py','hold_awake_for_training.py']:
        sources['scripts/'+name]=sha(ROOT/'scripts'/name)
    contract={'sources':sources,'configs':{str(path):sha(path) for path in [args.segmentation_config,args.detection_config,args.light_config]},
        'main_audit_sha256':sha(args.root/'audit.json'),'phase_supplement':verify_supplement(args.supplement_root,sha(args.root/'audit.json')),
        'inputs':fingerprint_rgb_bundle(args.rgb_root,'inputs/frames/frame.json'),
        'runs':{key:str(getattr(args,key)) for key in ['segmentation_run','detector_run','light_run']}}
    if (args.output/'contract.json').exists():
        if read(args.output/'contract.json')!=contract:raise ValueError('Batch source/config/data/input contract changed; use frozen sources to recover')
    else:write_json(args.output/'contract.json',contract)
    stages=read(args.output/'stages.json') if (args.output/'stages.json').exists() else []
    # The existing bounded helper clears its thread-owned request on parent exit.
    if os.name=='nt':
        from hold_awake_for_training import creation_filetime
        guardian_log=(args.output/'awake.log').open('a',encoding='utf-8')
        subprocess.Popen([sys.executable,str(ROOT/'scripts/hold_awake_for_training.py'),'--pid',str(os.getpid()),
            '--expected-start-filetime',str(creation_filetime(os.getpid())),'--max-hours','6','--output',str(args.output/'awake.json')],
            cwd=ROOT,stdout=guardian_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        guardian_log.close()

    def stage(name,command,allowed=(0,),outputs=()):
        old=next((s for s in reversed(stages) if s['name']==name and s.get('finished')),None)
        if old and old['return_code'] in allowed and old.get('outputs') and all(p.exists() and sha(p)==old['outputs'].get(str(p)) for p in outputs):
            print(json.dumps({'stage':name,'status':'already_completed_verified'}),flush=True);return old['return_code']
        item={'name':name,'command':[str(v) for v in command],'started_utc':datetime.now(timezone.utc).isoformat(),'finished':False}
        stages.append(item);write_json(args.output/'stages.json',stages)
        print(json.dumps({'stage':name,'status':'starting'}),flush=True);start=time.monotonic()
        with (args.output/(name+'.log')).open('a',encoding='utf-8') as log:
            process=subprocess.Popen([str(v) for v in command],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            item['pid']=process.pid;write_json(args.output/'stages.json',stages)
            for line in process.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
            code=process.wait()
        item.update(finished=True,return_code=code,wall_s=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat(),
            outputs={str(p):sha(p) for p in outputs if p.exists()},log_sha256=sha(args.output/(name+'.log')))
        write_json(args.output/'stages.json',stages)
        if code not in allowed:raise RuntimeError(f'{name} failed with return code{code}; see preserved stage log')
        return code

    py=sys.executable
    stage('gpu-regression',[py,'-m','unittest','discover','-s','tests','-v'],outputs=[args.output/'gpu-regression.log'])
    stage('detector-capacity',[py,ROOT/'scripts/verify_sunny_capacity.py','--config',args.detection_config,'--output',args.output/'detector-capacity.json'],outputs=[args.output/'detector-capacity.json'])
    for task,run,config in [('segmentation',args.segmentation_run,args.segmentation_config),('detection',args.detector_run,args.detection_config)]:
        if run.exists() and (run/'state.json').exists() and read(run/'state.json')['status']=='completed':
            qualified(run)
            if sha(run/'config.json')!=sha(config) or read(run/'provenance.json')['source']['files']!=source_fingerprint()['files']:raise ValueError('Completed run does not match this batch')
        else:
            command=[py,ROOT/'scripts/train_sunny.py','--task',task,'--config',config,'--run',run,'--dataset-root',args.root]
            if run.exists():
                if not (run/'last.pt').exists():raise ValueError('No complete epoch to resume; preserve failed run and choose a new version')
                command.append('--resume')
            stage('train-'+task,command,outputs=[run/'state.json',run/'best.pt',run/'validation.json',run/'validation_frames.jsonl'])
    if not (args.light_run/'state.json').exists():
        stage('train-phase',[py,ROOT/'scripts/train_light_state.py','--root',args.root,'--config',args.light_config,'--run',args.light_run],
            outputs=[args.light_run/'state.json',args.light_run/'best.pt',args.light_run/'validation.json'])
    phase=read(args.light_run/'state.json')
    if phase['status']!='completed' or sha(args.light_run/'best.pt')!=phase['best_sha256'] or sha(args.light_run/'config.json')!=sha(args.light_config):raise ValueError('Incomplete/changed phase run')
    stage('validation-joint',[py,args.light_run/'source/scripts/validate_light_joint.py','--run',args.light_run,'--detector-run',args.detector_run,'--root',args.root],
        allowed=(0,2),outputs=[args.light_run/'validation_joint.json',args.light_run/'validation_joint_quality.json'])
    gates={'segmentation':qualified(args.segmentation_run),'detection':qualified(args.detector_run),
        'phase_crops':phase['validation_quality'],'phase_joint':read(args.light_run/'validation_joint_quality.json')}
    write_json(args.output/'validation_gates.json',gates)
    val_passed=all(g['passed'] for g in gates.values())
    print(json.dumps({'validation_qualified':val_passed,'failed_checks':{task:[k for k,v in g['checks'].items() if not v] for task,g in gates.items()}}),flush=True)
    if val_passed:
        for task,run in [('detection',args.detector_run),('segmentation',args.segmentation_run)]:
            if (run/'final_evaluation.json').exists():final_receipt(run,read(run/'state.json'))
            else:
                stage('heldout-'+task,[py,run/'source/scripts/evaluate_sunny.py','--run',run,'--dataset-root',args.root],allowed=(0,2),outputs=[run/'final_evaluation.json'])
        if (args.light_run/'final_evaluation.json').exists():final_receipt(args.light_run,phase)
        if read(args.detector_run/'final_evaluation.json')['status']=='passed' and not (args.light_run/'final_evaluation.json').exists():
            stage('heldout-phase',[py,args.light_run/'source/scripts/evaluate_light_state.py','--root',args.root,'--run',args.light_run,
                '--detector-run',args.detector_run,'--supplement-root',args.supplement_root],allowed=(0,2),outputs=[args.light_run/'final_evaluation.json'])
    else:
        print('Validation failed: all heldout datasets remain unopened for this candidate tuple.',flush=True)
    demo=args.output/'rgb-only-demo';manifest='inputs/frames/frame.json'
    stage('rgb-only-demo',[py,ROOT/'scripts/predict_perception.py','--detector',args.detector_run/'best.pt','--segmentation',args.segmentation_run/'best.pt',
        '--light-checkpoint',args.light_run/'best.pt','--root',args.rgb_root,'--manifest',manifest,'--output',demo],outputs=[demo/'perception.json',demo/'surround.png'])
    benchmarks=[]
    for task,run in [('detection',args.detector_run),('segmentation',args.segmentation_run)]:
        output=args.output/(task+'-benchmark.json');benchmarks.append(output)
        stage('benchmark-'+task,[py,ROOT/'scripts/benchmark_sunny.py','--checkpoint',run/'best.pt','--root',args.rgb_root,'--manifest',manifest,'--output',output],outputs=[output])
    integrated=args.output/'integrated-benchmark.json';benchmarks.append(integrated)
    stage('benchmark-integrated',[py,ROOT/'scripts/benchmark_perception.py','--detector',args.detector_run/'best.pt','--segmentation',args.segmentation_run/'best.pt',
        '--light-checkpoint',args.light_run/'best.pt','--root',args.rgb_root,'--manifest',manifest,'--output',integrated],outputs=[integrated])
    stage('export',[py,ROOT/'scripts/export_sunny_results.py','--detector-run',args.detector_run,'--segmentation-run',args.segmentation_run,
        '--light-run',args.light_run,'--root',args.root,'--demo',demo,'--benchmark',*benchmarks,'--output',args.output/'export'],outputs=[args.output/'export/summary.json'])
    summary=read(args.output/'export/summary.json');write_json(args.output/'state.json',{'status':'phase3_metrics_passed_pending_visual_review' if summary['phase3_complete'] else 'quality_failed',
        'phase3_metrics_passed':summary['phase3_complete'],'validation_qualified':val_passed,'summary_sha256':sha(args.output/'export/summary.json')})
    return 0 if summary['phase3_complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'data/sunny-cross-town-v3-007')
    p.add_argument('--supplement-root',type=Path,default=ROOT/'data/light-lockbox-supplement-v1-002');p.add_argument('--rgb-root',type=Path,default=ROOT/'artifacts/runs/sunny-rgb-only-input-v1-001')
    p.add_argument('--segmentation-config',type=Path,required=True);p.add_argument('--detection-config',type=Path,default=ROOT/'configs/sunny_detection_v5_1.json')
    p.add_argument('--light-config',type=Path,default=ROOT/'configs/light_state_v1_1.json')
    p.add_argument('--segmentation-run',type=Path,default=ROOT/'artifacts/runs/sunny-segmentation-v5-001');p.add_argument('--detector-run',type=Path,default=ROOT/'artifacts/runs/sunny-detection-v5-001')
    p.add_argument('--light-run',type=Path,default=ROOT/'artifacts/runs/sunny-light-state-v1-003');p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:raise SystemExit(main(args))
    except Exception as exc:
        if args.output.exists():write_json(args.output/'failure.json',{'error':type(exc).__name__,'message':str(exc),'note':'Preserve existing runs/markers; inspect stages.json before an exact-source resume.'})
        raise
