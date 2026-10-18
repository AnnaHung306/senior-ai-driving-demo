"""One-shot post-batch visual artifacts; never trains or opens a heldout split."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def phase_contacts(run,out):
    from PIL import Image
    source=run/'validation_phase_errors';catalog=read(source/'catalog.json');selected=catalog['selected'];out.mkdir()
    panels=[]
    for row in selected:
        name=row['image']
        if Path(name).name!=name or sha(source/name)!=row['image_sha256']:raise ValueError('Phase gallery image changed')
        panels.append({'path':name,'sha256':row['image_sha256']})
    pages=[]
    for start in range(0,len(panels),8):
        subset=panels[start:start+8];page=Image.new('RGB',(960,220*((len(subset)+1)//2)))
        for i,item in enumerate(subset):
            with Image.open(source/item['path']) as panel:
                if panel.size!=(480,220):raise ValueError('Unexpected saved phase panel dimensions')
                page.paste(panel,((i%2)*480,(i//2)*220))
        p=out/f'page-{start//8:02}.png';page.save(p);pages.append({'path':p.name,'sha256':sha(p)})
    write(out/'receipt.json',{'status':'contact_sheets_from_saved_GPU_phase_decisions','source_catalog_sha256':sha(source/'catalog.json'),
        'source_images':panels,'pages':pages,'scope':'No new inference; actual saved phase wrong/missed cases. Assistant visual review still pending.'})
    (out/'source_catalog.json').write_bytes((source/'catalog.json').read_bytes())


def main(args):
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    if args.output.exists() or not args.output.is_relative_to(ROOT/'artifacts/runs'):raise ValueError('New project QA output required')
    batch=read(args.batch/'contract.json')
    for name in ['segmentation_run','detector_run','light_run']:
        if Path(batch['runs'][name]).resolve()!=getattr(args,name):raise ValueError('QA tuple differs from frozen batch')
    args.output.mkdir(parents=True)
    args._created_output=True
    tools=['review_sunny_segmentation.py','review_saved_detection.py','run_phase3_final_qa.py','hold_awake_for_training.py']
    contract={'batch_contract_sha256':sha(args.batch/'contract.json'),'tools':{name:sha(ROOT/'scripts'/name) for name in tools},
        'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'One-shot current-session wait then validation-only visual artifacts. No training, tuning, new heldout access or scheduling.'}
    write(args.output/'contract.json',contract)
    for name in tools:
        dest=args.output/'tool_snapshot'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes((ROOT/'scripts'/name).read_bytes())
    if os.name=='nt':
        from hold_awake_for_training import creation_filetime
        with (args.output/'awake.log').open('w',encoding='utf-8') as log:
            subprocess.Popen([sys.executable,str(ROOT/'scripts/hold_awake_for_training.py'),'--pid',str(os.getpid()),
                '--expected-start-filetime',str(creation_filetime(os.getpid())),'--max-hours','6','--output',str(args.output/'awake.json')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    deadline=time.monotonic()+6*3600
    write(args.output/'state.json',{'status':'waiting_for_entire_GPU_batch_to_finish'})
    while True:
        if (args.batch/'failure.json').exists():raise RuntimeError('Batch failed; inspect original failure before QA')
        try:state=read(args.batch/'state.json')
        except (OSError,json.JSONDecodeError):state={}
        if state.get('status') in ['quality_failed','phase3_metrics_passed_pending_visual_review']:break
        if time.monotonic()>deadline:raise TimeoutError('Six-hour bound expired; no QA GPU work started')
        time.sleep(10)
    if sha(args.batch/'contract.json')!=contract['batch_contract_sha256'] or sha(args.batch/'export/summary.json')!=state['summary_sha256']:raise ValueError('Batch contract/result changed')
    for relative,digest in batch['sources'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Batch implementation changed before QA: '+relative)
    for name,digest in contract['tools'].items():
        if sha(ROOT/'scripts'/name)!=digest:raise ValueError('QA tool changed during wait')
    proof_deadline=time.monotonic()+180
    while not args.detector_integrity.exists():
        if time.monotonic()>proof_deadline:raise TimeoutError('Completed detector consistency proof missing; inspect file-check waiter')
        time.sleep(5)
    commands=[['review_sunny_segmentation.py','--run',args.segmentation_run,'--root',args.root,'--output',args.output/'segmentation','--device','cuda','--all-classes'],
        ['review_saved_detection.py','--run',args.detector_run,'--root',args.root,'--integrity',args.detector_integrity,'--output',args.output/'detection']]
    for command in commands:
        write(args.output/'state.json',{'status':'rendering_validation_artifacts','tool':command[0]})
        with (args.output/(command[0]+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts'/command[0]),*[str(v) for v in command[1:]]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    phase_contacts(args.light_run,args.output/'phase')
    write(args.output/'state.json',{'status':'completed_artifacts_pending_assistant_visual_review','phase3_quality_pass_not_implied':True,
        'batch_state_sha256':sha(args.batch/'state.json'),'segmentation_catalog_sha256':sha(args.output/'segmentation/catalog.json'),
        'detection_catalog_sha256':sha(args.output/'detection/catalog.json'),'phase_receipt_sha256':sha(args.output/'phase/receipt.json')})
    print(json.dumps(read(args.output/'state.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT/'data/sunny-cross-town-v3-007')
    p.add_argument('--segmentation-run',type=Path,required=True);p.add_argument('--detector-run',type=Path,required=True);p.add_argument('--light-run',type=Path,required=True)
    p.add_argument('--detector-integrity',type=Path,required=True);args=p.parse_args();args._created_output=False
    try:main(args)
    except Exception as exc:
        if args._created_output:write(args.output/'failure.json',{'error':type(exc).__name__,'message':str(exc),'scope':'Preserve partial QA and original batch; no automatic retry or model mutation'})
        raise
