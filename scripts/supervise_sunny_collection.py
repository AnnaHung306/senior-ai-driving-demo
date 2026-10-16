"""Bounded client recovery ONLY between successfully cleaned episodes.

Native CARLA client assertions can bypass Python finally/exception handlers.
Never retry a failed/incomplete scene: retain it for diagnosis/new capture ID.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from collect_dataset import ROOT,sha
from senior_ai.core import write_json


def snapshot(root):
    if not root.exists():return [],None
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'scenes').glob('*/report.json')]
    report=json.loads((root/'report.json').read_text(encoding='utf-8')) if (root/'report.json').exists() else None
    return rows,report


def main(args):
    root=args.output.resolve()
    if not root.is_relative_to((ROOT/'data').resolve()):raise ValueError('Acquisition output must stay under project data')
    log_root=ROOT/'artifacts/reports';name=root.name
    monitor=log_root/(name+'-execution.json');history=[]
    command=[sys.executable,str(ROOT/'scripts/collect_sunny_dataset.py'),'--plan',str(args.plan.resolve()),'--output',str(root)]
    if args.reuse_root:command+=['--reuse-root',str(args.reuse_root.resolve())]
    for attempt in range(1,4):
        log=log_root/f'{name}-client-attempt-{attempt:02d}.log'
        with log.open('xb') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            while child.poll() is None:
                rows,report=snapshot(root)
                state={'status':'running','pid':child.pid,'attempt':attempt,'updated_utc':datetime.now(timezone.utc).isoformat(),
                       'completed_scenes':sum(r['status']=='passed' for r in rows),'active_scenes':[r['scene_id'] for r in rows if r['status']!='passed'],
                       'observed_images':sum(r['images'] for r in rows),'log':str(log),'history':history}
                write_json(monitor,state);time.sleep(5)
        rows,report=snapshot(root);code=child.returncode
        history.append({'attempt':attempt,'returncode':code,'log':str(log),'log_sha256':sha(log)})
        if code==0 and report and report['status']=='passed':
            write_json(monitor,{'status':'completed','updated_utc':datetime.now(timezone.utc).isoformat(),'completed_scenes':len(rows),'history':history});return 0
        # Cleaned complete scenes and no started next scene: reconnecting a new
        # client is safe. A per-scene failure is never hidden or retried here.
        recoverable=report and report['status']=='in_progress' and rows and all(r['status']=='passed' and r['checks'].get('cleanup') for r in rows)
        if not recoverable:
            write_json(monitor,{'status':'needs_diagnosis','updated_utc':datetime.now(timezone.utc).isoformat(),'returncode':code,'history':history});return 1
    write_json(monitor,{'status':'client_recovery_exhausted','updated_utc':datetime.now(timezone.utc).isoformat(),'history':history});return 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-root',type=Path);sys.exit(main(p.parse_args()))
