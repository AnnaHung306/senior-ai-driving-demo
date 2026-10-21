"""Sequential GPU/CARLA tests; each has an immutable output and bounded runtime."""
import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
jobs=[('unseen-straight2',['scripts/run_phase89_drive.py','--route','straight-2'],'phase89-drive-unseen-straight2-v1-001'),
      ('blocker-probe',['scripts/probe_phase89_traffic.py','--scenario','blocker'],'phase89-drive-blocker-probe-v1-001'),
      ('pedestrian-probe',['scripts/probe_phase89_traffic.py','--scenario','pedestrian'],'phase89-drive-pedestrian-probe-v1-001'),
      ('sensor-fault',['scripts/run_phase89_drive.py','--route','straight-1','--fault','sensor_loss'],'phase89-drive-sensor-fault-v1-001')]
if __name__=='__main__':
    if any((ROOT/'artifacts/runs'/name).exists() for _,_,name in jobs):raise FileExistsError('New suite paths required; never overwrite runs')
    results=[]
    for label,command,name in jobs:
        log=ROOT/'artifacts/reports'/f'phase89-{label}-001.log'
        print('BEGIN '+label,flush=True)
        with log.open('wb') as out:
            child=subprocess.run([sys.executable,'-X','utf8',*command,'--output',str(ROOT/'artifacts/runs'/name)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=1250)
        results.append({'job':label,'exit_code':child.returncode,'run':name});print(results[-1],flush=True)
        if child.returncode:break
    (ROOT/'artifacts/reports/phase89-suite-001.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
