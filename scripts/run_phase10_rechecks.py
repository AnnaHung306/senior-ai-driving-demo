"""Repeat only the two diagnosed engineering failures, then physical fault checks."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    jobs=[('blocker',['scripts/probe_phase89_traffic.py','--scenario','blocker','--audit','--output','artifacts/runs/phase10-drive-blocker-probe-v1-002']),
        ('bicycle',['scripts/probe_phase89_traffic.py','--scenario','blocker','--audit','--asset','vehicle.diamondback.century','--output','artifacts/runs/phase10-drive-bicycle-probe-v1-002']),
        ('sensor-validity',['scripts/run_phase89_drive.py','--fault','sensor_loss','--output','artifacts/runs/phase10-drive-sensor-fault-v1-001']),
        ('camera-timeout',['scripts/run_phase89_drive.py','--fault','camera_timeout','--output','artifacts/runs/phase10-drive-camera-timeout-v1-001']),
        ('setup-fault',['scripts/probe_phase10_setup_fault.py'])]
    results=[]
    for name,args in jobs:
        log=ROOT/f'artifacts/reports/phase10-recheck-{name}-001.log'
        if log.exists():raise FileExistsError(log)
        with log.open('w',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1500)
        results.append({'case':name,'exit_code':r.returncode})
        (ROOT/'artifacts/reports/phase10-rechecks-v1-001.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
        print(results[-1],flush=True)

if __name__=='__main__':main()
