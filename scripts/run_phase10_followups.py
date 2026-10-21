"""Fixed post-correction probes, sequential and with every failed attempt retained."""
import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    commands=[('performance',[sys.executable,'-X','utf8','scripts/run_phase10_suite.py','--route','phase10-straight-1','--no-GT'])]
    for tag,scenario,extra in [('blocker','blocker',[]),('pedestrian','pedestrian',[]),
        ('bicycle','blocker',['--asset','vehicle.diamondback.century']),('lead','lead_brake',[]),('long-red','signal',['--stop-hold-s','15'])]:
        commands.append((tag,[sys.executable,'-X','utf8','scripts/probe_phase89_traffic.py','--scenario',scenario,
            '--audit','--output',f'artifacts/runs/phase10-drive-{tag}-probe-v1-001',*extra]))
    results=[]
    for name,command in commands:
        log=ROOT/f'artifacts/reports/phase10-followup-{name}-001.log'
        if log.exists():raise FileExistsError(log)
        with log.open('w',encoding='utf8') as f:result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1500)
        results.append({'probe':name,'exit_code':result.returncode,'command':command})
        (ROOT/'artifacts/reports/phase10-followups-v1-001.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
        print(results[-1],flush=True)

if __name__=='__main__':main()
