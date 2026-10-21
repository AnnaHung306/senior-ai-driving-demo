"""Bounded sequential audit with preserved failures and no model tuning."""
import argparse,json,sys,threading,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json
from senior_ai.navigation_runtime import drive
from senior_ai.phase10_observer import EvaluationObserver


def main():
    p=argparse.ArgumentParser();p.add_argument('--route');p.add_argument('--no-GT',action='store_true');args=p.parse_args()
    catalog=read(ROOT/'artifacts/runs/phase10-route-catalog-v1-001/catalog.json')
    routes=[r for r in catalog['routes'] if not args.route or r['id']==args.route]
    if not routes:raise ValueError('Unknown case')
    results=[]
    for route in routes:
        name='phase10-drive-'+route['id'].removeprefix('phase10-')+('-performance' if args.no_GT else '-audit')+'-v1-001'
        out=ROOT/'artifacts/runs'/name
        if out.exists():raise FileExistsError('Never reuse results: '+name)
        try:
            report=drive(route,out,threading.Event(),audit_sink=None if args.no_GT else EvaluationObserver())
            results.append({'run_id':name,'status':report['status'],'passed':report['engineering_route_passed'],
                            'frames':report['frames'],'report_sha256':sha(out/'report.json')})
        except Exception as e:
            results.append({'run_id':name,'status':'runner_exception','error':str(e),'traceback':traceback.format_exc()})
        write_json(ROOT/('artifacts/reports/phase10-suite-'+('performance' if args.no_GT else 'audit')+'.json'),results)
        print(json.dumps(results[-1]),flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
