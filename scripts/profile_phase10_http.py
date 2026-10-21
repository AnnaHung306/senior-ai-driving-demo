"""Measure actual loopback JSON/PNG sizes and end-to-end response time."""
import hashlib,json,time,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE='http://127.0.0.1:8768'

def request(path,body=None):
    req=urllib.request.Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers={} if body is None else {'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as r:return r.read()

def main():
    out=ROOT/'artifacts/reports/phase10-HTTP-performance-final.json'
    if out.exists():raise FileExistsError(out)
    run='phase10-drive-pedestrian-probe-v1-001';request('/api/navigation/replay',{'run_id':run,'index':18})
    state=json.loads(request('/api/navigation/state'));frame=state['latest']['frame'];digest=state['latest']['RGB']['front']['sha256'];rows=[]
    for kind,path in [('state','/api/navigation/state'),('RGB',f'/api/navigation/rgb?run_id={run}&frame={frame}&camera=front')]:
        for repeat in range(20):
            started=time.perf_counter();data=request(path);elapsed=(time.perf_counter()-started)*1000
            if kind=='RGB':assert hashlib.sha256(data).hexdigest()==digest
            rows.append({'kind':kind,'repeat':repeat,'response_bytes':len(data),'request_through_read_ms':elapsed})
    report={'status':'passed_real_loopback_timing_and_RGB_bytes','run_id':run,'frame':frame,'rows':rows,
        'summary':{kind:{'response_bytes':next(r['response_bytes'] for r in rows if r['kind']==kind),
            'p50_ms':float(np.median([r['request_through_read_ms'] for r in rows if r['kind']==kind])),
            'p95_ms':float(np.percentile([r['request_through_read_ms'] for r in rows if r['kind']==kind],95))} for kind in ['state','RGB']},
        'scope':'same host warm filesystem, CARLA/inference off, includes HTTP+serialization/read+PNG SHA, no public network or browser GPU time'}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(report['summary'])

if __name__=='__main__':main()
