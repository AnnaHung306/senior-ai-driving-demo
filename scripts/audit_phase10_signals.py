"""Offline static stop-waypoint and rendered-lamp association diagnostics, not control."""
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json
from senior_ai.instance_labels import decode_instance_rgb

def projection(point,path):
    p=np.asarray(path,float);d=np.diff(p,axis=0);length=np.linalg.norm(d,axis=1)
    t=np.clip(((np.asarray(point)-p[:-1])*d).sum(1)/np.maximum(length**2,1e-12),0,1)
    distance=np.linalg.norm(p[:-1]+t[:,None]*d-point,axis=1);k=int(distance.argmin())
    return float(distance[k]),float(np.r_[0,np.cumsum(length)][k]+t[k]*length[k])

def main():
    out=ROOT/'artifacts/reports/phase10-signal-association-final.json'
    if out.exists():raise FileExistsError(out)
    signal_map=ROOT/'artifacts/reports/phase10-static-signal-map-002.json';metadata=read(signal_map);counts=Counter();examples=[];crossings=[]
    global_phases={}
    for line in (ROOT/'artifacts/reports/phase10-signal-observer-v1-001.jsonl').read_text().splitlines():
        row=json.loads(line);global_phases[row['readback_frame']]={int(l['id']):l['state'] for l in row['lights']}
    for folder in sorted((ROOT/'artifacts/runs').glob('phase10-drive-*')):
        if not (folder/'report.json').exists():continue
        report=read(folder/'report.json')
        if not report.get('records'):continue
        route=report['route']['points_xy_m'];stops=[]
        for light in metadata['lights']:
            for stop in light['stop_waypoints']:
                distance,progress=projection(stop['xy_m'],route)
                if distance<=1.5:stops.append({'id':light['id'],'progress':progress})
        records=[read(folder/e['file']) for e in report['records']];local={}
        for g in (folder/'audit').glob('[0-9]*.json'):
            row=read(g)
            if row.get('signal_phases_readback'):local[row['frame']+1]={int(k):v for k,v in row['signal_phases_readback'].items()}
        for i,r in enumerate(records):
            before=local.get(r['frame']-1) or global_phases.get(r['frame']-1,{})
            after=local.get(r['frame']+1) or global_phases.get(r['frame']+1,{})
            if i:
                for stop in stops:
                    if records[i-1]['navigation']['progress_m']+2.2<stop['progress']<=r['navigation']['progress_m']+2.2:
                        phase=before.get(stop['id']) if before.get(stop['id'])==after.get(stop['id']) else None
                        crossings.append({'run_id':folder.name,'frame':r['frame'],'lamp_id':stop['id'],'stable_phase':phase,
                            'map_stop_waypoint_proxy_not_painted_line':True})
            path=folder/'audit'/f'{r["frame"]:08d}'/'front-instance.png'
            if not path.exists():continue
            expected={s['id'] for s in stops if -2<=s['progress']-r['navigation']['progress_m']<=18}
            if r['plan']['state']=='SIGNAL_STOP' and not expected:counts['SIGNAL_STOP_samples_without_nearby_map_stop_candidate']+=1
            tags,ids=decode_instance_rgb(np.array(Image.open(path)))
            for signal in r['relevant_RGB_signals']:
                x0,y0,x1,y1=signal['bbox_xyxy'];box=[max(0,int(x0)),max(0,int(y0)),min(640,int(np.ceil(x1))),min(360,int(np.ceil(y1)))]
                region=ids[box[1]:box[3],box[0]:box[2]];mask=tags[box[1]:box[3],box[0]:box[2]]==7
                values,n=np.unique(region[mask],return_counts=True)
                if not len(n) or n.max()<16 or n.max()/n.sum()<.8:counts['unresolvable_RGB_ROI_instance']+=1;continue
                identifier=int(values[n.argmax()]);counts['resolved_ROI_samples']+=1
                if expected and identifier not in expected:
                    counts['possible_other_direction_ROI_samples']+=1
                    if len(examples)<30:examples.append({'run_id':folder.name,'frame':r['frame'],'observed_raw_lamp_id':identifier,
                        'map_stop_candidate_ids':sorted(expected),'RGB_phase':signal['phase'],'plan_state':r['plan']['state']})
    result={'status':'completed_independent_signal_proxy_audit','counts':dict(counts),'possible_association_examples':examples,
        'stop_waypoint_crossings':crossings,'red_proxy_crossings':sum(x['stable_phase']=='Red' for x in crossings),
        'not_certified_traffic_violation_detector':True,'source_map_sha256':sha(signal_map),'tool_sha256':sha(Path(__file__)),
        'limits':['Stop waypoints and 2.2m front offset are a map proxy, not painted stop-line geometry or independent violation certification.',
                  'Lamp-ID association uses >=16 raw semantic pixels with >=80% dominant ID inside observed RGB ROI; other cases unresolved.',
                  'A possible other-direction association needs visual/context review; it is not asserted to be an actual red-light violation.']}
    write_json(out,result);print({'counts':dict(counts),'crossings':len(crossings),'red_proxy':result['red_proxy_crossings']})

if __name__=='__main__':main()
