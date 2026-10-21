"""Check final public browser geometry, labels, RGB, plans and control against source."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json

def main():
    folder=ROOT/'artifacts/reports/phase10-browser-001';source=folder/'final-pedestrian-v2-frames.json';rows=read(source)
    out=ROOT/'artifacts/reports/phase10-browser-verified-final.json'
    if out.exists():raise FileExistsError(out)
    assert len(rows)==67 and len({r['frame'] for r in rows})==67
    objects=forecasts=0;deltas=[]
    def same(a,b):
        if isinstance(b,dict):
            assert isinstance(a,dict) and set(a)==set(b)
            for k in b:same(a[k],b[k])
        elif isinstance(b,list):
            assert isinstance(a,list) and len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        elif isinstance(b,bool) or b is None:assert a is b
        elif isinstance(b,(int,float)):
            d=abs(a-b);assert np.isfinite(a) and d<=1e-12;deltas.append(d)
        else:assert a==b
    for index,row in enumerate(rows):
        base=ROOT/'artifacts/runs'/row['run_id'];report=read(base/'report.json');entry=report['records'][index]
        assert entry['frame']==row['frame'] and sha(base/entry['file'])==entry['sha256'];record=read(base/entry['file'])
        assert row['display_errors']==0 and row['other_actor_GT_used'] is False
        for observed,expected in [('plan','plan'),('control','control'),('route','navigation'),('RGB','RGB')]:same(row[observed],record[expected])
        assert row['RGB_display']['frame']==str(row['frame']) and row['RGB_display']['width']==640 and row['RGB_display']['height']==360
        assert row['run_id'] in row['RGB_display']['current_src']
        assert len(row['objects'])==len(row['rendered_objects'])==len(record['perception']['objects'])
        for actual,shown,rendered in zip(record['perception']['objects'],row['objects'],row['rendered_objects']):
            assert actual['track_id']==shown['track_id']==rendered['id'];same(shown['center'],actual['center_ego_m'])
            x,y,z=actual['center_ego_m'];L,W,H=actual['size_lwh_m'];same(rendered['position'],[y,z,-x]);same(rendered['dimensions'],[W,H,L]);same(rendered['yaw'],-actual['yaw_rad']);objects+=1
        expected=[o for o in record['perception']['objects'] if o['forecast']['status']=='available'];assert len(expected)==len(row['rendered_forecasts'])
        for actual,rendered in zip(expected,row['rendered_forecasts']):
            assert actual['track_id']==rendered['id'];p=np.asarray(actual['forecast']['positions_ego_xy_m']);xyz=np.c_[p[:,1],np.full(len(p),.12),-p[:,0]]
            np.testing.assert_allclose(np.asarray(rendered['positions']).reshape(-1,3),xyz,atol=1e-5,rtol=0);forecasts+=1
        assert row['object_summary'].startswith('模型物件：') and '可能漏檢或誤判' in row['object_summary']
    assert read(folder/'console-final.json')==[]
    checks=read(folder/'final-checks.json');assert checks['mobile']['client']==checks['mobile']['scroll']
    assert checks['completed_playback_last_frame']==rows[-1]['frame'] and len(checks['eight_RGB'])==8
    write_json(out,{'status':'passed_final_browser_source_control_Three_RGB_and_labels','frames':len(rows),'objects':objects,'forecasts':forecasts,
        'receipts_sha256':sha(source),'checks_sha256':sha(folder/'final-checks.json'),'maximum_JSON_numeric_delta':max(deltas),
        'JSON_numeric_absolute_tolerance':1e-12,'forecast_Float32_absolute_tolerance_m':1e-5,
        'real_200_percent_browser_zoom_verified':False,'formal_quality_passed':False})
    print({'frames':len(rows),'objects':objects,'forecasts':forecasts})

if __name__=='__main__':main()
