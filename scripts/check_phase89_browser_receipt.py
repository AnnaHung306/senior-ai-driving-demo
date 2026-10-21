"""Check exported public DOM diagnostics against immutable captured RGB runs."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.core import write_json
def main():
    folder=ROOT/'artifacts/reports/phase89-browser-001';counts={'frames':0,'objects':0,'forecasts':0};by_file=[]
    deltas=[]
    def same(observed,expected):
        if isinstance(expected,dict):
            assert isinstance(observed,dict) and set(observed)==set(expected)
            for k in expected:same(observed[k],expected[k])
        elif isinstance(expected,list):
            assert isinstance(observed,list) and len(observed)==len(expected)
            for a,b in zip(observed,expected,strict=True):same(a,b)
        elif isinstance(expected,bool) or expected is None:assert observed is expected
        elif isinstance(expected,(int,float)):
            assert isinstance(observed,(int,float)) and not isinstance(observed,bool)
            delta=abs(observed-expected);deltas.append(delta);assert np.isfinite(observed) and delta<=1e-12
        else:assert observed==expected
    for p in sorted(folder.glob('*-frames.json')):
        receipts=read(p);assert receipts
        for row in receipts:
            base=ROOT/'artifacts/runs'/row['run_id'];report=read(base/'report.json');entry=next(x for x in report['records'] if x['frame']==row['frame']);record=read(base/entry['file'])
            assert sha(base/entry['file'])==entry['sha256'] and row['other_actor_GT_used'] is False
            for a,b in [('plan','plan'),('control','control'),('route','navigation'),('RGB','RGB')]:same(row[a],record[b])
            assert row['display_errors']==0 and len(row['objects'])==len(record['perception']['objects'])==len(row['rendered_objects'])
            for obj,shown,rendered in zip(record['perception']['objects'],row['objects'],row['rendered_objects'],strict=True):
                assert obj['track_id']==shown['track_id']==rendered['id'];same(shown['center'],obj['center_ego_m'])
                x,y,z=obj['center_ego_m'];L,W,H=obj['size_lwh_m'];np.testing.assert_allclose(rendered['position'],[y,z,-x],atol=1e-10)
                np.testing.assert_allclose(rendered['dimensions'],[W,H,L],atol=1e-10);same(rendered['yaw'],-obj['yaw_rad']);counts['objects']+=1
            forecasts=[o for o in record['perception']['objects'] if o['forecast']['status']=='available']
            assert len(forecasts)==len(row['rendered_forecasts'])
            for o,rendered in zip(forecasts,row['rendered_forecasts'],strict=True):
                assert rendered['id']==o['track_id'];p3=np.array(o['forecast']['positions_ego_xy_m']);expected=np.c_[p3[:,1],np.full(len(p3),.12),-p3[:,0]]
                np.testing.assert_allclose(np.array(rendered['positions']).reshape(-1,3),expected,atol=1e-5,rtol=0);counts['forecasts']+=1
            counts['frames']+=1
        by_file.append({'file':p.name,'frames':len(receipts),'unique_frames':len({r['frame'] for r in receipts}),'sha256':sha(p)})
        assert len(receipts)==len({r['frame'] for r in receipts})
    checks=read(folder/'checks.json');events={r['test']:r for r in checks}
    assert events['custom_live_arrived']['diagnostic']['report']['engineering_route_passed']
    assert events['full_54_frame_playback']['label'].startswith('54 / 54')
    assert events['mobile_390']['layout']['client']==events['mobile_390']['layout']['scroll']
    assert events['final_seek_after_blur']['index']=='18' and events['final_seek_after_blur']['diagnostic']['frame']==121465
    assert all(c['loaded'] for c in checks if c['test']=='eight_RGB_cameras') and len([c for c in checks if c['test']=='eight_RGB_cameras'])==8
    for name in ['manual_stop_completed','graphics_stop_completed']:
        event=events[name]['diagnostic'];assert event['status']=='cancelled' and event['worker_active'] is False
        assert event['report']['checks']['cleanup_stopped'] and event['report']['cleanup_final_speed_mps']<=.3
    assert events['actual_graphics_loss']['diagnostic']['context_valid'] is False
    assert events['graphics_reload_recovered']['diagnostic']['context_valid'] is True
    assert read(folder/'console-final.json')==[]
    fps=events['custom_live_arrived']['diagnostic']['render_FPS_samples']
    write_json(ROOT/'artifacts/reports/phase89-browser-verified-final.json',{'status':'passed_public_DOM_source_control_Three_transforms_and_forecast_buffers',**counts,
        'files':by_file,'checks_sha256':sha(folder/'checks.json'),'render_median_FPS':float(np.median(fps)),
        'maximum_JSON_numeric_absolute_delta':max(deltas),'JSON_numeric_absolute_tolerance':1e-12,'forecast_Float32_absolute_tolerance_m':1e-5,
        'manual_stop_and_graphics_loss_stop_confirmed':True,'mobile_overflow':False,'formal_quality_passed':False,
        'limits':'Actual DOM/Three transforms and forecast Float32 buffers. Does not certify perception or arbitrary route safety.'})
    print(counts)
if __name__=='__main__':main()
