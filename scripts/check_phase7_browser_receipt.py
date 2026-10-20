"""Compare public DOM/WebGL-buffer observations to all real and synthetic sources."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.core import write_json

def main():
    folder=ROOT/'artifacts/reports/phase7-browser-001';bundle=ROOT/'artifacts/runs/phase7-viewer-bundle-v1-001';manifest=read(bundle/'bundle.json')
    records=read(folder/'frames.json');rows={r['key']:r for r in manifest['frames']};assert len(records)==len(rows)==240 and {r['key'] for r in records}==set(rows)
    counts={'objects':0,'forecasts':0}
    for row in records:
        frame=read(bundle/rows[row['key']]['file']);assert row['error'] is None and row['source']==frame['world_source_sha256'] and row['motion_source']==frame['motion_prediction_source_sha256']
        assert row['object_count']==len(frame['objects'])
        available=[o for o in frame['objects'] if o['forecast']['status']=='available'];assert len(row['rendered_forecasts'])==len(available)
        for observed,source,transform in zip(row['objects'],frame['objects'],row['transforms'],strict=True):
            assert observed['track_id']==source['track_id']==transform['track_id']
            for name in ['motion','forecast','risk','lifecycle']:assert observed[name]==source[name]
            x,y,z=source['center_ego_m'];L,W,H=source['size_lwh_m'];np.testing.assert_allclose(transform['position_three'],[y,z,-x],atol=1e-12)
            np.testing.assert_allclose(transform['dimensions_three'],[W,H,L],atol=1e-12);assert transform['yaw_y_three']==-source['yaw_rad'];counts['objects']+=1
        for render,o in zip(row['rendered_forecasts'],available,strict=True):
            assert render['track_id']==o['track_id'];p=np.array(o['forecast']['positions_ego_xy_m']);expected=np.stack([p[:,1],np.full(len(p),.11),-p[:,0]],axis=1)
            np.testing.assert_allclose(render['positions_three'],expected,rtol=0,atol=1e-5);counts['forecasts']+=1
    checks=read(folder/'checks.json');plays=[c for c in checks if c['test']=='full_timestamp_playback'];assert len(plays)==4
    events={c['test']:c for c in checks}
    assert events['actual_WebGL_loss_final']['diagnostic']['context_valid'] is False
    recovered=events['context_reload_recovers_final']['diagnostic']
    assert recovered['context_valid'] is True and recovered['error'] is None and recovered['frame_key']=='town01-00/166989'
    hidden=events['prediction_off_final']['diagnostic']
    assert hidden['motion_layers']['prediction'] is False and not hidden['rendered_forecasts']
    cameras=[c for c in checks if c['test']=='eight_RGB_cameras_final']
    assert len(cameras)==8 and len({c['camera'] for c in cameras})==8 and all(c['loaded'] for c in cameras)
    for name in ['main_mobile_390','lab_mobile_viewbox_corrected']:
        assert events[name]['layout']['client']==events[name]['layout']['scroll']
    assert events['lab_mobile_viewbox_corrected']['viewBox']=='140 0 420 530'
    assert '視窗之外' in events['lab_outside_view_marked']['text']
    assert read(folder/'console-final.json')==[]
    for c in plays:
        expected=next(s['frames'] for s in manifest['scenes'] if s['id']==c['scene']);assert c['tail']==expected and c['last']==expected[-1] and c['stopped']=='播放'
    perf=plays[-1];assert perf['performance']['p95']<200 and perf['render']['median']>=30
    assert plays[-1]['memory']['geometries']==next(c for c in checks if c['test']=='all_240_frames')['last']['renderer_memory']['geometries']
    traces={t['scenario']:t for t in read(bundle/manifest['synthetic_diagnostics']['file'])};lab=0
    for scenario in traces:
        observed=read(folder/f'lab-{scenario}.json');assert len(observed)==51
        for i,r in enumerate(observed):
            source=traces[scenario]['frames'][i];assert r['source_mode']=='synthetic_geometry_not_AI_or_CARLA' and r['scenario']==scenario and r['frame']==i
            for name in ['time_s','measured_xy','forecast','motion','risk']:assert r[name]==source[name]
            x,y=r['measured_xy'];np.testing.assert_allclose(r['projected_measured_xy'],[350+y*12,438-x*12],rtol=0,atol=1e-10);lab+=1
    write_json(ROOT/'artifacts/reports/phase7-browser-verified-final.json',{'status':'passed_real_browser_sources_geometry_playback_and_synthetic_isolation',
        'real_frames':len(records),**counts,'synthetic_frames':lab,'complete_11_8s_playbacks':len(plays),
        'WebGL_loss_and_reload_recovery':True,'eight_RGB_camera_switches':8,'mobile_horizontal_overflow':False,'final_warning_error_console':[],
        'CPU_apply_ms':perf['performance'],'render_FPS':perf['render'],'renderer_memory':perf['memory'],'second_cycle_geometry_drift':0,
        'bundle_sha256':sha(bundle/'bundle.json'),'browser_receipt_sha256':sha(folder/'frames.json'),'checks_sha256':sha(folder/'checks.json'),
        'formal_model_quality_passed':False,'limits':'Observed 240 WebGL frames and 204 clearly synthetic SVG frames. CPU apply excludes GPU present; recorded replay, no live CARLA producer.'})
    print({'status':'passed','real_frames':len(records),**counts,'synthetic_frames':lab,'CPU_p95_ms':perf['performance']['p95'],'render_median_FPS':perf['render']['median']})

if __name__=='__main__':main()
