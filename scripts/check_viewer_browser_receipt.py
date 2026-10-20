"""Verify captured public browser diagnostics against the actual export."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha


def main(a):
    receipt=read(a.receipt);manifest=read(a.bundle/'bundle.json');rows={r['key']:r for r in manifest['frames']}
    actual=receipt['frames'];assert len(actual)==len(rows)==32
    assert {r['frame_key'] for r in actual}==set(rows)
    for r in actual:
        source=read(a.bundle/rows[r['frame_key']]['file']);assert r['error'] is None and r['context_valid'] is True
        assert r['frame_source_sha256']==source['world_source_sha256'] and r['checkpoint_sha256']==manifest['checkpoint_sha256']
        assert r['run_contract_sha256']==manifest['run_contract_sha256'] and r['object_count']==len(source['objects'])
        for rendered,o in zip(r['rendered_object_transforms'],source['objects'],strict=True):
            assert rendered['track_id']==o['track_id'];x,y,z=o['center_ego_m'];L,W,H=o['size_lwh_m']
            np.testing.assert_allclose(rendered['position_three'],[y,z,-x],rtol=0,atol=1e-12)
            np.testing.assert_allclose(rendered['dimensions_three'],[W,H,L],rtol=0,atol=1e-12)
            assert rendered['yaw_y_three']==-o['yaw_rad']
    checks=receipt['checks'];find=lambda name:[r for r in checks if r['test']==name]
    assert len(find('RGB_camera'))==8 and all(r['loaded'] for r in find('RGB_camera'))
    free=find('free_drag_zoom_reset')[0];assert free['freeBefore']!=free['freeAfter'] and free['freeAfter']!=free['zoomAfter'] and free['freeBefore']==free['resetAfter']
    assert find('layers_off_on')[0]['rgb_hidden'] is True
    assert find('layers_off_on')[0]['diagnostic']['layers']=={'boxes':False,'IDs':False,'voxels':True,'free':True}
    for r in find('timestamp_playback_half_speed'):
        assert r['stopped_label']=='播放' and r['final']==next(s['frames'][-1] for s in manifest['scenes'] if s['id']==r['scene'])
        # Each source frame was actually applied; do not accept a jump to final only.
        tail=r['applied'][-8:];assert tail==next(s['frames'] for s in manifest['scenes'] if s['id']==r['scene'])
    assert {r['speed'] for r in find('playback_speed')}=={'1','2'}
    assert find('actual_WEBGL_lose_context')[0]['diagnostic']['context_valid'] is False
    assert find('reload_after_context_loss')[0]['diagnostic']['context_valid'] is True
    assert find('feed_GT_field_rejected')[0]['diagnostic']['frame_key']=='town04-00/246282'
    assert 'schema' in find('feed_GT_field_rejected')[0]['status']
    recovered=find('feed_recovers_after_invalid')[0];assert recovered['diagnostic']['error'] is None and recovered['overlay_hidden'] is True
    assert '逾時' in find('stale_feed_retains_last')[0]['status']
    assert find('canvas_raycast_select')[0]['selected']=='town04-01:T000024'
    assert find('follow_view')[0]['diagnostic']['view']=='follow' and find('top_view')[0]['diagnostic']['camera']['type']=='OrthographicCamera'
    mobile=find('responsive_390')[0]['layout'];assert mobile['scroll']==mobile['viewport'] and mobile['canvas']['width']<=mobile['viewport']
    perf=find('feed_applied_next')[0]['diagnostic'];assert perf['stream_applied']>=2
    assert perf['frame_apply_ms']['p95']<200 and perf['render_fps']['median']>=30
    assert perf['renderer_memory']['max_geometries']<=90
    result={'status':'passed_actual_browser_UI_and_geometry','frames':32,'objects':sum(r['object_count'] for r in actual),'checks':len(checks),
            'receipt_sha256':sha(a.receipt),'bundle_sha256':sha(a.bundle/'bundle.json'),
            'performance':{key:perf[key] for key in ['frame_apply_ms','verified_fetch_ms','render_fps','renderer_memory','total_applied_frames','stream_applied','feed_age_ms']},
            'console_logs':receipt['console_logs'],'formal_model_quality_passed':False,
            'limits':'Actual local recorded prediction replay/monitored pointer only. CPU apply excludes GPU present. Short sequences, no CARLA live inference or destination navigation.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,default=ROOT/'artifacts/reports/phase6-browser-001/receipt.json')
    p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/runs/phase6-viewer-bundle-002');p.add_argument('--output',type=Path,required=True);main(p.parse_args())
