"""Archive this completed local research viewer without RGB or model weights."""
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import sha,read


def main():
    target=ROOT/'evidence/phase6/world-viewer-v1-001';target.mkdir(parents=True,exist_ok=True)
    if (target/'manifest.json').exists():raise FileExistsError('Evidence already sealed')
    if {p.name for p in target.iterdir()}!={'README.md'}:raise ValueError('New README-only archive required')
    def copy(source,relative):
        dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise FileExistsError(dest)
        shutil.copyfile(ROOT/source,dest)
    copy('artifacts/runs/phase56-runtime-audit-001/report.json','runtime_audit/report.json')
    audit=ROOT/'artifacts/runs/phase56-runtime-audit-001/source'
    for p in audit.rglob('*'):
        if p.is_file():copy(p.relative_to(ROOT),'runtime_audit/source/'+p.relative_to(audit).as_posix())
    names=['phase56-regression-final.log','phase6-node-tests-final.log','phase6-http-tests-final.log',
           'phase6-bundle-integrity-001.json','phase6-Three-real-bundle-001.log','phase6-browser-verified-final.json','phase6-publisher-002.log',
           'phase5-integrity-v1-002.json','phase4-artifact-integrity-001.json']
    for name in names:copy('artifacts/reports/'+name,'checks/'+name)
    copy('artifacts/runs/phase5-world-v1-002-evaluation/report.json','checks/phase5-evaluation-report.json')
    copy('artifacts/reports/phase56-regression-001.log','history/first-system-TEMP-regression.log')
    browser=ROOT/'artifacts/reports/phase6-browser-001'
    for p in browser.iterdir():
        if p.is_file():copy(p.relative_to(ROOT),'browser/'+p.name)
    bundle=ROOT/'artifacts/runs/phase6-viewer-bundle-002';copy(bundle.relative_to(ROOT)/'bundle.json','display/bundle.json')
    for p in (bundle/'frames').rglob('*'):
        if p.is_file():copy(p.relative_to(ROOT),'display/frames/'+p.relative_to(bundle/'frames').as_posix())
    history={'status':'resolved_for_normal_host_local_workflow','events':[
        'First default sandbox Python regression could not clean system TEMP; rerun used only process-local TEMP/TMP under repo: 235 tests, no skips.',
        'Sandbox loopback could not reach test service (WinError10013/timeout). Normal-host loopback was verified; no firewall or security changes.',
        'Two early sandbox pointer heartbeats saw PermissionError replacing latest.json. Added bounded two-second atomic replace retry and unique pending file; mocked transient lock regression passed and normal-host publisher completed 80 writes, exit0. Exact cause of early mixed sandbox/normal-host permissions is not asserted.',
        'Browser found recovery with same feed key could leave an error overlay; valid reapply now clears it. Diagnostics histories capped at512; renderer resources disposed each frame.',
        'Viewer service and bounded publishers stopped at handoff. Port8766 has no listener. Desktop GPU entries dwm/BongoCat are unrelated and were left running.'
    ]}
    (target/'history/events.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
    scripts=['audit_phase5_runtime.py','export_world_viewer.py','serve_world_viewer.py','vendor_phase6_three.py','publish_viewer_frame.py',
             'check_world_viewer.py','check_world_viewer.mjs','check_viewer_http.py','check_viewer_browser_receipt.py','archive_phase6_evidence.py']
    sources=[ROOT/'scripts'/n for n in scripts]+[ROOT/'src/senior_ai'/n for n in ['world_display.py','world_viewer_service.py']]
    sources += [ROOT/'tests/test_world_viewer.py']+[p for p in (ROOT/'web/world-viewer').rglob('*') if p.is_file()]
    sources += [ROOT/n for n in ['PHASE6.md','PHASE5.md','CURRENT_TASK.md','README.md','ROADMAP.md','Start-WorldViewer.ps1']]
    inventory={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    (target/'sources.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(target/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT).as_posix())
    files={p.relative_to(target).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in target.rglob('*') if p.is_file()}
    (target/'manifest.json').write_text(json.dumps({'status':'sealed_research_engineering_evidence_not_model_quality_pass','files':files,
         'original_RGB_and_weights_in_git':False,'formal_model_quality_passed':False},indent=2)+'\n',encoding='utf-8')
    for name,row in files.items():assert sha(target/name)==row['sha256']
    print(json.dumps({'status':'sealed','files':len(files),'bytes':sum(r['bytes'] for r in files.values()),'path':str(target)}))


if __name__=='__main__':main()
