"""Seal navigation engineering evidence, including failed attempts and source versions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/phase89/navigation-closed-loop-v1-001'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'})


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    reports = ROOT/'artifacts/reports'
    integrity = read(reports/'phase89-integrity-final.json')
    browser = read(reports/'phase89-browser-verified-final.json')
    validation = read(reports/'phase89-validation-final.json')
    assert integrity['frames'] == 1123 and len(integrity['runs']) == 15
    assert browser['frames'] == 411 and browser['objects'] == 333 and browser['forecasts'] == 169
    assert len(validation['named_final_route_cases']) == 5 and not validation['formal_quality_passed']
    assert all(r['status'] == 'arrived' and r['collisions'] == r['lane_events'] == 0
               for r in validation['named_final_route_cases'])
    assert len(read(reports/'phase89-offline-HTTP-final.json')['checks']) == 23
    assert read(reports/'phase89-portable-smoke-final.json')['manifest_files_checked'] == 607
    assert read(reports/'phase89-teardown-final.json')['listeners'] == []
    assert 'Ran 295 tests' in (reports/'phase89-regression-delivery.log').read_text(encoding='utf8')
    assert 'pass 48' in (reports/'phase89-JS-delivery.log').read_text(encoding='utf8')
    assert read(reports/'phase89-browser-001/console-final.json') == []
    if OUT.exists():
        raise FileExistsError('Evidence is immutable; use a new version')
    OUT.mkdir(parents=True)

    def copy(source, relative):
        target = OUT/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def archive(name, paths):
        inventory = []
        with zipfile.ZipFile(OUT/name, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in sorted(set(paths)):
                relative = p.relative_to(ROOT).as_posix()
                z.write(p, relative)
                inventory.append({'path': relative, 'bytes': p.stat().st_size, 'sha256': sha(p)})
        with zipfile.ZipFile(OUT/name) as z:
            assert z.testzip() is None
            for row in inventory:
                assert hashlib.sha256(z.read(row['path'])).hexdigest() == row['sha256']
        write(OUT/(name+'.inventory.json'), inventory)

    for p in sorted(reports.glob('phase89-*')):
        if p.is_file():
            copy(p, 'reports/'+p.name)
    for p in files(ROOT/'artifacts/runs/phase89-visual-QA-v1-001'):
        copy(p, 'visual-QA/'+p.name)
    for p in (reports/'phase89-browser-001').glob('*.jpg'):
        copy(p, 'screenshots/'+p.name)
    copy(ROOT/'PHASE89.md', 'REPORT.md')
    copy(ROOT/'DEPLOYMENT.md', 'DEPLOYMENT.md')
    copy(ROOT/'artifacts/runs/phase89-portable-replay-v1-001/SHA256_MANIFEST.json',
         'portable-replay-SHA256_MANIFEST.json')
    archive('browser-receipts.zip', [p for p in files(reports/'phase89-browser-001') if p.suffix != '.jpg'])
    run_files = []
    for folder in sorted((ROOT/'artifacts/runs').glob('phase89-*')):
        if folder.is_dir() and (folder.name.startswith('phase89-drive-') or folder.name.startswith('phase89-route-catalog-')):
            # Every attempt, including preflight/partial/cancelled/failed versions.
            run_files += [p for p in files(folder) if p.suffix in {'.json', '.py', '.txt', '.log'} or p.name == 'CANCEL']
    assert all(p.suffix not in {'.png', '.pt', '.npz'} for p in run_files)
    archive('runs-and-failed-experiments.zip', run_files)
    source = [ROOT/n for n in ['README.md', 'ROADMAP.md', 'CURRENT_TASK.md', 'PHASE89.md', 'DEPLOYMENT.md',
        'Start-NavigationDemo.ps1', 'Start-WorldViewer.ps1', '.gitattributes', '.gitignore', 'requirements-navigation-viewer.txt']]
    for folder in ['src', 'scripts', 'configs', 'tests', 'web/world-viewer']:
        source += files(ROOT/folder)
    source += [ROOT/'artifacts/tmp/finalize_phase89_metadata.py']
    archive('source-snapshot.zip', source)
    sample = ROOT/'artifacts/runs/phase89-drive-pedestrian-probe-v1-001/frames/00121465'
    record = read(sample/'record.json')
    assert sha(sample/'front.png') == record['RGB']['front']['sha256']
    copy(sample/'front.png', 'sample-RGB/pedestrian-F121465-front.png')
    write(OUT/'sample-RGB/source.json', {'run_id': record['run_id'], 'frame': record['frame'],
        'camera': 'front', 'original_path': str((sample/'front.png').relative_to(ROOT)),
        'sha256': sha(sample/'front.png'), 'not_all_RGB_archived': True})
    (OUT/'README.md').write_text('''# Phase 8–9：導航與真閉環工程研究版（2026-10-21）

已完成有限場景工程整合；正式模型品質、任意路線與泛化仍未通過。方法、門檻、數字、失敗及命令見[REPORT.md](REPORT.md)，部署與展示見[DEPLOYMENT.md](DEPLOYMENT.md)。

- 真八RGB→四個凍結CUDA模型→BEV／因果運動→全域／局部／行為規劃→自寫控制→CARLA→新感測。地圖與自車定位／速度是明示先驗；其他actor GT不輸入自車planner，沒有BasicAgent／ego autopilot。
- 五個最終具名案例（3條地理路線）共370影格：兩直路、紅→綠轉彎、障礙移除恢復、行人橫穿均抵達，碰撞0／越線0，最大路線橫向誤差.706m。保留全部18次開發嘗試，不宣稱全部嘗試成功或統計可靠性。
- 真網頁自訂179→62、11m抵達；人工停止與真WebGL中斷收尾速度0。真正front sensor.stop造成15秒timeout後速度0；有效性故障注入另列，不能混為實體感測中斷。
- 295 Python、48 JS、23真HTTP；15個有完整結果run共1123影格／8984PNG來源及各次凍結planner/control重算；實際Three730cuboids；真browser411影格／333物件／169預測buffer核對。
- 約1.2秒推論／循環，牆鐘慢8–9倍；5Hz是同步模擬時間，render約60FPS含idle。錯類、誤停、停走抖動與上游品質缺口保留，非即時穩健自駕。
- 007 Town01 train自車軌跡幾何分離不代表完整歷史／視野／資產獨立，正式heldout未評。原catalog001 selector的精確source是事後依hash重建，收據明示，沒有偽稱當時已保存。

![實際行人案3D與八鏡頭操作頁](screenshots/final-desktop.jpg)

![路線與速度的實際檢視圖](visual-QA/contact-0.png)

`reports/`保留完整驗證、模型SHA、效能、失敗／清理／回歸及可攜包收據；`visual-QA/`含7個曲線面板、2個實看contact頁與另存review，原catalog不覆寫；`screenshots/`含桌機／手機、網頁自訂抵達及WebGL故障畫面。`sample-RGB/`僅含帶hash的一張真前鏡頭。

`runs-and-failed-experiments.zip`保留兩個catalog、全部導航嘗試／director的逐幀JSON、契約、source與取消旗標，含中斷沒有report的26影格。`browser-receipts.zip`保留411影格DOM、真Float32buffer、checks、fresh最終空console及另列的歷史重啟診斷。`source-snapshot.zip`保留最終程式、文件與既有vendor授權。每個ZIP均CRC與逐項SHA核對，有原bytes inventory。

完整RGB／權重／約133.1MiB可攜ZIP保留本機ignored artifacts，沒有上傳／公開。Git只封存可攜包SHA與607檔manifest；單靠此Git證據不能播放所有原影像。獨立可攜包含64幀真行人案／512RGB，從新資料夾HTTP實測、禁止起車且不import Torch或CARLA，使用說明見部署文件。

四權重未變、沒有新訓練／套件安裝／dev或lockbox評分。交付前CARLA與導航服務均停止，相關連接埠無listener；舊Phase3–7證據與品質結果未修改。manifest證明檔案一致，並不證明模型品質。
''', encoding='utf8')
    manifest = [{'path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files(OUT)]
    write(OUT/'SHA256_MANIFEST.json', {'status': 'sealed_engineering_evidence_not_formal_quality',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'manifest_self_excluded': True,
        'formal_quality_passed': False, 'files': manifest})
    print({'status': 'sealed', 'files': len(manifest), 'bytes': sum(x['bytes'] for x in manifest)})


if __name__ == '__main__':
    main()
