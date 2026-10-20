"""Seal Phase7 receipts and sources without RGB, weights or large upstream arrays."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/phase7/causal-motion-v1-001'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'})


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    # Authoritative receipts must already exist; never manufacture a successful run.
    reports = ROOT / 'artifacts/reports'
    integrity = json.loads((reports / 'phase7-integrity-final.json').read_text(encoding='utf8'))
    browser = json.loads((reports / 'phase7-browser-verified-final.json').read_text(encoding='utf8'))
    assert integrity['frames'] == browser['real_frames'] == 240
    assert integrity['forecasts'] == browser['forecasts'] == 889
    assert browser['WebGL_loss_and_reload_recovery'] is True
    assert (ROOT / 'artifacts/runs/phase7-visual-QA-v1-001/assistant_review.json').is_file()
    if OUT.exists():
        raise FileExistsError('Evidence is immutable; use a new version instead')
    OUT.mkdir(parents=True)

    def copy(source, relative):
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def archive(name, paths):
        paths = sorted(set(paths))
        manifest = []
        with zipfile.ZipFile(OUT / name, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in paths:
                rel = p.relative_to(ROOT).as_posix()
                z.write(p, rel)
                manifest.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': sha(p)})
        with zipfile.ZipFile(OUT / name) as z:
            assert z.testzip() is None
            for r in manifest:
                assert hashlib.sha256(z.read(r['path'])).hexdigest() == r['sha256']
        write(OUT / (name + '.inventory.json'), manifest)

    for p in sorted(reports.glob('phase7*')):
        if p.is_file():
            copy(p, 'reports/' + p.name)
    for p in files(ROOT / 'artifacts/runs/phase7-visual-QA-v1-001'):
        copy(p, 'visual-QA/' + p.name)
    for p in (reports / 'phase7-browser-001').glob('*.jpg'):
        copy(p, 'screenshots/' + p.name)
    copy(ROOT / 'PHASE7.md', 'REPORT.md')
    browser_files = files(reports / 'phase7-browser-001')
    archive('browser-receipts.zip', [p for p in browser_files if p.suffix != '.jpg'])
    run_files = []
    for name in ['phase7-inputs-v1-001', 'phase7-bev-v1-001', 'phase7-world-v1-001']:
        folder = ROOT / 'artifacts/runs' / name
        run_files += list(folder.glob('*.json'))
        if (folder / 'source').is_dir():
            run_files += files(folder / 'source')
        # Input staging is RGB-only but huge. Only its inventory is archived.
        if name != 'phase7-inputs-v1-001':
            run_files += [p for p in files(folder) if p.suffix == '.json']
    for n in ['001', '002', '003', '004']:
        run_files += files(ROOT / f'artifacts/runs/phase7-motion-v1-{n}')
    for n in ['001', '002', '004']:
        run_files += files(ROOT / f'artifacts/runs/phase7-evaluation-v1-{n}')
    run_files += files(ROOT / 'artifacts/runs/phase7-synthetic-v1-001')
    assert all(p.suffix not in {'.png', '.jpg', '.npz', '.pt', '.pth'} for p in run_files)
    archive('runs-and-failed-experiments.zip', run_files)
    bundle = ROOT / 'artifacts/runs/phase7-viewer-bundle-v1-001'
    archive('viewer-display-without-RGB.zip', [p for p in files(bundle) if p.suffix in {'.json', '.bin'}])
    sources = [ROOT / n for n in ['README.md', 'ROADMAP.md', 'CURRENT_TASK.md', 'PHASE7.md',
                                'Start-WorldViewer.ps1', '.gitignore', '.gitattributes']]
    for name in ['src', 'scripts', 'configs', 'tests', 'web/world-viewer']:
        sources += files(ROOT / name)
    archive('source-snapshot.zip', sources)
    readme = '''# Phase7：因果運動與三維動態展示（2026-10-20）

工程研究版完成；正式模型品質仍未通過。完整方法、數字、失敗與重現命令見 [REPORT.md](REPORT.md)。

- 主畫面為240組真RGB凍結感知推論，1091個物件／889次運動預測；GT幾何不進主場景。
- validation速度向量MAE .290m/s、ADE1 .950m、FDE2 1.205m；2秒GT覆蓋61.22%、all-GT miss47.52%。ID2同原版，無改善宣稱。
- 新關聯器v1(66 IDs)／v1.1(58)都差於原50，未採用。003離群計數問題修成004，原結果全部保留。
- 256 Python、40 JS、22 HTTP；真browser全部240組、889個實際WebGL軌跡buffer、4段完整播放；204個合成診斷影格另核對。CPU套用p9544.4ms／render60FPS，非即時感知。
- 合成跟車／轉彎／切入／急煞明標非RGB AI，不能代替新CARLA情境驗收。風險是保守circle接近、非校準碰撞機率；nominal recall50%，potential誤警很多。

![最終三維動態介面](screenshots/final-desktop.jpg)

![GT預測錯例與代表例](visual-QA/contact-0.png)

`reports/`保存完整驗證、失敗及回歸log；`visual-QA/`保存8面板（7獨立例）、原catalog與另存實看收據；`screenshots/`含手機與故障恢復。

`runs-and-failed-experiments.zip`保存4版motion、3版evaluation、固定輸入inventory／上游JSON與source／合成trace；`viewer-display-without-RGB.zip`保存bundle與顯示JSON/bin；`browser-receipts.zip`保存240真browser與204合成DTO、最終checks及空console；`source-snapshot.zip`保存本輪最終程式與文件（含既有vendor授權）。每個ZIP有原bytes SHA inventory且已CRC／逐項SHA核對。

原1920RGB、權重及上游大型NPZ僅留本機ignored目錄，未放入Git；因此此封存可審閱、核對數字及來源，不是完整可攜推論資料集。未訓練／安裝新套件、未開dev/lockbox、未發車／控制。兩個測試服務與計算jobs交付前均停止。

封存manifest只證明檔案一致，不把研究成績改成正式pass。舊Phase3/4/5/6證據未修改。
'''
    (OUT / 'README.md').write_text(readme, encoding='utf8')
    rows = [{'path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files(OUT)]
    write(OUT / 'SHA256_MANIFEST.json', {'status': 'sealed_file_integrity_not_formal_quality',
          'created_utc': datetime.now(timezone.utc).isoformat(), 'files': rows,
          'manifest_self_excluded': True, 'formal_model_quality_passed': False})
    print({'status': 'sealed', 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows)})


if __name__ == '__main__':
    main()
