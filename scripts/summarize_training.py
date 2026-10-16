"""Build a model card and compact, versionable evidence from a completed run."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json


def summarize(run, destination):
    read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
    report, test, val = read('report.json'), read('test.json'), read('validation.json')
    config, environment, timing = read('config.json'), read('environment.json'), read('latency.json')
    if report['status'] != 'completed':
        raise ValueError('Only completed experiments can enter the model registry')
    destination.mkdir(parents=True, exist_ok=False)
    for name in ['report.json', 'test.json', 'validation.json', 'config.json', 'environment.json', 'provenance.json',
                 'cuda_overfit.json', 'latency.json', 'train_validation_statistics.json', 'test_statistics.json',
                 'untrained_validation.json', 'history.json', 'test_started.json']:
        shutil.copyfile(run / name, destination / name)
    header = '| 類別 | Test IoU | Precision | Recall | GT pixels |\n|---|---:|---:|---:|---:|\n'
    percentage = lambda value: '—' if value is None else f'{value * 100:.2f}%'
    rows = ''.join(f'| {c["name"]} | {percentage(c["iou"])} | {percentage(c["precision"])} | '
                   f'{percentage(c["recall"])} | {c["support_pixels"]} |\n' for c in test['classes'])
    scenes = '\n'.join(f'- {name}：focus mIoU {percentage(value["focus_macro_iou"])}'
                       for name, value in test['by']['scene_id'].items())
    model_card = f'''# Model card：{run.name}

探索性單視角語意分割模型；八方向共用權重，從零訓練。不是偵測器、3D／BEV 模型或自動駕駛政策，未作正式感知品質驗收。

## 來源與訓練

- 資料：{config['dataset']}；train／validation／test=3／1／2 個場景群組，1200／400／800 views；共同 Town01、十秒短片，不能視為 2400 個獨立樣本。
- Tiny U-Net、{report['parameter_count']:,} parameters、640×360、AdamW、batch {config['batch_size']}、{config['epochs']} epochs、seed {config['seed']}、AMP；無預訓練權重。只 train 做亮度／對比增強與 loss weights。
- {environment['gpu']}；PyTorch {environment['torch']} / CUDA {environment['cuda_runtime']} / Python {environment['python']}。
- 七類 focus + other_annotated；raw unknown 0 忽略 255。other 包含其餘已標註物件，不能解釋成可行駛空間。小型物件保留原始解析度，未縮圖。
- Best epoch {report['best_epoch']}，只依 validation supported-focus macro IoU 選模；untrained validation control={percentage(read('untrained_validation.json')['focus_macro_iou'])}。
- Data SHA256：`{report['provenance']['data_sha256']}`；source SHA256：`{report['provenance']['source_sha256']}`。完整環境、source snapshot、RNG / optimizer checkpoint 在原 run。
- Checkpoint：`{run.relative_to(ROOT).as_posix()}/best.pt`；SHA256：`{report['checkpoint_sha256']}`。checkpoint 是本機自產 tensors / primitives，載入使用 weights_only=True。

## 最終評估

Validation focus mIoU **{percentage(val['focus_macro_iou'])}**；test focus mIoU **{percentage(test['focus_macro_iou'])}**（{test['focus_supported_classes']} 類），pixel accuracy {percentage(test['pixel_accuracy'])}。多數類常數基準 focus mIoU={percentage(test['baselines']['constant_train_majority']['focus_macro_iou'])}；train-prior 解析期望基準={percentage(test['baselines']['train_prior_expected_counts']['focus_macro_iou'])}。解析期望基準為 expected counts 的比率，不是重複隨機抽樣平均。

{header}{rows}
{scenes}

完整 scene／camera／weather 分解與 confusion matrix 在 test.json。零預測的 precision 為 null；有 GT 但完全漏判時 IoU / recall 為 0。沒有支持的類別不冒充完美，也不混入 supported-focus 平均。

## 工程與速度

- CUDA / overfit sanity、checkpoint 重載 forward 一致性通過。train-only overfit loss ratio={read('cuda_overfit.json')['loss_ratio']:.4f}。GPU peak allocated={report['peak_allocated_vram_bytes'] / 2**20:.1f} MiB；reserved={report['peak_reserved_vram_bytes'] / 2**20:.1f} MiB。
- Batch 1、warmup {timing['warmup']}、{timing['samples']} 次同步 CUDA：network p50 / p95={timing['network_ms']['p50']:.2f} / {timing['network_ms']['p95']:.2f}ms；warm-cache 載圖到 mask+confidence p50 / p95={timing['warm_file_cache_load_to_mask_and_confidence_ms']['p50']:.2f} / {timing['warm_file_cache_load_to_mask_and_confidence_ms']['p95']:.2f}ms。
- 八方向順序處理單 bundle={timing['one_eight_view_sequential_mask_only_ms']:.2f}ms；不是分位數，也不含感測器、模擬器渲染或寫檔，不代表八鏡頭閉環即時達標。
- RGB-only 生產推論 CLI：scripts/predict_segmentation.py；輸出 model IDs 0..7、frame / timestamp / camera / checkpoint hash、未校準 softmax。模型 forward 不接觸 labels、depth 或 simulator。

## 誤差與適用限制

最差 scene frame 與各 focus 類最差正樣本按固定規則保存於原 run 的 validation_errors / test_errors，RGB／GT／prediction 並排；GT 只供評估。錯誤清單保留所有 frame 的逐類 IoU / support，不能只展示好結果。

單城、固定車型、腳本演出與極少群組限制很大；validation 一組與 test 兩組不足以聲稱可靠泛化信賴區間。像素準確率不能替代兩輪車／行人召回。大型車、號誌狀態、臨時障礙物、可見性正確的 instance detection、遮擋分層、OOD 與 confidence calibration 尚未完成。語意道路不是導航可行駛區域。不得將本模型直接接上車輛控制。

本次 test 結果已揭露，後續若據此改模型就屬開發資料；正式驗收必須補新的場景 lockbox。後續資料與偵測標註協議詳見 PHASE3.md。
'''
    (destination / 'MODEL_CARD.md').write_text(model_card, encoding='utf-8')
    (run / 'MODEL_CARD.md').write_text(model_card, encoding='utf-8')
    # An HTML index opens the real model outputs, not a rendered or scripted AI mockup.
    panels = []
    for kind, result in [('validation_errors', val), ('test_errors', test)]:
        for error in result['error_gallery']:
            panels.append(f'<figure><img src="{kind}/{error["image"]}" alt="RGB, GT, learned prediction">'
                f'<figcaption>{html.escape(kind)} / {html.escape(error["scene_id"])} / {html.escape(error["camera"])} '
                f'/ frame {error["frame"]} / focus mIoU {percentage(error["focus_macro_iou"])}</figcaption></figure>')
    page = ('<!doctype html><meta charset="utf-8"><title>Segmentation experiment evidence</title>'
        '<style>body{font:16px system-ui;background:#111;color:#eee;margin:24px}img{width:100%}'
        'figure{margin:24px 0}pre{white-space:pre-wrap}</style><h1>Actual RGB / GT / learned prediction</h1>'
        '<p>Exploratory; GT is evaluation only. Panels selected by worst per-scene / per-class IoU.</p>'
        + '<pre>' + html.escape(model_card) + '</pre>' + ''.join(panels))
    (run / 'review.html').write_text(page, encoding='utf-8')
    registry = ROOT / 'evidence/phase3/registry.json'
    entries = json.loads(registry.read_text()) if registry.exists() else []
    if any(e['run'] == run.name for e in entries):
        raise ValueError('Run already registered')
    entries.append({'run': run.name, 'model_card': (destination / 'MODEL_CARD.md').relative_to(ROOT).as_posix(), **report})
    write_json(registry, entries)
    return destination / 'MODEL_CARD.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(summarize(args.run.resolve(), args.destination.resolve()))
