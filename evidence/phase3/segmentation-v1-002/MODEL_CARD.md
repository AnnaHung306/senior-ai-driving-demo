# Model card：segmentation-v1-002

探索性單視角語意分割模型；八方向共用權重，從零訓練。不是偵測器、3D／BEV 模型或自動駕駛政策，未作正式感知品質驗收。

## 來源與訓練

- 資料：data/town01-motion-v2-001；train／validation／test=3／1／2 個場景群組，1200／400／800 views；共同 Town01、十秒短片，不能視為 2400 個獨立樣本。
- Tiny U-Net、488,120 parameters、640×360、AdamW、batch 4、16 epochs、seed 20261016、AMP；無預訓練權重。只 train 做亮度／對比增強與 loss weights。
- NVIDIA GeForce RTX 4070 Laptop GPU；PyTorch 2.10.0+cu128 / CUDA 12.8 / Python 3.12.14。
- 七類 focus + other_annotated；raw unknown 0 忽略 255。other 包含其餘已標註物件，不能解釋成可行駛空間。小型物件保留原始解析度，未縮圖。
- Best epoch 12，只依 validation supported-focus macro IoU 選模；untrained validation control=0.34%。
- Data SHA256：`6d8e99f1f68a02173f702be15391b42ed06aebd6b1ff2a9ba3ab82f2abdc4f55`；source SHA256：`3f5b26463e277e2dbc87150fa8d06996dbba0423c2f8e0cb21b08bdc309cbcc0`。完整環境、source snapshot、RNG / optimizer checkpoint 在原 run。
- Checkpoint：`artifacts/runs/segmentation-v1-002/best.pt`；SHA256：`f06e4b110ea767d2c4b910d2cbfa8f9ea3ef96fa807c950bd2afd5f54947a1d4`。checkpoint 是本機自產 tensors / primitives，載入使用 weights_only=True。

## 最終評估

Validation focus mIoU **27.50%**；test focus mIoU **26.22%**（7 類），pixel accuracy 68.00%。多數類常數基準 focus mIoU=0.00%；train-prior 解析期望基準=2.69%。解析期望基準為 expected counts 的比率，不是重複隨機抽樣平均。

| 類別 | Test IoU | Precision | Recall | GT pixels |
|---|---:|---:|---:|---:|
| other_annotated | 65.92% | 79.34% | 79.59% | 120988447 |
| road | 36.82% | 60.58% | 48.41% | 32843069 |
| sidewalk | 26.23% | 43.68% | 39.64% | 26294740 |
| pedestrian | 13.70% | 14.67% | 67.41% | 310474 |
| car | 17.01% | 18.87% | 63.24% | 3089958 |
| motorcycle | 49.01% | 55.98% | 79.75% | 559584 |
| bicycle | 6.13% | 8.10% | 20.21% | 25412 |
| road_line | 34.63% | 52.10% | 50.82% | 208275 |

- scene-04：focus mIoU 49.04%
- scene-05：focus mIoU 11.83%

完整 scene／camera／weather 分解與 confusion matrix 在 test.json。零預測的 precision 為 null；有 GT 但完全漏判時 IoU / recall 為 0。沒有支持的類別不冒充完美，也不混入 supported-focus 平均。

## 工程與速度

- CUDA / overfit sanity、checkpoint 重載 forward 一致性、訓練後全影像／metadata／source 再驗證通過。train-only overfit loss ratio=0.3716。GPU peak allocated=1382.4 MiB；reserved=1538.0 MiB。
- Batch 1、warmup 10、50 次同步 CUDA：network p50 / p95=9.07 / 9.85ms；warm-cache 載圖到 mask+confidence p50 / p95=25.64 / 32.04ms。
- 八方向順序處理單 bundle=217.09ms；不是分位數，也不含感測器、模擬器渲染或寫檔，不代表八鏡頭閉環即時達標。
- RGB-only 生產推論 CLI：scripts/predict_segmentation.py；輸出 model IDs 0..7、frame / timestamp / camera / checkpoint hash、未校準 softmax。模型 forward 不接觸 labels、depth 或 simulator。

## 誤差與適用限制

最差 scene frame 與各 focus 類最差正樣本按固定規則保存於原 run 的 validation_errors / test_errors，RGB／GT／prediction 並排；GT 只供評估。錯誤清單保留所有 frame 的逐類 IoU / support，不能只展示好結果。

error_catalog.json 另保存每 split / focus 類的前三大 false-positive / false-negative 像素案例，及 class GT support 為 0、1..100、101..1000、>1000 pixels 的影像分層。這是每張圖的語意支持像素分層，不能冒充 instance 大小、遮擋標註或 small-object AP。

單城、固定車型、腳本演出與極少群組限制很大；validation 一組與 test 兩組不足以聲稱可靠泛化信賴區間。像素準確率不能替代兩輪車／行人召回。大型車、號誌狀態、臨時障礙物、可見性正確的 instance detection、遮擋分層、OOD 與 confidence calibration 尚未完成。語意道路不是導航可行駛區域。不得將本模型直接接上車輛控制。

本次 test 結果已揭露，後續若據此改模型就屬開發資料；正式驗收必須補新的場景 lockbox。後續資料與偵測標註協議詳見 PHASE3.md。
