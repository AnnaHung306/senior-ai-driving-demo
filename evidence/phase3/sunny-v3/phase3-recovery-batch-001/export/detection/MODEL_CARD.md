# detection — sunny-detection-v5-001

實際訓練完成；heldout 驗收：`not_evaluated`。權重保留本機 ignored 路徑 `C:/code/senior_AI/artifacts/runs/sunny-detection-v5-001/best.pt`，未宣稱已上傳或納入 Git。

SHA256 `ec1427214d5fc3e51b9bbb2fed652fa762e97edeb34cfc23a989bdc879c8575f`。

MobileNetV3 Faster R-CNN，P2/P3/P4/P5＋pool，官方 COCO backbone／部分 FPN、RPN transfer，再自行微調八類可見物件。兩輪車含同 ID rider/body；臨時障礙物目前只涵蓋核准交通桶資產。使用明確 ignore 的 visible-box 101 recall AP，非完整 COCO evaluator。

本次從專案已訓練權重 `artifacts/runs/sunny-detection-v3-001/best.pt`（SHA `9cbed7cfd641261396e916128535297ecdc267e0cf1d5bbc2af531bd15db5549`）再微調，重設optimizer/scheduler；不是從零初始化。

原始RGB/框為640×360；模型內部resize最短邊720、最長邊1280，輸出框回復原始座標。

## 驗證／選模

mAP50 70.85%；mAP50:95 45.32%。

| 類別 | GT 物件 | AP50 | AP50:95 | Precision@.5 | Recall@.5 |
|---|---:|---:|---:|---:|---:|
| car | 946 | 81.80% | 57.30% | 91.47% | 71.46% |
| pedestrian | 539 | 93.20% | 64.42% | 90.16% | 91.84% |
| bicycle | 729 | 40.10% | 29.11% | 66.49% | 33.74% |
| motorcycle | 845 | 60.30% | 44.79% | 93.52% | 51.24% |
| truck | 1042 | 65.47% | 45.83% | 70.19% | 62.38% |
| bus | 1018 | 84.59% | 53.89% | 95.60% | 83.30% |
| traffic_light | 2368 | 56.10% | 28.78% | 96.62% | 50.76% |
| temporary_obstacle | 419 | 85.26% | 38.40% | 89.57% | 79.95% |

## 使用限制

八方向逐鏡頭二維感知；沒有跨鏡頭融合、深度／BEV／3D occupancy、導航規劃或 learned vehicle control。只有晴朗白天合成資料；共用資產、短序列與少量城鎮／路線高度相關，像素／frames 不是獨立試驗，沒有可靠廣泛泛化信賴區間。分類與偵測分數均未校準。規則資料生成器的零碰撞不代表 learned driving。
