# detection — sunny-detection-v6-001

實際訓練完成；heldout 驗收：`not_evaluated`。權重保留本機 ignored 路徑 `C:/code/senior_AI/artifacts/runs/sunny-detection-v6-001/best.pt`，未宣稱已上傳或納入 Git。

SHA256 `ba6d2f4269a7f3f95e4b92d7daff1f330f4203a4619e1f5902c752aa371f7155`。

MobileNetV3 Faster R-CNN，P2/P3/P4/P5＋pool，官方 COCO backbone／部分 FPN、RPN transfer，再自行微調八類可見物件。兩輪車含同 ID rider/body；臨時障礙物目前只涵蓋核准交通桶資產。使用明確 ignore 的 visible-box 101 recall AP，非完整 COCO evaluator。

本次從專案已訓練權重 `artifacts/runs/sunny-detection-v5-001/best.pt`（SHA `ec1427214d5fc3e51b9bbb2fed652fa762e97edeb34cfc23a989bdc879c8575f`）再微調，重設optimizer/scheduler；不是從零初始化。

原始RGB/框為640×360；模型內部resize最短邊720、最長邊1280，輸出框回復原始座標。

只對含GT自行車的train視圖，以2/3機率縮小到半尺度；所有框同步變換、填補區域ignore。驗證與正式推論不作幾何變換。

## 驗證／選模

mAP50 70.46%；mAP50:95 45.10%。

| 類別 | GT 物件 | AP50 | AP50:95 | Precision@.5 | Recall@.5 |
|---|---:|---:|---:|---:|---:|
| car | 946 | 75.51% | 54.07% | 90.56% | 67.97% |
| pedestrian | 539 | 93.42% | 63.46% | 89.13% | 92.76% |
| bicycle | 729 | 49.27% | 34.27% | 95.87% | 41.43% |
| motorcycle | 845 | 59.15% | 43.53% | 89.91% | 49.59% |
| truck | 1042 | 66.36% | 47.34% | 73.11% | 66.79% |
| bus | 1018 | 84.31% | 50.97% | 96.04% | 83.40% |
| traffic_light | 2368 | 49.68% | 26.87% | 96.58% | 46.45% |
| temporary_obstacle | 419 | 86.00% | 40.34% | 77.70% | 84.01% |

## 使用限制

八方向逐鏡頭二維感知；沒有跨鏡頭融合、深度／BEV／3D occupancy、導航規劃或 learned vehicle control。只有晴朗白天合成資料；共用資產、短序列與少量城鎮／路線高度相關，像素／frames 不是獨立試驗，沒有可靠廣泛泛化信賴區間。分類與偵測分數均未校準。規則資料生成器的零碰撞不代表 learned driving。
