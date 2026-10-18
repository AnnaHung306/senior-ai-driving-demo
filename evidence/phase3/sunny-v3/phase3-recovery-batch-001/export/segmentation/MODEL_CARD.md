# segmentation — sunny-segmentation-v5-001

實際訓練完成；heldout 驗收：`not_evaluated`。權重保留本機 ignored 路徑 `C:/code/senior_AI/artifacts/runs/sunny-segmentation-v5-001/best.pt`，未宣稱已上傳或納入 Git。

SHA256 `3fe7bd08d885e481b7435fdf466945f142f8323447cca46bf31c32455835a53f`。

MobileNetV3 encoder（官方 COCO detector 完整 body transfer、FrozenBN）＋自行訓練多尺度 U-Net decoder，17類；CUDA BF16 autocast、FP32參數／loss／結果；paired scale/crop/ignore-padding/flip只用於train。 weighted CE＋foreground Dice。road 是語意道路，可見道路界面只取 road/line 與 sidewalk/terrain 相鄰處，不推斷遮擋後路界或可安全通行區域。

本次從專案已訓練權重 `artifacts/runs/sunny-segmentation-v4-001/best.pt`（SHA `a7031897c0ce35cc97f4ab035277c51e0a35558dccc225b69455a16cc6f9eb67`）再微調，重設optimizer/scheduler；不是從零初始化。

另用train-only靜態分割補景：1024個獨特視圖、128個相關道路錨點，每輪重複採樣4次；不是新增動態路線。補景audit SHA `06703e9bf4d88f3788f1cab169cd2eb04c9b2b4ee165300bdcbca7a0919ca68f`。

## 驗證／選模

16 類 focus mIoU 59.00%；可見道路界面 F1（2px）49.45%。

| 類別 | GT pixels | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| other_annotated | 413857841 | 88.53% | 95.26% | 92.61% |
| road | 198638171 | 89.96% | 96.77% | 92.75% |
| sidewalk | 31957757 | 48.78% | 50.12% | 94.81% |
| pedestrian | 399691 | 29.79% | 56.71% | 38.56% |
| rider | 604233 | 56.06% | 77.86% | 66.69% |
| car | 2651790 | 70.37% | 87.33% | 78.37% |
| truck | 1342377 | 70.77% | 84.76% | 81.09% |
| bus | 17321696 | 95.59% | 97.27% | 98.23% |
| motorcycle | 446570 | 57.39% | 68.82% | 77.56% |
| bicycle | 201966 | 38.07% | 67.21% | 46.75% |
| road_line | 5694147 | 76.24% | 87.21% | 85.84% |
| traffic_light | 212553 | 50.20% | 79.37% | 57.74% |
| wall | 2048168 | 9.38% | 11.45% | 34.19% |
| fence | 8168548 | 61.47% | 68.53% | 85.65% |
| pole | 8926242 | 45.75% | 52.87% | 77.26% |
| temporary_obstacle | 432081 | 69.24% | 91.87% | 73.76% |
| terrain_or_ground | 136292335 | 74.98% | 92.37% | 79.93% |

## 使用限制

八方向逐鏡頭二維感知；沒有跨鏡頭融合、深度／BEV／3D occupancy、導航規劃或 learned vehicle control。只有晴朗白天合成資料；共用資產、短序列與少量城鎮／路線高度相關，像素／frames 不是獨立試驗，沒有可靠廣泛泛化信賴區間。分類與偵測分數均未校準。規則資料生成器的零碰撞不代表 learned driving。
