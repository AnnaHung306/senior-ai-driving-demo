# segmentation — sunny-segmentation-v6-001

實際訓練完成；heldout 驗收：`not_evaluated`。權重保留本機 ignored 路徑 `C:/code/senior_AI/artifacts/runs/sunny-segmentation-v6-001/best.pt`，未宣稱已上傳或納入 Git。

SHA256 `0a3fe3f66fb93be373f2e82218632a4f8cc277bf6067f48bb4683a71f1b2af30`。

MobileNetV3 encoder（官方 COCO detector 完整 body transfer、FrozenBN）＋自行訓練多尺度 U-Net decoder，17類；CUDA BF16 autocast、FP32參數／loss／結果；paired scale/crop/ignore-padding/flip只用於train。 weighted CE＋foreground Dice。road 是語意道路，可見道路界面只取 road/line 與 sidewalk/terrain 相鄰處，不推斷遮擋後路界或可安全通行區域。

本次從專案已訓練權重 `artifacts/runs/sunny-segmentation-v5-001/best.pt`（SHA `3fe7bd08d885e481b7435fdf466945f142f8323447cca46bf31c32455835a53f`）再微調，重設optimizer/scheduler；不是從零初始化。

另用train-only靜態分割補景：1024個獨特視圖、128個相關道路錨點，每輪重複採樣2次；不是新增動態路線。補景audit SHA `06703e9bf4d88f3788f1cab169cd2eb04c9b2b4ee165300bdcbca7a0919ca68f`。

保留訓練地圖的靜態分割擴充：1024視圖、128相關道路錨點，每輪重複2次；只用於分割訓練，非新動態路線。audit SHA `326cd9fda7d36d598b4cad880b483a3bb962fbed2bf192e003687eb23afcc18c`，review SHA `30ab42a79ae6ff032aaceadb9091130c1f3b93c7f973a688e1646f644d01956f`。

## 驗證／選模

16 類 focus mIoU 60.65%；可見道路界面 F1（2px）62.22%。

| 類別 | GT pixels | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| other_annotated | 413857841 | 89.99% | 97.07% | 92.50% |
| road | 198638171 | 95.23% | 97.66% | 97.45% |
| sidewalk | 31957757 | 59.79% | 63.17% | 91.80% |
| pedestrian | 399691 | 30.42% | 59.99% | 38.16% |
| rider | 604233 | 55.37% | 77.91% | 65.68% |
| car | 2651790 | 68.57% | 80.90% | 81.82% |
| truck | 1342377 | 69.31% | 87.52% | 76.91% |
| bus | 17321696 | 95.65% | 97.76% | 97.79% |
| motorcycle | 446570 | 56.13% | 71.19% | 72.62% |
| bicycle | 201966 | 39.72% | 68.54% | 48.57% |
| road_line | 5694147 | 76.45% | 83.62% | 89.91% |
| traffic_light | 212553 | 57.46% | 80.36% | 66.84% |
| wall | 2048168 | 15.65% | 18.74% | 48.68% |
| fence | 8168548 | 60.58% | 73.08% | 77.97% |
| pole | 8926242 | 44.34% | 49.25% | 81.66% |
| temporary_obstacle | 432081 | 67.05% | 93.38% | 70.39% |
| terrain_or_ground | 136292335 | 78.73% | 89.49% | 86.75% |

## 使用限制

八方向逐鏡頭二維感知；沒有跨鏡頭融合、深度／BEV／3D occupancy、導航規劃或 learned vehicle control。只有晴朗白天合成資料；共用資產、短序列與少量城鎮／路線高度相關，像素／frames 不是獨立試驗，沒有可靠廣泛泛化信賴區間。分類與偵測分數均未校準。規則資料生成器的零碰撞不代表 learned driving。
