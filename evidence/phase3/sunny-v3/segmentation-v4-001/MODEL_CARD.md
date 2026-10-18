# MobileUNet segmentation v4-001 — 驗證未通過

- Run：`artifacts/runs/sunny-segmentation-v4-001`；訓練完成，8 epochs / 9600 updates，best第7輪。Phase 3尚未完成。
- 理想ClearNoon模擬；主資料 `sunny-cross-town-v3-007`，train Town01/03 9600views/8groups，validation Town04 3600views/3groups。development/lockbox尚未評分。
- 17類（16focus+other），unknown/未驗證Dynamic ignore255。物件遮擋後的不可見道路界面不推估。
- COCO MobileNetV3完整encoder transfer、FrozenBN、多尺度decoder/RGB stem；3,589,953參數。8×batch8，encoder LR5e-5 / decoder5e-4，cosine；train paired scale/crop/flip與RGB光度增強。
- CUDA BF16卷積，FP32參數/optimizer/loss/output；CE inverse-sqrt訓練頻率權重加.5 foreground Dice。資料、原設定與凍結來源見相鄰JSON。
- best SHA256 `a7031897c0ce35cc97f4ab035277c51e0a35558dccc225b69455a16cc6f9eb67`；last `af3f0eee89a5dd3b7738331e51f8ab7eceedc236b7bc5471b5a2108e10da087f`。exact reload通過，27個凍結source及3600張逐張混淆總和另經CPU檔案檢查通過。
- audit SHA `55c233e337632e3e8d5e1a197c20b2bdab325818d6f38b24d1da113fce4fd11f`；source SHA `c15c5cfb8f6f71b0587c3a3eaaf82ba8a97069e3b4bb99e16c2e9a20b431d7c1`；config SHA `25c02eefe5f5328f97773a024efe400ea4b786cfcc516a4c00d8b5a4fae853de`。

## 驗證

Focus mIoU **0.50845608** ≥ .5，16類有支持；但wall/fence各低於.2、visible margin F1 **0.40733339** < .5，整體品質失敗。

| 類別 | GT pixels | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| other_annotated | 413857841 | 0.840101 | 0.909751 | 0.916480 |
| road | 198638171 | 0.901831 | 0.957785 | 0.939162 |
| sidewalk | 31957757 | 0.453456 | 0.473071 | 0.916223 |
| pedestrian | 399691 | 0.295739 | 0.358154 | 0.629221 |
| rider | 604233 | 0.551648 | 0.766511 | 0.663070 |
| car | 2651790 | 0.663415 | 0.850762 | 0.750787 |
| truck | 1342377 | 0.642209 | 0.700894 | 0.884661 |
| bus | 17321696 | 0.942531 | 0.954349 | 0.987032 |
| motorcycle | 446570 | 0.555317 | 0.662032 | 0.775030 |
| bicycle | 201966 | 0.379285 | 0.678904 | 0.462197 |
| road_line | 5694147 | 0.734299 | 0.884543 | 0.812140 |
| traffic_light | 212553 | 0.331058 | 0.881955 | 0.346408 |
| wall | 2048168 | 0.061948 | 0.069637 | 0.359409 |
| fence | 8168548 | 0.007284 | 0.038894 | 0.008883 |
| pole | 8926242 | 0.349024 | 0.393138 | 0.756716 |
| temporary_obstacle | 432081 | 0.649316 | 0.862670 | 0.724170 |
| terrain_or_ground | 136292335 | 0.616938 | 0.920538 | 0.651641 |

## 已觀測限制

- 木/白色柵欄漏檢、草地判sidewalk、路緣位移與誤報；見 `../segmentation-v4-validation-review-001/` 實際檢視頁面。
- 訓練/驗證類別占比與材質不同。固定4例顯示train隧道wall與混凝土Fence擬合好，但val木柵欄幾乎漏光；這些刻意選例不是full train成績，也不能單獨證明因果。
- CPU FP32錯例重建與正式GPU BF16評分分開；驗收仍使用本run原始完整GPU報告。
- 實測峰值CUDA allocated4046.51MiB/reserved4662MiB。training_wall_s42396.5保留原值，包含牆鐘長時間跳增，不能當純GPU速度；尚無最終benchmark/8view即時承諾。
- 尚未用此run打開development/lockbox；品質不合格前不開。Softmax未校準；沒有AI駕駛閉環或3D能力驗收。
