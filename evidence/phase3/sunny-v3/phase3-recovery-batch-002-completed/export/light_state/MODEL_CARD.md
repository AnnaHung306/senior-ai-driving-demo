# light_state — sunny-light-state-v1-004

實際訓練完成；heldout 驗收：`not_evaluated`。權重保留本機 ignored 路徑 `C:/code/senior_AI/artifacts/runs/sunny-light-state-v1-004/best.pt`，未宣稱已上傳或納入 Git。

SHA256 `f9836ef6474d74b607124cbb4887492105b3949f23d3c9065741d86c160b451e`。

LightStateNet 四類 RGB crop 分類；native phase 只作 GT，必須有可見相容燈色，否則 unknown。正式輸入沒有模擬器狀態。主動態 lockbox 黃燈不足，依訓練前凍結計畫額外補收固定視角相位影像；動態／控制場景分別報告，合併結果只供號誌驗收。

## 驗證／選模

Macro recall 94.38%；accuracy 93.62%。

| 類別 | GT crops | Precision | Recall |
|---|---:|---:|---:|
| unknown_or_unobservable | 1429 | 93.46% | 96.99% |
| red | 753 | 95.90% | 86.99% |
| yellow | 31 | 88.57% | 100.00% |
| green | 155 | 86.83% | 93.55% |

含偵測漏判的正確狀態 macro recall：39.91%。

| 狀態 | GT 物件 | 含偽號誌的狀態 Precision | 含漏判正確狀態 Recall |
|---|---:|---:|---:|
| unknown_or_unobservable | 1429 | 90.11% | 44.02% |
| red | 753 | 95.64% | 40.77% |
| yellow | 31 | 25.00% | 25.81% |
| green | 155 | 86.36% | 49.03% |

## 使用限制

八方向逐鏡頭二維感知；沒有跨鏡頭融合、深度／BEV／3D occupancy、導航規劃或 learned vehicle control。只有晴朗白天合成資料；共用資產、短序列與少量城鎮／路線高度相關，像素／frames 不是獨立試驗，沒有可靠廣泛泛化信賴區間。分類與偵測分數均未校準。規則資料生成器的零碰撞不代表 learned driving。
