# 多鏡頭三維環境感知與自主駕駛研究系統

長期需求見 [PROJECT_SPEC.md](PROJECT_SPEC.md)，當前任務見 [CURRENT_TASK.md](CURRENT_TASK.md)。

實測已通過：CARLA 車輛加速／煞停與單鏡頭、五條固定規則導航路線、八鏡頭同步擷取，以及六場景資料 pilot；本輪另完成含移動自車、行人與兩輪車的 Phase 2 v2 基礎資料驗收。使用本地修正 CARLA 0.9.16 與 DX12／Epic；官方原樣 binary 的失敗紀錄保留。

目前以 **ClearNoon理想晴天** 為範圍。Phase 3主資料007已完成18場、86,400 PNG的獨立audit與assistant視覺抽查。17類MobileUNet v5從v4微調6輪，另用核准的1024個train-only靜態視角，完整validation focus mIoU **59.00%**、fence IoU **61.47%**；wall IoU9.38%、可見邊界F1 .4945仍未達標。新高解析偵測v5正在訓練，號誌v1-003將依唯一批次接續。前一輪detector v3的mAP50 60.21%，行人／自行車／號誌recall未達50%；phase v1-002的GT crop macro recall94.38%通過，但actual detector-box joint macro36.02%未達40%。新模型尚未對development/lockbox評分，**整個Phase 3未通過**，BEV／occupancy、3D導航與AI駕駛閉環仍未完成。

目前協議見 [PHASE3_TRAINING_PROTOCOL.md](PHASE3_TRAINING_PROTOCOL.md)，逐類成績見 [v5 MobileUNet模型卡](evidence/phase3/sunny-v3/sunny-segmentation-v5-001-completed-validation/MODEL_CARD.md)與[號誌模型卡](evidence/phase3/sunny-v3/light-state-v1-002-validation/MODEL_CARD.md)，泛化缺口依據見 [PHASE3_DOMAIN_RECOVERY.md](PHASE3_DOMAIN_RECOVERY.md)，中斷接續見 [CURRENT_TASK.md](CURRENT_TASK.md)。權重與凍結source保留於本機artifacts/runs（ignored）；可追溯數值與實際檢視圖存於evidence/phase3。分割錯例工具逐張atomic保存並能重跑，CPU重建只作診斷，不取代完整GPU驗證成績。

歷史首輪Tiny U-Net為8類，validation focus mIoU27.50%、test26.22%，其模型卡與 [PHASE3.md](PHASE3.md)保留原始範圍與失敗。早期instance／ignore工程採集與同幾何晴天／夜雨診斷另見[圖文證據](evidence/phase3/visibility-v1/REPORT.md)及[執行規格](PHASE3_VISIBILITY.md)；不與目前17類、跨城鎮的新資料成績混算。

## 環境與安裝

Windows 11、RTX 4070 Laptop 8GB、64GB RAM；server／client 固定 0.9.16。本機 .venv 為 Python 3.12.14，不修改系統 Python。

```powershell
# 新機建立環境；baseline/data 使用額外固定依賴。
.\scripts\setup.ps1 -Python312 '你的 Python 3.12 python.exe 絕對路徑'
python -m pip --python .venv\Scripts\python.exe install -r requirements-dataset.txt

# 官方 archive 約 7.3GiB；下載、ZIP CRC 驗證與安全解壓。
python scripts/download_carla.py
.\.venv\Scripts\python.exe scripts/install_carla.py

# 僅支援已核對 SHA256 的特定 build，保留 .exe.stock；先關閉 server。
.\.venv\Scripts\python.exe scripts/patch_pixel_reader.py --pool background
# 還原：.\.venv\Scripts\python.exe scripts/patch_pixel_reader.py --restore
```

Patch 改正 PixelReader 的 TaskGraph 執行緒選擇，源碼差異在 patches/pixel-reader-task-thread.patch。完整雜湊、適用範圍與失敗分析見 [DECISIONS.md](DECISIONS.md)。本地 archive SHA256 僅供追溯，沒有冒充發布者 checksum。

## 執行與驗收

使用專案專用 server、唯一 tick client；以下依序執行，每次換新輸出目錄。

```powershell
# 預設 DX12 / Epic / offscreen / 隱藏背景程序。
.\scripts\start_carla.ps1
.\.venv\Scripts\python.exe scripts/verify_phase0.py --run-id my-new-phase0-run

# Town01 初次載入後沿用；路線與門檻已在 configs 凍結。
.\.venv\Scripts\python.exe scripts/baseline.py run --output artifacts/runs/my-new-baseline
.\.venv\Scripts\python.exe scripts/rig_capture.py --output artifacts/runs/my-new-rig

# 24 感測器先五幀測試；--probe-scene 1 可驗證 NPC 清理。
.\.venv\Scripts\python.exe scripts/collect_dataset.py probe --output data/my-new-probe
.\.venv\Scripts\python.exe scripts/collect_dataset.py collect --output data/my-new-pilot
.\.venv\Scripts\python.exe scripts/validate_dataset.py data/my-new-pilot

.\scripts\stop_carla.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

verify_phase0 等 RPC 後集中執行 doctor／smoke，各子程序最長 300 秒。14 張實際 RGB、140 幀遙測與原數值門檻均保存；額外 brake tick 只排空 readback，不改善驗收的最終車速。命令失敗時先看 run 報告，不繼續下一步。

collect_dataset prepare 僅在建立新的版本化場景計畫時執行；已提交的 configs/dataset_scenes_v1.json 不覆寫。場景按幾何選定與分組，整個 scene 不跨 split；這不保證共用城鎮中的所有可見內容完全互斥。

## 資料與證據

- evidence/phase0/：環境、原生修正、歷史失敗及正式通過報告。
- evidence/phase1/：五條規則路線結果，標記 simulator GT／BasicAgent。
- evidence/phase2/：八鏡頭、資料 pilot 與獨立稽核結果。
- data/town01-pilot-v1-epic/inputs/：RGB 與不含 GT 的輸入 manifest。
- data/town01-pilot-v1-epic/calibration/：八方位 K 與相機至自車變換。
- data/town01-pilot-v1-epic/labels/：GT 語意、深度、NPC 3D boxes、自車狀態與道路；正式推論禁止讀取。
- data/town01-pilot-v1-epic/indexes/：train／validation／test 整群組清單；audit.json 與 image_inventory.jsonl 為磁碟檢查及逐圖 hash。

語意 PNG 紅色通道存原始 class ID；深度 PNG 保留 CARLA 24-bit packed RGB，不能當普通灰階。公式與依據保存於 encoding.json。3D boxes 包含不可見／遮擋 NPC，尚未建立逐相機可見度標籤。

大型模擬器、影像、logs、環境與權重不進 Git。4320 PNG 是小型管線資料，尚缺長序列、移動自車、多城鎮、行人與兩輪車。所有工作僅限模擬研究。

Low 版本保留在 data/town01-pilot-v1b；其完整性通過，但抽樣雨天有黑天空與明顯雜訊。固定場景 Epic 對照改善後，另收目前 Epic 版本；兩者都不是完整研究資料集。

## Phase 2 v2 動態資料

新的凍結計畫為六個十秒場景，自車使用規則導航，包含行人、自行車、機車與汽車；影像保存 5Hz、模擬步進 20Hz。每場景 50 組、共 300 組／7200 PNG。不要把這些控制器稱為訓練好的 AI。

```powershell
# server 啟動並確認 Town01 已載入後，先短測試，再完整收集。
.\.venv\Scripts\python.exe scripts/collect_dataset.py probe --criteria configs/dataset_motion_criteria_v2.json --plan configs/dataset_motion_scenes_v2.json --output data/my-motion-probe
.\.venv\Scripts\python.exe scripts/collect_dataset.py collect --criteria configs/dataset_motion_criteria_v2.json --plan configs/dataset_motion_scenes_v2.json --output data/my-motion-dataset
.\scripts\stop_carla.ps1
.\.venv\Scripts\python.exe scripts/validate_dataset.py data/my-motion-dataset --criteria configs/dataset_motion_criteria_v2.json
.\.venv\Scripts\python.exe scripts/export_training_index.py data/my-motion-dataset
```

完整單場景回歸可在 probe 加 `--probe-bundles 50`；它仍是 probe，不能直接匯入訓練。prepare_motion_dataset.py 只用來凍結新版本計畫，不覆寫既有 scenes。

SegmentationSamples 輸出 uint8 RGB HWC／語意 ID HW，保留整 scene 切分；export_training_index.py 另外輸出各 split 的 per-view RGB／target 清單，並以 Pillow 獨立解碼每場景首尾八視角。load_inference_rgb 只讀 RGB，不讀 GT。只有完整 acquisition 與 audit 均 passed 的資料可進入訓練讀取器。

新版位於 data/town01-motion-v2-001；舊停駐 pilot 與失敗的 motion probes 保留。僅六組共享 Town01、固定演出與有限車型，尚不足以宣稱跨城鎮或真實道路泛化。相機與 timestamp 的完整性、類別覆蓋通過，不代表感知模型準確率。
