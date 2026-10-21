# 多鏡頭三維環境感知與自主駕駛研究系統

長期需求見 [PROJECT_SPEC.md](PROJECT_SPEC.md)，當前任務見 [CURRENT_TASK.md](CURRENT_TASK.md)。

**2026-10-21：Phase 8–9導航與真閉環工程研究版已完成。** 網頁可選Town01道路起終點、預覽路線、啟動八RGB→本機CUDA模型→規劃→控制，並看3D、八鏡頭與完成紀錄回放。五個最終具名直路／轉彎／紅綠燈／障礙車／橫穿行人案例抵達，碰撞與越線皆0；另從網頁自訂11m路線實際抵達，人工停止、WebGL中斷及真相機timeout均驗證煞停。295 Python、48 JS、23真HTTP、1123保存影格與411真browser影格核對完成；18次開發嘗試的失敗與中斷均保留。**目前依賴地圖與自車定位先驗，推論約1.2秒／循環、牆鐘慢8–9倍；研究品質與任意路線可靠性未通過。** 執行 `./Start-NavigationDemo.ps1 -StartSimulator`，開啟 `http://127.0.0.1:8768/`；純回放用 `-ReplayOnly`。詳見[部署與展示方式](DEPLOYMENT.md)、[Phase 8–9結果](PHASE89.md)及[封存證據](evidence/phase89/navigation-closed-loop-v1-001/README.md)。亦已準備不需GPU／CARLA的本機可攜回放ZIP。以下保留舊階段交付狀態；目的地導航與閉環進展以本節為準。

**2026-10-20：Phase7動態展示工程研究版已完成。** `./Start-WorldViewer.ps1`預設開啟240組RGB推論的速度／方向、3秒軌跡、暫失與接近風險；另有四種明示非RGB AI的合成診斷。256 Python、40 JS、22 HTTP及真browser全240組／1091物件／889預測核對通過，render median60FPS、CPU套用p9544.4ms。原模型選擇validation速度向量MAE .290m/s、ADE1 .950m、FDE2 1.205m，2秒GT覆蓋61.22%（all-GT miss47.52%），ID switches2與舊版相同。**這是運動學研究展示；正式品質、未見路線、連續即時推論及控制仍未通過／未提供。** 見[Phase7結果與啟動方式](PHASE7.md)及[實測證據](evidence/phase7/causal-motion-v1-001/README.md)。以下保留各階段原交付狀態，速度／預測以本節為準。

**2026-10-20：Phase5工程再驗證＋Phase6互動3D研究版已交付。** 執行 `./Start-WorldViewer.ps1`，再開啟 `http://127.0.0.1:8766/`。含三視角、物件檢視、道路線／局部候選路徑、佔據格／八RGB對照、四序列時間軸及已驗證輸出監看。235項Python、20項JS與真瀏覽器操作通過，32影格／147物件／256RGB來源一致；本機render median60FPS、CPU影格套用p9541.1ms。**模型品質、模擬器連續即時推論、目的地導航及控制仍未通過／未提供**，60FPS畫面不是感知頻率。見[使用與狀態矩陣](PHASE6.md)及[截圖／實測證據](evidence/phase6/world-viewer-v1-001/README.md)。

**2026-10-20：Phase 5 已完成首版訓練與 World Model 串接。** 新增 RGB 3D 方框模型、八鏡頭融合、基本追蹤 ID、粗 semantic occupancy 與 WorldFrame 契約；223 項測試通過。validation 短片的嚴格定位P/R為61.11%／55.93%，完整3D方框Recall@IoU.25僅25.42%，仍有朝向、行人佔據與ID切換問題，**Phase5品質尚未通過**。已保存獨立GT／先驗比較、全部錯例與來源，不追加盲目訓練。見[Phase5規劃](PHASE5.md)及[圖文實測報告](evidence/phase5/rgb-cuboid-world-v1-001/README.md)。

**2026-10-20：Phase 4 首版道路 BEV 已實作並完成有限實測。** 八鏡頭幾何投影、跨鏡頭地面接觸點去重、RGB 時序配準及標線中央候選軌跡已串接；193 項回歸測試通過。固定 32 組原 train／validation 畫面的八鏡頭時序道路 IoU 95.20%、標線 F1 90.27%；候選軌跡平均 RMSE 15.9cm，其中單側標線加寬度先驗為 30.6cm、雙側標線為 1.1cm。這是已知短序列的原型結果，尚待新完整路線驗收；物件定位 P/R 為 60.3%／55.0%，每組計算約 1.10 秒，未達即時行駛。見[執行規劃與後續條件](PHASE4.md)及[實測圖文報告](evidence/phase4/geometric-bev-v1-001/README.md)。

**最新決定：Phase 3 依目前成果收尾，深度分支不納入，不再追補 Wall。** 後續以晴天、無近牆路線及道路／車道線跟隨為預設。歷史驗證未達項目保留，下一步重心轉向道路 BEV 與行駛整合；見 [階段收尾與後續工作](PHASE3_CLOSEOUT.md)。以下「未通過」描述原門檻結果，不表示要重啟 Phase 3。

實測已通過：CARLA 車輛加速／煞停與單鏡頭、五條固定規則導航路線、八鏡頭同步擷取，以及六場景資料 pilot；本輪另完成含移動自車、行人與兩輪車的 Phase 2 v2 基礎資料驗收。使用本地修正 CARLA 0.9.16 與 DX12／Epic；官方原樣 binary 的失敗紀錄保留。

目前以 **ClearNoon理想晴天** 為範圍。Phase 3主資料007已完成18場、86,400 PNG的獨立audit與assistant視覺抽查。最新v6批次11階段、GPU錯例重播與18頁實際視覺檢查均完成。17類MobileUNet v6微調6輪，使用原Town01/03與新Town06/07各1024個核准靜態分割視角；完整validation focus mIoU **60.65%**、fence IoU **60.58%**、可見邊界F1 **.6222**通過，僅wall IoU **15.65%**未達20%。偵測v6微調4輪後最佳為第一輪：mAP50 **70.46%**，自行車／機車／紅綠燈Recall未達50%。號誌v1-004 GT crop macro **94.38%**通過，但搭配v6 detector的joint macro **39.9064%**未達40%。八RGB展示與三種benchmark已保存，8view暖機推論p50 **688ms**（排除模擬器與控制）。development/lockbox未開模型評分，**整個Phase 3未通過**；完整結果見[v6批次證據](evidence/phase3/sunny-v3/phase3-recovery-batch-002-completed/README.md)。道路 BEV、粗occupancy及本機3D介面已有研究版；正式即時導航與AI駕駛閉環尚未完成。

目前協議見 [PHASE3_TRAINING_PROTOCOL.md](PHASE3_TRAINING_PROTOCOL.md)，逐類成績見 [v6 MobileUNet模型卡](evidence/phase3/sunny-v3/sunny-segmentation-v6-001-completed-validation/MODEL_CARD.md)、[v6 detector模型卡](evidence/phase3/sunny-v3/sunny-detection-v6-001-completed-validation/MODEL_CARD.md)及[號誌模型卡](evidence/phase3/sunny-v3/phase3-recovery-batch-002-completed/export/light_state/MODEL_CARD.md)，泛化缺口依據見 [PHASE3_DOMAIN_RECOVERY.md](PHASE3_DOMAIN_RECOVERY.md)，中斷接續見 [CURRENT_TASK.md](CURRENT_TASK.md)。權重與完整逐幀結果保留於本機artifacts/runs（ignored）；可追溯數值、凍結source及實際檢視圖存於evidence/phase3。最新分割錯例已重播原GPU batch並核對confusion／boundary exact，CPU診斷不取代GPU驗證。

深度輔助分支已完成32步GPU可行性檢查及control／depth各一輪（各1712更新）的完整3600張validation對照。其他15類mIoU **64.0694%→64.1508%**，只有+0.0814個百分點，未達事前0.2pp研究篩選值；行人／人行道改善，但bike／motor／rider退步，Wall仍17.53%未過。兩組實際增強影像bytes一致，3頁比較圖已實看；這版先停、不追加第二輪或正式納入。詳見[深度A/B完整結果](evidence/phase3/sunny-v3/depth-auxiliary-AB-v1-001-completed/README.md)。

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
