# 實驗紀錄

## ENV-001 — 2026-10-16，環境建立

- 模型 / 權重 / dataset：無；未開始訓練。
- 機器：Windows 11 build 26200、Core Ultra 7 165H、22 logical CPU、63.43GiB RAM、RTX 4070 Laptop 8188MiB、driver 595.95。
- 初始磁碟剩餘約 346.5GiB。
- 系統 Python 3.13.5 不相容 CARLA wheel；用 bundled Python 3.12.14 建立 `.venv`。
- `ensurepip` 失敗，使用既有 pip 的 `--python` 指定 venv 補齊 pip 26.2.1 / carla 0.9.16。import 成功。
- Windows CIM 查詢受限；改用 nvidia-smi、Win32 memory API、registry，成功取得硬體資訊。
- doctor：Python / API / GPU / RAM / disk 通過，server RPC 未就緒。

## TEST-001 — 2026-10-16，驗收邏輯

- 首次 6 項 unittest：5 通過，1 項在 sandbox tempdir 清理遇到權限錯誤。
- 修正測試將拒絕寫入的 NaN 報告位置放在專案 artifacts，不再依賴外部 tempdir 清理。
- 第二次 6 項全部通過；沒有更動驗收數值門檻。
- 故障測試：`scripts/run.py smoke --port 29999 --timeout 1 --output artifacts/runs/phase0-unavailable-server`。
- 結果：退出碼 1，保存 server timeout / 空 metrics / ai_model=null；符合預期，不誤報通過。

## INSTALL-001 — 2026-10-16，官方模擬器下載

- 官方 release 指向 `https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip`。
- HEAD：7,812,252,810 bytes；ETag `ee067f2519194c02ea0120f1657c09c9-117`；本地 hash 完成後另記。
- Sandbox DNS 不可用；經允許的 network escalation 後可連線。
- Python urllib 預設 UA 曾收到 403；curl 完整下載可工作。
- Range / 分段請求停滯，已終止該方案，不再反覆嘗試；改用有時限的單連線完整下載。
- 等待下載後程式自動驗證 archive、CRC、解壓、啟動及集中驗收；目前未有可宣稱的實機測試結果。

## 模型版本登錄

尚無模型版本。後續每次訓練必須記錄 architecture、初始化來源（自訓 / 微調 / 直接預訓練）、dataset / split 版本、seed、hyperparameters、耗時與峰值資源、權重路徑、測試版本、逐類指標、已知限制。

## START-001 — 2026-10-16，CARLA 啟動等待

- 官方 archive 下載完成；SHA256 `0972f26fa3f1c741fbfcb4602b4ccf520415b8f8cf3d06cb83c9502fe8118bd2`，ZIP CRC 通過，解壓約 18.07GiB。
- 第一次自動等待 RPC 180 秒後失敗；已保存 `evidence/phase0/startup-attempt-001.json`。
- 同時獨立新建 client 可成功取得 server 0.9.16；重新執行 doctor 全數通過。
- 推測啟動前建立的 client 保留失敗連線狀態；最小修正為每次 readiness retry 重建 client，保留原 180 秒上限與所有車輛驗收門檻。
- 此次尚未執行車輛 smoke，不將 doctor 通過冒充車輛驗收完成。

## SIM-001 — 2026-10-16，第一次實機控制 / RGB

- 程式版本 `9a965b2`，原始 phase0-v1 門檻未修改。
- doctor 通過；smoke 在第一張 RGB 前 server 崩潰，沒有有效行駛指標，結果 failed。
- CrashContext：D3D12、RTX 4070 Laptop、TaskGraphThreadNP 0、NULL access violation；bIsOOM=0。不能僅根據觀察到約 7GB GPU 使用量就判定是 OOM。
- 證據 `evidence/phase0/crash-attempt-001.json` 與 `simulation-attempt-001.json`。
- 下一個有限對照實驗僅改渲染後端為 `-dx11`；保持同一地圖、攝影機、控制 protocol 與驗收標準。
- 此為相容性假說，並非已證實的根因或修復。

## SIM-002 / PROBE-001 / PROBE-002 — 2026-10-16，有限對照與停止決策

- D3D11 完整 smoke 仍在第一張影像前 crash，沒有有效指標；渲染後端切換未解決問題。原始報告保存在 evidence。
- Probe 先載入相同 Town01 / 同步設定、建立車輛、tick，再準備加入 camera。Tesla 與 Lincoln 均在沒有 camera 時僅完成一個 tick 後 server 崩潰；因此 camera 不是必要觸發條件，仍不能精確宣稱物理引擎根因。
- 官方 PDB 的本地解析嘗試回報 error 126，未取得符號，沒有根據未解碼位址臆測函式。
- 依避免無限修正規則停止重跑 0.9.16 本設定；提出 D004 決策報告，下一個任務選為隔離版本對照，未執行新版本下載。
- verifier 增加 300 秒子程序期限，將重複 streaming 原始 log 留在檔案，只回傳精簡狀態，避免 LLM 可見輸出膨脹。

## DEBUG-002 — 2026-10-16，官方符號解析成功

- 使用者授權繼續多方向嘗試；先回看共同前置條件，避免重跑原先的 renderer / vehicle 組合。
- 發現本機 Visual Studio 2022 安裝含 `llvm-symbolizer.exe`，對官方 Shipping.exe + PDB 執行 `--relative-address` 成功。
- `0x2331d26` → `FLandscapeRenderSystem::FetchHeightmapLODBiases`（LandscapeRender.cpp:986）；`0x2f570ec` → `UTexture2D::GetNumResidentMips`（Texture2D.cpp:213）。輸出保存 `evidence/phase0/landscape-symbols.txt`。
- 這項新證據定位到地形渲染／貼圖讀取，不能再將「物理 tick 時崩潰」的時間關聯當成物理引擎故障的根因。
- 新矩陣 `configs/landscape_diagnostics_v1.json` 在執行前固定：Town01 只 load 一次、啟動時直接 Town01、原 reload 但 Epic。先測空場景，再車輛，再精確同幀 RGB。既有完整驗收門檻保持不變。

## LANDSCAPE-001 — 2026-10-16，移除第二次 reload

- v1 首案 `load-once-low`：空場景 10 ticks + Tesla 20 ticks 成功；RGB 第一幀等待逾時，但 actor 清理與 settings restore 沒有錯誤，server 並未像原先崩潰。
- 批次切換下一案時，Windows 終止程序後的埠釋放競態使防護中止；因此 v1 第二、三案**沒有執行**。
- 修正 runner 在停止自己啟動的程序樹後，等待埠確實關閉再啟動下一案。
- 新 v2 先沿用 working map lifecycle，只增加 5 個有記錄的 sensor startup warmup ticks，之後仍要求 image.frame 等於 world.tick；不是容許錯配或跳過正式驗收。

## DEBUG-003 — 多版本有限診斷與真正等待點

- 使用者要求繼續嘗試後，v1–v12 的獨立 server 實驗、原始結果及雜湊集中於 evidence/phase0/diagnostic-matrix-results.json；每案有期限、設定讀回及清理，不重寫失敗狀態。
- 載入流程、不同官方 CPython 3.12／3.11、callback 所有權、緩衝、prefetch、節奏、on_tick queue、RHI／單執行緒／Vulkan 等隔離，沒有單獨解決原始停滯。啟動時指定 Town01 的 launcher 參數未生效，map readback 拒絕該案，不能當成 Town01 成功。
- 輕量 dump、匹配 PDB／DIA enum 與官方 PixelReader source 定位到 StatsThread 的 GPU readback 等待。不是看著未解碼位址猜測；stack scan 與真正 instruction pointer 分開標示。
- Normal enum 767 修正後，嚴格等待目前 frame 仍逾時；採用一 tick pipeline、將影像配回原始 snapshot 才通過三幀 probe。正式 protocol v2 在 140 幀測試前凍結，原數值門檻沒有降低。
- 正式 normal 與 background 修正版均通過 140 幀與五條固定規則路線。重複載入已在 Town01 的地圖曾造成 Baseline preflight 失敗；修正為沿用並驗證 OpenDrive hash，未刪除失敗紀錄。
- 0.9.15 備案下載在 6265982548 bytes 停滯後退出，沒有完成 server 測試；不宣稱降版有效。

## RIG-001 — 八鏡頭與繪圖後端

- normal／DX11 八鏡頭失敗；dump 顯示多個 normal workers 含 PixelReader wait candidates。背景池 enum 2815 保留 task priority，但 DX11 八鏡頭依然失敗。
- 第二案 CrashContext 實際為 GPUCrash／D3D device lost 0x887A0020；不是憑 GPU 用量宣稱 OOM。
- 修正後僅換 DX12：phase0-dx12-background-003 通過原門檻；rig-eight-003-dx12 通過 20 組／160 PNG、零時間戳誤差與完整清理。約 4.34 組／牆鐘秒，取樣 GPU 最高 2824MiB，不是即時 20Hz 或模型峰值顯存。
- 最終 DX12 五路線回歸 phase1-baseline-004-dx12 為 5/5，無碰撞／跨線／off-driving-lane。仍為 simulator GT BasicAgent，號誌行為沒有刺激。

## DATA-001 — 多模態資料與 NPC 清理

- 六場景／rig／門檻在收集前凍結，先 24 感測器五幀 probe 通過。
- 首次全場景在 scene-01 清理時程序退出，磁碟其實已寫完該案 30 組；NPC 五幀獨立重現以 faulthandler 證實 native abort 在 finally 的 world.apply_settings。兩次均記為失敗。
- 修正為先取消 NPC autopilot 註冊，world 恢復前保持 TM 同步，才關閉 TM 同步；同一 NPC 五幀 probe 通過，之後 town01-pilot-v1b 六案均完整通過。
- 共 180 組／4320 PNG，RGB 與 GT 深度／語意／3D／道路分存；630 筆 NPC box 標註，按原始 snapshot 複製，未混入未來 actor transform。
- 獨立磁碟稽核 4320 PNG 的解壓／尺寸／CRC、逐圖 SHA256、每圖 frame／timestamp、K/T、GT 對齊與整群組 split 全通過。train／validation／test=90／30／60，群組交集 0，原點最小距離 164.88m。
- 匯出 JSON 排版與原 config byte 格式不同；auditor 比對凍結來源檔 byte hash，並比對匯出 JSON 值，未修改感測器資料或門檻。audit-provenance.json 保存實際 auditor hash。
- 新增資料邊界測試初次又遇到 sandbox 外部 tempdir 子資料夾權限；改用專案 artifacts/runs/test-temp，之後 13 項通過。保留此次失敗原因，不改產品門檻。

## QUALITY-001 — 資料完整性與視覺品質分開

- 從每個場景第一組抽八視角，共 48 張 RGB。亮度平均範圍 6.15–176.02，最暗夜雨畫面近黑像素約 59.91%；實際檢視看到 Low／DX12 雨天有黑天空與明顯雜訊。這些為觀測描述，不是事後設定的通過門檻。
- 完整性通過不等於影像足以訓練。先凍結 render_quality_probe_v1.json，以同一雨天 scene-02／seed／rig 作 Epic 對照，不修改驅動或全域設定；新結果另記。
- Pilot 停駐自車、每場景 1.5 秒、單一 NPC 車型、共享 Town01，且缺行人／兩輪車；完整 Phase 2 未通過，無模型／checkpoint／準確率宣稱。

## QUALITY-002 — Epic 對照與最終 pilot

- Epic 單鏡頭 Phase 0 原門檻通過；同一雨天／六 NPC／24 感測器 probe 通過。Shipping 命令列核對確實為 Epic，人工檢視天空／光照恢復；雨霧、反射與模糊仍存在。不宣稱圖像品質基準全面通過。
- 新資料 town01-pilot-v1-epic 不覆寫 Low；六場景均通過，共 180 組／4320 PNG。source 08c718d；新 auditor 4a4dd8b。
- Epic auditor 92.875 秒完成，4320 PNG、約 677.4MB、CRC 錯誤 0、同步誤差 0；train／validation／test 90／30／60，群組交集 0。K／T 數值符合凍結 FOV 與安裝幾何，630 筆 NPC box GT。
- Epic 48 張探索性 RGB 樣本平均值範圍 14.92–167.10、最大近黑比例約 49.49%；不是事後制定的品質閾值，也不是模型效能。
- phase1-baseline-005-epic 最終回歸 5/5，無碰撞／跨線／off-driving-lane。模擬器已關閉，未啟動模型訓練。
- Background native patch 實際 restore／reapply，前後精確為 b0370d6d...；原檔 ba05ec36... 保留。14 項單元測試通過。

## REGRESSION-001 — 同 server 重跑

- source c853f10；smoke 改為僅地圖不同時 load，保存 map_lifecycle，未更動 protocol／門檻。
- 同一 server 的 005a 初次 load 與 005b reuse，兩次 doctor／140 幀 smoke 均通過。測後關閉自有 server；14 項單元測試再次通過並保存 log。
- 已核對完整路徑及原始失敗下載大小後，刪除唯一未完成的 0.9.15 archive cache（約 6.27GB）；0.9.16 完整 ZIP、stock 備份及失敗紀錄仍保留。

## MOTION-001 — Phase 2 v2 凍結與短測試

- criteria dataset-motion-v2、scenes town01-motion-scenes-v2 在影像收集前固定；保留既有六個空間群組／split，route roads 在 train／validation／test 分別為 [8,11,12,24,256]／[3,13,15]／[6,10]，交集 0。
- 舊 stationary probe 回歸通過。新 20-tick 短測試 5 組／120 PNG 通過，實際語意出現 12 pedestrian、18 motorcycle、19 bicycle，不以生成角色取代覆蓋驗證。
- 完整 motion-v2-full-probe-002：50 組／1200 PNG、同步與移動通過，但 ego collision callbacks=10，結果 failed。其他 actor 259 為對向車；ego 先偏向對向車道。
- 單車初始 snapshot 診斷證實第一 tick 前讀取位置為 (0,0,0)，不是 spawn pose。source 1f61d27 改成暖機後導航；相同 frozen scene 的 full-probe-003 通過：自車位移 17.6586m、標籤跨度 9.8s、五個角色位移皆超過 1m、碰撞 0。失敗資料與 JSON 均保留。
- 全場景 run town01-motion-v2-001 依相同 plan 啟動，任一 case failed 即停止；完整結果在集中分析時另填，不提前宣稱通過。

## FORMAT-001 — 獨立訓練資料讀取

- 安裝 Pillow 12.0.0，新增 requirements-dataset.txt、SegmentationSamples 與 export_training_index.py。
- 舊 Epic stationary 資料成功輸出 train／validation／test=720／240／480 個 per-view 樣本；首尾每 scene 八視角共 96 個樣本，用 Pillow 實際讀 RGB／原始語意紅通道與形狀檢查通過。
- 19 項單元測試通過；含新 cadence、actor identity、靜止角色拒絕、像素覆蓋不足拒絕、RGB-only 真正解碼與 probe 不可訓練。

## MOTION-002 — 六場景正式收集／稽核通過

- acquisition source 1f61d27；六案各 50 組、共 300 組／7200 PNG。自車位移分別 17.6586、17.2544、20.4886、16.5617、15.8407、17.5501m，各案 ego collision 0，角色移動／9.8s label span／原同步與完整性門檻皆通過。總 case wall 約 388.11s，未宣稱實時。
- auditor source 74db217／SHA256 c621a7381ad572a910f201b02ce442eba403ec2c0db47b2e987e2d7af20c8699；256.141s 通過全部 PNG，約 1.23GB、CRC 0、timestamp error 0。固定 class coverage 與 group／planned-road split 全通過；觀測 ego 跨 split 最小距離 147.36m，actor 最小位移 7.56m。
- 新資料 train／validation／test=150／50／100 bundles，training export=1200／400／800 view；各 scene 首尾共 96 個 sample 以 Pillow 獨立解碼通過。
- 19 項 unit tests 通過，沒有訓練、checkpoint 或感知準確率。資料與模擬器程序均已結束，Phase 2 v2 基礎版本驗收通過，下一步 Phase 3 CUDA／小型模型實驗。
