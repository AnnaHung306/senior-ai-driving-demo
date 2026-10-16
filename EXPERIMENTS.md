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
