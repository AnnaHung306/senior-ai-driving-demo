# 技術決策與風險

## D001 — 2026-10-16：先驗證 CARLA 預編譯版

選擇 CARLA 0.9.16 UE4 Windows 預編譯包，client 同版。硬體：Core Ultra 7 165H、22 邏輯核心、約 63.4GiB RAM、RTX 4070 Laptop 8188MiB、初始可用 346.5GiB。
CARLA 官方 quickstart 建議 8GB GPU；這只是啟動可行性依據，吞吐量必須實測。
不先編譯 Unreal，不先建立整個 AI 模型。原生 Windows 能先減少額外 WSL 圖形環境變數。

比較：DRIVE Sim 需先確認授權與可用環境，目前無此證據；Unity / 自製 Unreal 需要額外重做道路、感測與標註，先不採用。縮小實驗以 CARLA Low / 小影像 / 無交通開始；若 GPU 不足，可離線收集、關閉 CARLA 後訓練，或日後指定遠端 CARLA host。這些替代方案都不能替代本機 Phase 0 的實測結果。

## D002 — Python 與 CUDA 分階段驗證

現有系統 Python 3.13.5 不適合官方 CARLA 0.9.16 wheel。已使用 Codex bundled Python 3.12.14 建立專案 `.venv`，沒有修改系統 Python。
bundled runtime 的 ensurepip 失敗，改由系統 pip 指定 `--python .venv/Scripts/python.exe` 安裝 pip / carla。
此 venv 依賴 bundled base runtime 路徑；長期重現應改由獨立維護的 Python 3.12 執行 `scripts/setup.ps1` 重建，不搬移 venv。
目前不安裝 PyTorch / MMDetection3D；CUDA driver 可查詢不等於 PyTorch CUDA 已驗證。

## D003 — 感知候選與八鏡頭風險

BEVFormer 與 BEVDet 有官方開源實作。BEVFormer tiny config 標示至少約 6700M GPU memory，不能據此保證與 CARLA 共用 8GB GPU 可行；其資料相機數與本案八鏡頭也需要調整並重新驗證。
先完成 2D 單鏡頭小批次感知，再使用自採八鏡頭 calibration / grouped split，逐步處理幾何、BEV、時序與 occupancy。
關鍵風險：顯存競爭、Windows 舊 CUDA extension 編譯、模擬域過擬合、scene 洩漏、鏡頭同步延遲、遮擋深度不確定、真值進入推論、sensor timeout 未停止車輛。
對應措施：分時訓練、鎖版與獨立環境、保留場景測試集、資料 API 分離、逐幀 frame ID 檢查及超時失敗；完整閉環安全停止待 Phase 9 實作。

## 研究來源（2026-10-16 查閱）

- [CARLA 0.9.16 release](https://carla.org/2025/09/16/release-0.9.16/)：Python 3.10 / 3.11 / 3.12 wheel；固定採此版本。
- [CARLA quickstart](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)：預編譯包與啟動需求；頁面下載推薦仍寫 0.9.15，版本選擇以 0.9.16 release / PyPI wheel 為準。
- [Official release downloads](https://github.com/carla-simulator/carla/releases/tag/0.9.16)：Windows 官方下載連結。
- [PyPI carla](https://pypi.org/project/carla/0.9.16/)：cp312 win_amd64 wheel。
- [CARLA synchrony](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/)：同步、固定 timestep、單一 tick client 與 GPU sensor 延遲。
- [CARLA sensors](https://carla.readthedocs.io/en/0.9.16/ref_sensors/)：RGB frame、timestamp、BGRA 與相機參數。
- [BEVFormer official](https://github.com/fundamentalvision/BEVFormer)、[tiny config](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py)：相機 BEV / 時序模型候選。
- [BEVDet official](https://github.com/HuangJunJie2017/BEVDet)：多鏡頭 3D / BEV 候選。
- [BEVFormer 安裝](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md)：範例依賴 Python 3.8、torch 1.9.1 / CUDA 11.1、mmcv-full 1.4.0、mmdet3d 0.17.1，與本案 CARLA Python 3.12 不同；不能直接混裝，後續需要獨立環境與適配實驗。
- [BEVDet dev3.0 runtime](https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/requirements/runtime.txt)：包含舊版 numba / networkx 等約束，列為相容性風險，尚未試裝。
- [TorchVision 微調官方教學](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)：自訂 dataset / boxes / labels / image_id 與 COCO 評估流程，可作 Phase 3 的小規模可驗證起點；尚未安裝或訓練。
- [指定影片](https://www.youtube.com/watch?v=--cGYatMESY)：已嘗試存取但工具無法讀取內容；未據此捏造影片分析，僅採用需求書列出的公開概念。

官方 Windows archive 約 7.3GiB，需另留解壓空間；下載 checksum 僅記錄本地 SHA256，不宣稱已比對發布者 checksum。

## D004 — 2026-10-16：車輛物理 tick 崩潰，停止重複修正

歷史決策；使用者後續授權繼續深入嘗試，最新結果與原生修正見 D005。

**證據**：官方 0.9.16 archive CRC 通過、API / RPC 0.9.16 成功。完整 smoke 的 D3D12 / D3D11 均失敗，沒有有效影像。Tesla 與 Lincoln 的無攝影機隔離 probe 也在第一個 tick 後失去 server。CrashContext 為 TaskGraphThreadNP 存取違例，bIsOOM=0。切換後端沒有解決，不能宣稱顯存不足、攝影機或特定車型就是根因。

此為 Phase 0 重大阻礙；依規格第 7F 與第 11 節，不再反覆啟動相同失敗條件，不開始模型訓練或視覺化。所有本次 server 已退出，無背景工作正在進行。

| 路線 | 成本 / 限制 | 決策 |
|---|---|---|
| 獨立 CARLA 0.9.15 Windows + matching client | 再下載大檔、先核實 wheel 相容性；仍可能有相同故障 | 選為下一個最小研究任務，尚未執行 |
| 可用 Linux / 遠端 CARLA server | 需取得實際可存取的主機與 GPU；目前無主機資訊 | 備選，不假設已可用 |
| 小型 CPU 模擬器替代實驗 | 可縮小控制 / 感測試驗，但道路 / 標註 / 感知域需要重新驗證，不能冒充 CARLA 成果 | 僅作後備研究方案 |

下一個實驗先重現短物理 / RGB probe，通過才跑完整 smoke；保留原版失敗紀錄，固定新版本、輸入與門檻後再測試。此決策沒有刪除八鏡頭、AI 訓練、BEV、3D 或完整自主駕駛的長期目標。

## D005 — 2026-10-16：PixelReader 排程修正與精確影格 pipeline

以本機 LLVM / 官方 PDB 解析初次 crash 後，新增有限對照。空場景與車輛可先步進，不能再推論物理引擎本身故障。不同 Python、renderer、consumer 與節奏仍在五個 RGB callbacks 後停滯。

擷取**僅本專案 server** 的輕量 dump，StatsThread 指令在 `_Thrd_yield`，頂部 return-address candidate 對應 PixelReader.cpp:64 的 GPU readback 等待。DIA 讀取同一 PDB 的 enum：`StatsThread=0`、`HighTaskPriority=512`、`AnyNormalThreadHiPriTask=767`。官方 PixelReader.cpp:58 與實際反組譯都傳 512 給 AsyncTask；其 thread-index 位元為 0，會選到 StatsThread，而不是任意 worker。

本地僅把該 verified callsite 的 enum 改為 767，保留高 task priority，指定 AnyThread。已對照官方 ZIP 原檔、保存 `.exe.stock`、驗證一 byte 差異，並實際測試 restore → reapply 雜湊一致。Source patch 在 `patches/pixel-reader-task-thread.patch`，長期應用 source patch 重建；目前的 binary 修正只適用這一個已核實 build，腳本拒絕其他 build。

- Stock executable SHA256：`ba05ec36f7f0b548b266865ea2787a1dc5a028e298ebefcf0c49196ac629c4d3`。
- Local fixed SHA256：`9deba9928ffe56ee8c27cc89d88bcdd4e49204b771f86da2e438c613f4205245`。
- `scripts/patch_pixel_reader.py --restore` 可還原；套用不修改原 ZIP、不修改驅動或全域設定。
- Source patch 初次零 context 檢查失敗，已改為正常 context diff，並以 `git apply --check --ignore-space-change` 對官方 source fixture 驗證通過。此格式問題沒有改變實際二進位測試內容。

修正後 server 在 sensor timeout 後仍正常回應；再用一 tick lookahead，三張影像與原始 snapshot 幀號及時間戳全數配對通過。正式 v2 protocol 先凍結，再測原數值門檻，140 影格驗收通過。這不是把前一張影像當成目前影像，控制／遙測均按原始 frame 保存。

CARLA 0.9.15 的官方 wheel 僅到 Python 3.10，已準備 matching diagnostic runtime；大型 server 下載在 6,265,982,548 bytes 停滯後自行退出，等待安裝程序已中止，**未完成 0.9.15 測試**。既然 0.9.16 已通過，不再重下載這份備案。原始 source 的兩版本都含同一排程寫法，不能假設單純降版就可解決。

目前是模擬器工程驗收通過，仍沒有 AI 感知模型或真實道路能力。
