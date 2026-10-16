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

本地最初把該 verified callsite 的 enum 改為 767，保留高 task priority，指定 AnyThread。已對照官方 ZIP 原檔、保存 `.exe.stock`、驗證一 byte 差異，並實際測試 restore → reapply 雜湊一致。此歷史 normal 版本 source patch 在 `patches/pixel-reader-task-thread-normal.patch`；目前 active background 版本見 D006。長期應用 source patch 重建，binary 腳本只支援核對固定 SHA256 的 build。

- Stock executable SHA256：`ba05ec36f7f0b548b266865ea2787a1dc5a028e298ebefcf0c49196ac629c4d3`。
- Local fixed SHA256：`9deba9928ffe56ee8c27cc89d88bcdd4e49204b771f86da2e438c613f4205245`。
- `scripts/patch_pixel_reader.py --restore` 可還原；套用不修改原 ZIP、不修改驅動或全域設定。
- Source patch 初次零 context 檢查失敗，已改為正常 context diff，並以 `git apply --check --ignore-space-change` 對官方 source fixture 驗證通過。此格式問題沒有改變實際二進位測試內容。

修正後 server 在 sensor timeout 後仍正常回應；再用一 tick lookahead，三張影像與原始 snapshot 幀號及時間戳全數配對通過。正式 v2 protocol 先凍結，再測原數值門檻，140 影格驗收通過。這不是把前一張影像當成目前影像，控制／遙測均按原始 frame 保存。

CARLA 0.9.15 的官方 wheel 僅到 Python 3.10，已準備 matching diagnostic runtime；大型 server 下載在 6,265,982,548 bytes 停滯後自行退出，等待安裝程序已中止，**未完成 0.9.15 測試**。既然 0.9.16 已通過，不再重下載這份備案。原始 source 的兩版本都含同一排程寫法，不能假設單純降版就可解決。

目前是模擬器工程驗收通過，仍沒有 AI 感知模型或真實道路能力。

## D006 — 八鏡頭規模化與 DX12

Normal worker 版本的單鏡頭 140 幀與五條規則路線通過，但八鏡頭失敗；輕量 dump 的多個 normal worker 含 PixelReader GPU wait return-address candidates。改用 PDB 已核實的 AnyBackgroundHiPriTask=2815 隔離工作池，保留相同 task priority。此版本改兩個 byte，SHA256 `b0370d6dcd57b95299d12005cebad1dc6187804e83aeb51a184ccbc4b581fea8`，source patch 為 `patches/pixel-reader-task-thread.patch`。

Background 在 DX11 仍無法通過八鏡頭。新 CrashContext 對應 PID 3372，類型 GPUCrash，D3D device lost，0x887A0020 INTERNAL_ERROR。這是圖形裝置錯誤證據，不是 OOM 證據；當時取樣 GPU 約 2063MiB。移動執行緒本身不能宣稱解決八鏡頭。

先前 DX12 測試使用未修正的 lifecycle／排程，不能推論修正後也一定失敗。只改 graphics API 為 DX12，重新跑同一 Phase 0 v2 原數值門檻後通過，再跑原八鏡頭門檻亦通過，最後五條規則路線回歸 5/5。因此本專案目前預設 DX12 + background 修正；不修改驅動、登錄檔 TDR 或其他全域設定。

Dump 工具只讀專案自己的程序且不含完整 heap。stack scan 是 return-address candidates，並非已 unwind 的堆疊；實際 CrashContext、精確指令位址與匹配官方 PDB 分別記錄，不混稱。

## D007 — 24 感測器 pilot、GT 邊界與 NPC 清理

先凍結 `dataset_pilot_criteria_v1.json` 與 `dataset_scenes_v1.json`，按地圖幾何選六個不同道路群組，整群組先分 train／validation／test。先通過 24 感測器五幀 probe，再收完整資料，不因看過結果而挑選成功場景或降低影格／同步門檻。

NPC 場景最初在收完資料後 native abort；啟用 faulthandler 證實退出點在 finally 的 world.apply_settings，而非 3D box 計算。先取消 NPC autopilot 註冊，保持 TM 同步直到 world settings 恢復後再關閉 TM 同步；NPC 五幀 probe 與六場景完整收集均通過，原兩次失敗不回寫。

RGB manifest 僅存 RGB 路徑、frame／timestamp、scene ID 與 calibration；GT 深度、語意、自車 world pose、NPC boxes 及道路存 labels。物件 pose／velocity 取自該原始影格 snapshot，先轉成獨立 Python 資料，再推進下一 tick；不拿未來 live actor transform 配舊影像。RGB-only loader 會拒絕額外 GT 欄位及標籤路徑。

標籤編碼依 [CARLA 官方 sensors](https://carla.readthedocs.io/en/0.9.16/ref_sensors/)：語意原始 red class ID；深度保留 packed RGB，meters=(R+256G+65536B)*1000/16777215。沒有把調色盤或灰階展示圖作原始訓練標籤。Snapshot 依 [官方 ActorSnapshot API](https://carla.readthedocs.io/en/0.9.16/python_api/#carlaactorsnapshot)。

資料 pilot 只驗證擷取、同步、標籤隔離、群組切分與磁碟完整性。六個短停駐場景不具足夠多樣性；缺行人／兩輪車與長移動序列。保留完整 Phase 2 的要求，下一步先擴充可訓練覆蓋，再進入 Phase 3 CUDA／小型感知模型試跑。

## D008 — Low 畫質對照與 Epic pilot

磁碟完整性通過後抽看 RGB，Low／DX12 雨天有黑天空與明顯雜訊；不能因檔案可讀就直接當成訓練品質合格。先固定 render_quality_probe_v1.json，只以相同雨天 scene-02、rig、seed、五組／24 感測器比較 Epic。Shipping 程序命令列確實含 quality-level=Epic；單鏡頭原門檻與有六 NPC 的 probe 都通過，人工對照看到天空／光照恢復，仍有雨霧與模糊。

因此預設改為 DX12／Epic。原 Low 資料不覆寫、不改稱沒有瑕疵；Epic 另收六場景共 4320 PNG，五條規則路線也以 Epic 回歸 5/5。Probe 後 GPU 全裝置取樣 3382MiB，不是感測器 active 峰值。新 pilot 仍缺資料覆蓋，不將畫質改善等同模型準確率或完整 Phase 2 通過。

單元測試增至 14 項，加入不正確焦距／反向外參拒絕；Epic 磁碟 auditor 對每組 calibration 比對凍結 FOV、安裝位置／yaw 與 optical 軸轉換。原生 background 版本在 server 停止後實際 restore → reapply，雜湊完全回到 b0370d6d...，證據另存，不以文案宣稱可還原。

## D009 — smoke 重跑不再載入相同地圖

接手回歸檢查發現 smoke 雖移除了 reload_world，仍每次無條件 load_world(Town01)。新 server 正常，但重跑可能再次觸發已知 lifecycle 問題。改先讀目前地圖，只在不同時 load，記錄 map_lifecycle；原 protocol 與數值門檻不變。

以同一 DX12／Epic server 連續實測：phase0-map-reuse-005a=loaded_map_once、005b=reused_existing_map，兩者 doctor／140 幀 smoke 全通過，完整報告保留。測後僅停止路徑核對一致的自有程序樹；沒有留下訓練或監督工作。

## D010 — Phase 2 動態資料與初始 snapshot

新版 dataset-motion-v2 在收集前固定六組場景、各 200 個 0.05 秒 tick、每四 tick 保存一組（5Hz，共 50 組／場景），八方位 RGB／語意／深度不變。每場景有車、自行車、機車與兩名行人，自車使用 BasicAgent 規則控制，行人使用凍結人行道路徑，車輛使用 Traffic Manager。這是可觀測的模擬資料產生方法，不是 AI。

門檻新增自車位移至少 5m、標籤跨度至少 9.5s、每個參與者位移至少 1m、ego collision 0；每個 split 的道路／人行道／行人／汽車／機車／自行車／車道線各至少 1000 語意像素、10 張正樣本。以像素確認可見覆蓋，不把 spawn 成功當成影像樣本。保持既有空間群組切分，另檢查規劃路徑的道路 IDs 不跨 split。

幾何準備時，路口 waypoint 不一定有直接的 get_left_lane 連結；原準備程式在凍結前拒絕該案。改以 2.8–8m 距離與反向朝向條件，按幾何選最近反向車道，才產生固定 plan。未以駕駛成績選場景。

完整首案曾有 10 個碰撞 callbacks，不能通過。追查到導航初始化早於第一個 snapshot。單車診斷證實 spawn 後 get_location=(0,0,0)、snapshot 尚無該 actor，下一 tick 才回報實際 (92.11,227.22,0.2853)。改在暖機 snapshot 後才建立導航，與既有通過的 Baseline 順序一致。同一場景自車移動 17.66m，五個參與者均移動、碰撞 0、50 組／1200 PNG、9.8s 標籤跨度全部通過，沒有降低門檻。

## D011 — 訓練格式與推論 API 分離

新增固定 Pillow 12.0.0 於 requirements-dataset.txt，僅裝入專案 venv。其獨立 PNG decoder 可核對原本自製 lossless serializer 的實際色彩通道。SegmentationSamples 只接受完整 acquisition 與 audit 通過的資料，按整 scene 分組展開每個相機 view，回傳 uint8 RGB HWC 與原始語意 ID HW；不預先選模型、歸一化或重新映射類別。

load_inference_rgb 僅讀固定 RGB manifest／RGB PNG，不讀 labels、深度或 CARLA actor。新增測試確認沒有 labels 資料夾時仍可解碼，並拒絕將 probe／不完整資料匯入訓練。舊 Epic pilot 的 1440 個 per-view 樣本及各場景首尾共 96 個樣本，已用此獨立 decoder 實際驗證。

目前尚未安裝 PyTorch 或產生模型；先完成新版全場景資料驗收，再進入 Phase 3 CUDA／小型感知試跑。已查閱 PyTorch 官方安裝頁，頁面 selector 的 stable 顯示與部分文字版本不一致；正式選版前須核實可用官方 wheel，不把快取顯示當成全域最新版本。

## D012 — Phase 2 v2 基礎資料版本驗收

六場景 acquisition、全 7200 PNG 磁碟 audit、群組／路線 split、motion／class coverage 與 training export 均通過，允許下一步進入 Phase 3 的最小感知實驗。這是完成可訓練的基礎研究版本，不是宣稱資料覆蓋已足夠量產或模型品質已合格。

Train／validation／test 為 1200／400／800 per-view 樣本；獨立 Pillow decoder 抽查各場景首尾共 96 個樣本通過。19 項 unit tests 通過，資料卡與完整精簡證據見 evidence/phase2/motion-v2。探索性 RGB profile 48 張樣本平均值範圍約 15.28–164.99、最大近黑比例約 49.26%；夜雨低照度是限制，沒有事後畫質通過門檻。

長尾大型車、更多密度／城鎮、較長且更自然的交通互動仍待擴充；第一個模型的實驗必須記錄這些限制，不能藉固定劇本宣稱 AI 駕駛成功。下一步先關閉 CARLA、核實官方 torch CUDA wheel、量測小批次 forward／backward，再凍結訓練設定；test 不用於調參。

## D013 — Phase 3 先建立可稽核的 scratch segmentation 基準

固定官方 torch 2.10.0+cu128，不聲稱最新。保留 NumPy 2.3.5 / Pillow 12.0.0、Python 3.12.14 與同一 venv，pip check 通過；CARLA 關閉。原解析度 Tiny U-Net 能保留極少的兩輪車／車道線像素，先檢驗工程與資料訊號，後續再比較預訓練 backbone，不以模型大小推斷品質。

只學七類 focus + other_annotated，raw 0 ignore。其他類別合併的資訊損失明確記在 model card；大型車／交通燈／臨時障礙物的完整 Phase 3 目標沒有刪除。當前只有六個短場景，無法作可靠泛化驗收；supported-focus macro IoU 與逐類 recall 必須並列，不用背景 pixel accuracy 冒充成功。

嚴格 deterministic 的 CUDA CE reduction 在 GPU 測試失敗後，依官方 v2.10.0 源碼改為 per-pixel loss + tensor reduction，保持原 weighted-mean 定義；等價與 ignore 梯度測試通過。segmentation-v1-001 的完整解析度 overfit 又攔下 AMP 梯度溢位，失敗原樣保留；v1.1 加入 bounded GradScaler backoff / same-batch retry，人工 Inf 測試確認跳過壞更新，成功 retry 只推進一次 Adam step。未為通過檢查而關閉 finite / deterministic 保護。

正式設定、資料／程式／runtime 指紋及 test-once 政策在訓練前凍結。每 epoch checkpoint 可在相同環境精確恢復 RNG 與下一步更新；只宣稱已測的 epoch 邊界恢復，不宣稱任意中斷點或跨 GPU 版本 bitwise 一致。參見 PHASE3.md 與 configs/segmentation_v1_1.json。
