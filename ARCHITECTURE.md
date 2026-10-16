# 系統架構與介面

狀態：2026-10-16，Phase 0、初始規則 Baseline 與八鏡頭多模態資料 pilot 已實作及實測；AI、融合、3D 與完整閉環仍為設計。
專案僅限模擬研究。第一個 MVP 是 CARLA 的加速、煞停與同步 RGB 擷取；第一個研究 MVP 是固定路線 Baseline + 八鏡頭資料集 + 自行微調的 2D 感知模型。

## 模組邊界

| 模組 | 輸入 → 輸出 | 目前狀態 |
|---|---|---|
| A Simulation | scenario / seed → CARLA world、tick | v2 實機驗收通過；native PixelReader 修正與 frame pipeline 見 D005 |
| B Sensor | camera rig / tick → RGB、frame、timestamp、K、T | 八鏡頭 RGB 同步通過；8 方位 × RGB／語意／深度共 24 感測器 pilot 通過 |
| C Perception | RGB / calibration → 2D detections、semantic logits | 未實作、未訓練 |
| D Spatial Fusion | 多鏡頭特徵 / calibration / ego motion → BEV features | 未實作 |
| E World Model | BEV / temporal features → objects、lanes、occupancy | 未實作 |
| F Prediction | 追蹤歷史 → future trajectories / uncertainty | 未實作 |
| G Planning | navigation goal / map / inferred world → trajectory / behavior | CARLA BasicAgent + simulator GT 固定路線 Baseline；AI 規劃未實作 |
| H Control | trajectory / ego kinematics → throttle、brake、steer | smoke 與 BasicAgent 路線控制通過；未接感知模型 |
| I Visualization | inference WorldFrame → 3D view | 未實作 |
| J Evaluation | inference、獨立 GT → metrics / failures | Phase 0／規則路線／多鏡頭及磁碟稽核已實作；感知指標未實作 |
| K Data / Training | training split / labels → checkpoint / model card | 群組 split、RGB-only 輸入邊界與 GT labels 已實作；訓練未實作 |

選擇模組化離線收集、離線訓練、同步推論閉環，避免讓 LLM 逐幀參與。

## 座標與資料協定 v0（後續實作必須驗證）

- 原始資料使用 CARLA 左手座標：x 前、y 右、z 上，公尺、秒、角度明確註記。
- 相機 optical 座標：x 右、y 下、z 前。CARLA camera 向量 `[x,y,z]` 轉成 `[y,-z,x]`。
- `K` 為 3×3，相機安裝 `T_ego_from_camera_local` 為 4×4；使用列向量，禁止混用轉換方向。
- 後續顯示層統一在 adapter 轉換成 Three.js 座標，不在模型中隱式轉換。
- `SensorFrame`：schema_version、run_id、scenario_id、frame_id、timestamp_s、8 個 RGB 路徑、K、T、ego kinematics。不得包含其他 actor GT、深度 GT 或語意 GT。
- `LabelFrame`：同 frame_id 對應語意、深度、3D boxes、道路標籤，僅供 train / evaluator。
- `WorldFrame`：schema_version、frame_id、source=`inference`、model_version、dataset_version、latency_ms、ego、objects（track_id / class / confidence / center_m / size_m / yaw_rad / velocity_mps / covariance）、lanes、occupancy、predictions。
- 缺少推論、過期、低信心或座標不合法時，規劃器應輸出安全停止。不可用 GT 或假物件補洞。
- 此介面是設計草案，尚無宣稱已完成的推論服務或 3D 畫面。

## 檔案與真值隔離

`data/<version>/sensors/` 與 `calibration/`：模型推論可讀；`labels/`：訓練與評估可讀，正式推論不可讀。
dataset.py 的 read_sensor_frame 僅接受 inputs/ 的固定 RGB schema 與 calibration，拒絕 GT 欄位、labels 路徑、越界路徑和非有限時間戳。單元測試確認此邊界。訓練 loader／模型尚未實作。
同一 scenario / route / seed 的相鄰影格不得跨 train / validation / test。先固定場景群組 split，再建立影格 manifest。
模擬器導航代理只作 Baseline / 教師，必須標記來源，不能稱為自訓 AI。

目前 smoke 的 `evaluation/ego.jsonl` 是自車真值驗收；`sensors/front/` 是真實模擬 RGB 輸出。
`sensor_manifest.json` 連接 frame 與 image，`calibration.json` 保存內外參數；不產生周圍物件辨識。

## 執行與資源

CARLA 0.9.16 Windows 預編譯包、同版本 Python client、Python 3.12 venv。
目前採 DX12、Epic、640×360 八方位。歷史 Low RGB rig 約 4.34 組／牆鐘秒，不能當作 Epic 或 AI 的吞吐量；資源取樣與畫質對照限制見 EVALUATION。
CARLA 與訓練分時執行，未經實測不保證 8GB GPU 能同時負擔多鏡頭模型。
初始 2D 模型候選為 torchvision Faster R-CNN / DeepLabV3 的小批次微調；須先驗證 PyTorch CUDA 相容性與單批次峰值顯存，再鎖定版本。
BEVDet / BEVFormer 為後續研究候選，不在目前環境直接安裝舊版 OpenMMLab 依賴。
