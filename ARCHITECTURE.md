# 系統架構與介面

狀態：2026-10-16，僅 Phase 0 的環境檢查與單鏡頭控制驗收已實作；其餘為設計。
專案僅限模擬研究。第一個 MVP 是 CARLA 的加速、煞停與同步 RGB 擷取；第一個研究 MVP 是固定路線 Baseline + 八鏡頭資料集 + 自行微調的 2D 感知模型。

## 模組邊界

| 模組 | 輸入 → 輸出 | 目前狀態 |
|---|---|---|
| A Simulation | scenario / seed → CARLA world、tick | Phase 0 驗收程式可建立固定晴天場景；未驗收 |
| B Sensor | camera rig / tick → RGB、frame、timestamp、K、T | 單鏡頭已實作；八鏡頭未實作 |
| C Perception | RGB / calibration → 2D detections、semantic logits | 未實作、未訓練 |
| D Spatial Fusion | 多鏡頭特徵 / calibration / ego motion → BEV features | 未實作 |
| E World Model | BEV / temporal features → objects、lanes、occupancy | 未實作 |
| F Prediction | 追蹤歷史 → future trajectories / uncertainty | 未實作 |
| G Planning | navigation goal / map / inferred world → trajectory / behavior | 未實作 |
| H Control | trajectory / ego kinematics → throttle、brake、steer | 目前只有開迴路驗收；非導航控制器 |
| I Visualization | inference WorldFrame → 3D view | 未實作 |
| J Evaluation | inference、獨立 GT → metrics / failures | Phase 0 驗收已實作；感知指標未實作 |
| K Data / Training | training split / labels → checkpoint / model card | 未實作 |

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
未來 dataset loader 須有分離的 inference / training API，並以測試確認推論輸入沒有 GT。
同一 scenario / route / seed 的相鄰影格不得跨 train / validation / test。先固定場景群組 split，再建立影格 manifest。
模擬器導航代理只作 Baseline / 教師，必須標記來源，不能稱為自訓 AI。

目前 smoke 的 `evaluation/ego.jsonl` 是自車真值驗收；`sensors/front/` 是真實模擬 RGB 輸出。
`sensor_manifest.json` 連接 frame 與 image，`calibration.json` 保存內外參數；不產生周圍物件辨識。

## 執行與資源

CARLA 0.9.16 Windows 預編譯包、同版本 Python client、Python 3.12 venv。
模擬先採 Low、640×360 單鏡頭；通過後測量八鏡頭 VRAM / 延遲。
CARLA 與訓練分時執行，未經實測不保證 8GB GPU 能同時負擔多鏡頭模型。
初始 2D 模型候選為 torchvision Faster R-CNN / DeepLabV3 的小批次微調；須先驗證 PyTorch CUDA 相容性與單批次峰值顯存，再鎖定版本。
BEVDet / BEVFormer 為後續研究候選，不在目前環境直接安裝舊版 OpenMMLab 依賴。
