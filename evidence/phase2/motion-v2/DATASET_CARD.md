# Town01 Motion Dataset v2

狀態：六場景收集、完整磁碟稽核與訓練格式匯出均 passed；見 audit.json、training/report.json。Phase 2 v2 基礎資料版本驗收通過，可供下一步小型感知實驗。

來源：本地修正 CARLA 0.9.16、DX12／Epic，八方位 640×360、90° FOV。RGB inputs 與 GT labels 分存。自車為 BasicAgent 規則、車／自行車／機車為 Traffic Manager、兩名行人為固定人行道路徑控制；都不是已訓練 AI。

規模：六個空間 scene groups、每案 200 個 0.05 秒 tick（10 秒），每四 tick 保存一組，共 300 組／7200 PNG。相機原始 20Hz，存檔 5Hz；不宣稱牆鐘實時 5Hz，stride 省磁碟但不降低所有 sensor render 成本。選取影格由下一 world tick 回傳後，嚴格配對原 snapshot frame／timestamp，label timestamp 跨度約 9.8s。

切分：train 三場景 150 組／1200 個 view；validation 一場景 50 組／400 view；test 兩場景 100 組／800 view。來源道路群組與規劃路線道路 IDs 不跨 split。共享 Town01，無 unseen-town 或零視覺內容重疊宣稱。

每案角色：car、bicycle、motorcycle 各一與 pedestrian 二，六場景共 30 次 actor 實例、1500 個逐幀 3D box GT。3D boxes 包含視野外／被遮擋角色，尚未標記逐相機可見度。語意標籤是 PNG 紅通道原始 CARLA 0..28 ID；depth 是原始 24-bit packed RGB，不是灰階。

凍結驗收：每案 ego 位移至少 5m、標籤跨度至少 9.5s、每 actor 位移至少 1m、ego collision 0；每 split 的 [road=1, sidewalk=2, pedestrian=12, car=14, motorcycle=18, bicycle=19, roadline=24] 各至少 1000 像素／10 張正樣本，並驗證 PNG CRC／尺寸／逐圖 hash、frame／timestamp cadence、K/T 數值與參與者 identity。

訓練介面：SegmentationSamples 為已 audit 資料提供 RGB uint8 HWC 與 target uint8 HW；不內建模型、類別重映射、增強或歸一化。load_inference_rgb 只讀 RGB，不讀 labels。training/*.jsonl 保留整 scene 分組，phase3 需固定模型實驗後才訓練。

限制：只有六個短 episode、固定演出與有限藍圖，類別／天氣／場景不平衡，長尾大型車不足；夜雨有低照度、雨霧與模糊。不是閉環策略安全驗證或完整三維感知；沒有模型權重或準確率。舊 pilot 的密度 0／3／6 資料保留作管線對照，沒有在此版本冒充更多獨立群組。

失敗紀錄：暖機前以 client 的 (0,0,0) 初始化導航曾造成碰撞；單車診斷確證後修正，同場景與正式六案通過，原失敗不回寫或刪除。

實測稽核：7200 PNG／1,231,370,833 bytes、CRC 錯誤 0、時間戳誤差 0、群組交集 0；觀測跨 split ego 最小距離 147.36m。各 required class 的最少正樣本數為 86，均超過 10；所有角色最小位移 7.56m。訓練清單 1200／400／800 view，96 個首尾樣本用 Pillow 獨立解碼通過。19 項測試通過。
