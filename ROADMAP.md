# 路線圖

依原規格順序推進；目前沒有刪除任何長期目標。每次只執行一個可驗收任務，前一階段未通過時不得宣稱下一階段完成。

| 階段 | 最小任務與前置依賴 | 驗收依據 | 狀態 |
|---|---|---|---|
| 0 | 硬體 / 相容 Python / CARLA server / 單車與單鏡頭 | 環境報告 + 固定加速煞車 protocol | v2 通過：本地 native 排程修正 + 原始影格配對，原 v1 失敗保留 |
| 1 | Phase 0 通過後，固定 Town01 起終點、規則 Baseline、事件記錄 | 5 個預先固定 scenario；抵達 / 碰撞 / 偏離 / 時間；號誌未刺激 | 初始 Baseline 5/5 通過，DX12 最終設定回歸通過；非 AI |
| 2 | 8 RGB / calibration / 同幀 queue / 標籤隔離與場景切分 | 完整同步、無 split 群組交集；資料分佈稽核 | v2 基礎資料驗收通過：六個移動自車場景、300 組／7200 PNG、行人／兩輪車、train format；仍是有限單城研究版本 |
| 3 | 先小規模 2D 偵測 / 分割與 CUDA 試跑，再自訓或微調 | 保留測試集逐類 AP / precision / recall / IoU + 錯誤樣本 | 進行中：首輪 segmentation 原test26.22%；已完成可見instance政策、靜態/動態32 sensors、640 PNG與44 tests；配對夜雨診斷仍嚴重失敗。完整v3資料 / detector / 品質驗收未完成 |
| 4 | 多鏡頭幾何、BEV、跨鏡頭與時序融合 | 同物件去重 / 位置與道路誤差；單鏡頭消融 | 未開始 |
| 5 | 3D boxes → occupancy → 進階 geometry | 位置 / 尺寸 / 朝向 / semantic occupancy IoU | 未開始 |
| 6 | WorldFrame 推論輸出 → 3D 導航顯示 | 來源檢查、追蹤 / 俯視 / 自由視角、延遲與更新率 | 未開始 |
| 7 | 跨時追蹤、速度、運動預測 | ID switches / ADE / FDE / 風險情境 | 未開始 |
| 8 | global / local / behavior planning，先規則再學習 | 未見路線完成率、碰撞率、違規率、效率 | 未開始 |
| 9 | 感測 → 推論 → 規劃 → 控制閉環 | 起終點 / 動態交通 / 故障停止 / 診斷紀錄 | 未開始 |
| 10 | 固定 regression、未見場景、消融與再訓練 | 同條件模型比較、研究報告與展示影片 | 未開始 |

## 門檻凍結規則

Phase 0 的數值門檻在首次實機測試前保存為 `configs/phase0_acceptance.json`，每次 run 複製並記錄 SHA256。
Phase 1：5 個固定晴天無交通短路線，完成率 100%、碰撞 0、偏離事件 0、每路線上限 120 模擬秒；scenario IDs、路線點與規則定義在執行前固定，已通過。
Phase 2 v2：dataset_motion_criteria_v2.json 與 dataset_motion_scenes_v2.json 固定十秒移動序列、行人／兩輪車、自車零碰撞、每 split 類別像素／正樣本門檻；六案收集、獨立 audit 與訓練格式通過。
Phase 2 pilot：dataset_pilot_criteria_v1.json 與 dataset_scenes_v1.json 在收集前凍結；只驗證小型資料管線，不用來設定模型品質門檻。
Phase 3 之後的品質門檻尚未凍結，不假裝已有適合本資料集的數值；可先作探索性 Baseline，另用未參與調參的測試集正式驗收。
探索結果不能同時用來設定門檻與宣告同一測試通過。每個階段啟動前，應先補齊其固定輸入、版本、門檻與可重現命令。

## 進入下一階段

Phase 0 必須同時有 doctor=`passed` 與實際 smoke=`passed`，只有單元測試通過不算。
尚未產生 checkpoint；目前沒有自訓、微調或直接使用的預訓練模型。
