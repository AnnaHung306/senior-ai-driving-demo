# 路線圖

依原規格推進，沒有刪除長期目標。2026-10-21依使用者要求完成Phase 8–9導航與真閉環工程研究版；分列工程、品質與即時整合狀態，不宣稱正式階段全部通過。Phase4–7列保留當時驗證範圍，最新導航及閉環進展以Phase8–9列及PHASE89.md為準。

| 階段 | 最小任務與前置依賴 | 驗收依據 | 狀態 |
|---|---|---|---|
| 0 | 硬體 / 相容 Python / CARLA server / 單車與單鏡頭 | 環境報告 + 固定加速煞車 protocol | v2 通過：本地 native 排程修正 + 原始影格配對，原 v1 失敗保留 |
| 1 | Phase 0 通過後，固定 Town01 起終點、規則 Baseline、事件記錄 | 5 個預先固定 scenario；抵達 / 碰撞 / 偏離 / 時間；號誌未刺激 | 初始 Baseline 5/5 通過，DX12 最終設定回歸通過；非 AI |
| 2 | 8 RGB / calibration / 同幀 queue / 標籤隔離與場景切分 | 完整同步、無 split 群組交集；資料分佈稽核 | v2 基礎資料驗收通過：六個移動自車場景、300 組／7200 PNG、行人／兩輪車、train format；仍是有限單城研究版本 |
| 3 | 2D偵測／分割／號誌與資料管線 | 原逐類 AP/P/R/IoU、錯例、模型卡與來源保留 | 2026-10-18依使用者決定收尾，不追Wall、不納入depth；原品質未達項目保留，後續限定晴天／無近牆循線 |
| 4 | 多鏡頭幾何、BEV、跨鏡頭與時序融合 | 同物件去重 / 位置與道路誤差；單鏡頭消融 | 進行中：幾何 BEV 原型及 32 組 GPU／獨立 GT／視覺檢查完成，193 測試通過；時序道路 IoU 95.20%、線 F1 90.27%，候選軌跡 RMSE 15.9cm；新完整路線與效能／物件可靠度仍待驗收，見 PHASE4.md |
| 5 | 3D boxes → occupancy → 進階 geometry | 位置 / 尺寸 / 朝向 / semantic occupancy IoU | 工程研究版與再驗證完成：891裁切／targets、232 CUDA reload、32WorldFrames來源一致；定位P61.11/R55.93、3D IoU.25 R25.42%，品質與即時仍未過，詳PHASE5/6.md |
| 6 | WorldFrame 推論輸出 → 3D 導航顯示 | 來源檢查、三視角、延遲與更新率 | 本機互動研究版完成：32影格／147物件／256RGB、235Python＋20JS、真browser與故障恢復通過；render60FPS／CPU套用p9541.1ms；CARLA連續推論、新key串流、目的地導航、路口拓樸與速度仍待後續整合，非完整正式驗收 |
| 7 | 跨時追蹤、速度、運動預測 | ID switches / ADE / FDE / 風險情境 | 工程研究展示完成：240組RGB／1091物件／889預測、256Python＋40JS＋22HTTP及真browser核對；已知val速度MAE.290m/s、ADE1.950m、FDE2 1.205m、ID2與舊同。2秒覆蓋61.22%／all-GT miss47.52%，風險及上游品質未過；四情境為獨立合成診斷，非新CARLA驗收，詳PHASE7.md |
| 8 | global / local / behavior planning，先規則再學習 | 未見路線完成率、碰撞率、違規率、效率 | 工程研究版完成：地圖全域路線＋RGB道路／物件局部候選＋停等／避讓／抵達行為；3條與007 train自車軌跡幾何分離路線、5個最終具名案抵達／碰撞0／越線0。18次開發嘗試失敗保留；非heldout泛化、無學習式planner、任意路線及密集交通未驗收 |
| 9 | 感測 → 推論 → 規劃 → 控制閉環 | 起終點 / 動態交通 / 故障停止 / 診斷紀錄 | 工程研究版完成：真八RGB CUDA＋自寫轉向／速度控制；網頁自訂11m抵達、紅→綠起步、障礙移除恢復、行人減速、人工／WebGL／相機timeout停止。295Python／48JS／23HTTP及1123來源影格核對；5Hz模擬時間、約1.2s推論／牆鐘慢8–9倍，正式品質false，詳PHASE89／DEPLOYMENT.md |
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
Phase 3 已有自訓分割、偵測與號誌 checkpoint；Phase 4 沿用凍結版本，只新增幾何與時序融合，不新增模型訓練。原 Phase 3 未達數值保留，使用者已授權依晴天／無近牆循線範圍往後推進，詳見 PHASE3_CLOSEOUT.md。
