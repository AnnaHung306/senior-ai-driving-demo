# 目前任務

更新：2026-10-16（Asia/Taipei）。原始完整長期需求見 PROJECT_SPEC.md。

目前：Phase 0 與 Phase 1 初始規則 Baseline 已通過；Phase 2 的 v2 基礎資料版本也已完成收集、獨立磁碟稽核與訓練格式驗收。下一步是 Phase 3 的 CUDA／小型感知實驗，尚未安裝 PyTorch、訓練模型或產生 checkpoint。

## 本輪成果

- 資料：data/town01-motion-v2-001，六個十秒場景、八方位 RGB／語意／深度、300 組／7200 PNG（約 1.23GB）。自車 BasicAgent、車輛 Traffic Manager、行人固定人行道路徑；全部為資料生成規則，不是 AI。
- 每場景有汽車、自行車、機車各一與行人二；300 筆 frame 共有 1500 個 3D box GT。每個 participant 位移超過 1m，實際最小約 7.56m。
- 自車位移 15.84–20.49m，各場景 ego collision callbacks 均 0，label timestamp 跨度約 9.8s。
- 20Hz world／原始 sensors，每四 tick 保存一組（5Hz），嚴格一 tick readback 並配回原 snapshot；不宣稱牆鐘實時 5Hz。stride 只省磁碟，仍有每 tick render 成本。
- 稽核：7200 PNG 全部 CRC／尺寸／解壓／逐圖 SHA256、frame／timestamp cadence、角色 identity、K/T 數值與群組 split 通過；時間戳誤差 0，scene group 交集 0，跨 split 觀測 ego 最小距離 147.36m。
- 各 split 的道路、人行道、行人、汽車、機車、自行車、車道線，都滿足事先固定的至少 1000 像素／10 張正樣本。最少正樣本數實際為 86。
- train／validation／test=150／50／100 組，展開為 1200／400／800 view。SegmentationSamples 與 training/*.jsonl 已就緒，96 個首尾樣本以獨立 Pillow decoder 驗證；RGB-only inference API 不讀 GT。
- 19 項單元測試通過。精簡證據及 DATASET_CARD 在 evidence/phase2/motion-v2；原始資料、失敗 run 與 logs 不進 Git。

## 已完成的前置階段

Phase 0：本地修正 CARLA 0.9.16、DX12／Epic、Python 3.12.14；原 140 幀／14 PNG／加速煞停門檻通過，同 server 初次 load 與再次 reuse 皆通過。
Phase 1：五條預先固定的 40m 晴天無 NPC、非路口路線 5/5，碰撞／跨線／off-driving-lane 均 0；號誌沒有受測，不能稱為 AI 駕駛。
原停駐 Phase 2 pilot 保留在 data/town01-pilot-v1-epic，另有 Low 畫質對照與失敗資料，不回寫歷史。

## 不要重犯

1. Town01 已載入時沿用地圖；不要冗餘 reload 或只因車輛 tick 時 crash 就推斷物理引擎故障。
2. 官方 PixelReader 把 priority bit 當 thread enum，誤排 StatsThread。當前 background build SHA256=b0370d6dcd57b95299d12005cebad1dc6187804e83aeb51a184ccbc4b581fea8，固定 build guard、stock 備份、source patch 與 restore／reapply 驗證齊全。
3. DX11 單鏡頭成功不代表八鏡頭可行；已有 GPUCrash／device lost 證據。當前用 DX12／Epic，Low 雨天的黑天空／雜訊失敗保留。
4. NPC 清理先取消 autopilot，world restore 前保持 TM 同步，最後才關 TM 同步；避免既有 native abort。
5. 新生成 actor 在首次 tick 前，client get_location 實測為 (0,0,0)，snapshot 尚無該 actor。導航必須在暖機 snapshot 後初始化。首個完整動態 probe 曾有 10 collision callbacks；修正後同場景及正式六案通過，原失敗保留。
6. GT 只在 labels／訓練／評估，推論只讀 RGB。不得用腳本動作或 GT boxes 冒充 AI。

## 下一個單一任務

Phase 3 最小感知實驗：先核實官方 PyTorch CUDA wheel 與 Python 3.12／RTX 4070 相容，在 CARLA 關閉時完成小批次 forward／backward 及峰值 VRAM 實測；再凍結小型 2D segmentation 模型、類別映射、seed／hyperparameters／資料與 split hashes、validation 選模方式與基準比較，執行一次可追溯的自訓或微調實驗。只用 train 更新、validation 選模，test 不參與調參；逐類 IoU／precision／recall 與失敗樣本必須保留。

資料限制：六個短 episode、共享 Town01、固定演出、有限車型、長尾大型車不足；群組不交集不等於零視覺內容重疊，也不是跨城鎮泛化證據。這是 Phase 2 基礎研究版本驗收，並非完整資料覆蓋或感知品質通過。

模擬器與集中資料程序均已關閉；沒有模型訓練、監控排程或其他背景工作。沒有 BEV／occupancy、3D 導航或完整 AI 閉環。
