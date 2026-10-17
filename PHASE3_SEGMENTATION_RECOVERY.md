# Phase 3 分割修復：先補訓練場景覆蓋，再驗證

此檔記錄下一個實驗的執行依據。沒有新資料採集或新分割訓練已完成的宣稱；v4的失敗報告、原資料007及全部驗收門檻保留。

## 已實測的問題

- Mobile v4完成8輪，平均mIoU .50846達標，但wall .06195、fence .00728、margin F1 .40733未過；不能只憑總平均驗收。
- 全9600張train的wall/fence/terrain原始語意像素總數與凍結訓練統計完全相同，未發現這三類的轉換計數錯誤。資料見 `evidence/phase3/sunny-v3/full-train-material-survey-001/`，含壓縮逐張紀錄與SHA。
- Fence支持≥512px的train views只在Town01四原groups（526/455/839/767），Town03四groups均0。這是像素支持統計，不代表四個獨立材質族群。
- 實際看過17個按支持量/暖色/中性亮色選出的景物範例：train代表多混凝土與金屬網，val代表有木板與白色柵欄。自動掃描全train，但沒有人工式看過全部9600張，不聲稱所有木柵欄絕對缺失。
- 固定4例CPU probe：train隧道wall .9861、混凝土Fence .9067；val分隔道路wall .5151、木柵欄 .00758。牆面改善不是全面失敗；圍欄外觀/幾何覆蓋差異是值得優先驗證的假說。
- train/val有效pixel占比：wall8.719%/.247%、sidewalk13.648%/3.854%、terrain4.362%/16.437%。草地判sidewalk及錯誤margin有實際圖例；單純增加wall/fence權重可能增加背景誤報，不能直接當解法。

## 下一段：只查原訓練城鎮的原生景物

先讓現有GPU訓練/評估退出，再啟CARLA。`start_carla.ps1`已加入忙碌拒啟；不要為了快速查資料停掉目前detector，也不要與它並跑server。

`scout_training_scenery.py`只查已載入Town01/Town03的原生Fences/Walls環境物件與道路spawn points，綁原plan/audit/OpenDrive SHA，不load map、不tick、不改天氣/設定、不生actor或sensor。物件名稱只是尋找wood/picket等資產的提示，不是材質GT；bbox按原生值保存，尚不把它當作經驗證的相機規畫座標。

```powershell
# GPU程序全部結束後才執行，CARLA/RPC須在native host可達的權限環境。
.\scripts\start_carla.ps1
# 用獨立短client準備Town01；若已是Town01就不重載。
.venv/Scripts/python.exe scripts/scout_training_scenery.py --root data/sunny-cross-town-v3-007 --town Town01 --output artifacts/runs/train-scenery-scout-town01-001
# 另一個獨立client準備Town03，再呼叫新scout版本；不要在scout內切map。
.venv/Scripts/python.exe scripts/scout_training_scenery.py --root data/sunny-cross-town-v3-007 --town Town03 --output artifacts/runs/train-scenery-scout-town03-001
.\scripts\stop_carla.ps1
```

目前僅完成CLI與2項offline回歸，**沒有真正連server完成scout**。Town02/04/05在CLI與函式內都拒絕用作training scout。若原train城鎮找不到所需外觀，保存清單與實看證據，再另凍結未使用城鎮的train擴充計畫；不可挪用validation/development/lockbox城鎮或把它們重新命名成train。

## 補景資料的凍結條件

原主007保持immutable。補景另建 **segmentation-only / train-only** 版本與獨立audit/review，不冒充新的30秒動態駕駛episode，也不自動供detector、phase或導航訓練。模型provenance需同時綁main與supplement的指紋、實際views/groups與採樣權重，不能只記main卻漏掉額外資料。

1. 先以原生inventory與實際RGB/semantic預覽確認可用外觀，獨立標註wood panel、white picket、metal mesh、concrete barrier等「視覺外觀」；不要僅由mesh名稱或RGB暖色自動判材質。每個想補的外觀盡量至少2個不同原生地點，保存空間位置，重複frame/角度不當成不同地點。
2. 針對可用地點，先凍結相機姿態與採樣表再正式採集。沿用640×360、90°、車載高度約1.7m與八方向rig；先驗證距離/角度不同時的真實可見像素，不憑空填世界座標。道路、草地、sidewalk與護欄的背景negative也一起保留。
3. 新collector必須實測sensor屬性、世界pose、frame/timestamp；維持F snapshot後F+1 readback規則，depth按已驗證axial解碼。RGB、raw semantic、packed depth、instance均CRC/SHA檢查；類別映射沿用17類與ignore255，不能為補quota將Dynamic自行改成Fence。
4. 靜態補景若用虛擬rig anchor，明確記為虛擬座標架，不捏造ego位移、collision-free路線或獨立動態群組。這些frame只供二維segmentation，評分仍用原完整val與未開封heldout。
5. 先獨立audit，再實際看材質/薄柵欄/草路交界、遠近與遮擋的RGB/GT panels，另寫綁audit及實看圖SHA的review。沒有真實看圖、資料不足或geometry未確定時，不批准訓練。

## 新實驗與驗收順序

- 先測資料覆蓋改善，不同時更換架構、loss與分類映射。新run、新config，記明與v4的總updates/epochs差異；新增資料意味相同8epochs也不等於相同最佳化步數，不能聲稱嚴格因果比較。
- 主資料與supplement採樣需保留原動態物件/稀少類支持，不讓大量靜態牆面補景淹沒行人、兩輪車、號誌與交通桶。採樣策略與class weights只從train決定，val只用選模與查錯，不回灌train。
- 小型overfit/finite-gradient/容量檢查後重置模型正式訓練。仍需17類逐類IoU、margin F1、逐scene錯例、背景誤報；不降低每類.2或margin .5門檻。
- 若新資料確實改善覆蓋而val仍失敗，再單獨試hard-negative/邊界損失等方法；不能把還沒測的loss當作已解決。新模型val全部合格後才一次heldout，錯例、RGB-only實推論、模型/輸入SHA綁定benchmark及最終導出仍必須完成。

號誌/偵測可獨立繼續；號誌v1-002的GT crop通過不等於joint通過。全系統Phase3仍需三任務各自完整驗收，不能讓補景計畫或工具完成取代模型成績。
