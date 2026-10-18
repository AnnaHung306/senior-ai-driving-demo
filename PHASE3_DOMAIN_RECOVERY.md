# Phase3 牆面泛化缺口：資料恢復依據

這是後續實驗的決策依據，並非新資料已採集或模型已通過。正在執行的 v5 批次、原主資料及品質門檻不變。

第一輪 immutable checkpoint 對事先按 train GT/外觀分層選出的28個已見訓練畫面，CPU FP32 wall IoU .96965、precision .98159、recall .98762。六個最高 FP/FN 的畫面已實看（兩頁），水泥、磚牆大多吻合原GT。這是有偏的選定樣本診斷，不能稱完整train accuracy或獨立泛化估計；但與完整val wall約.08的落差，支持外觀／場景轉移為優先問題，而不是直接增加模型容量或重複既有牆面。

驗證錯例有木質實心圍牆、道路水泥分隔與岩面Wall誤報。原train實看主要磚牆、水泥及隧道；既有補景中的木板Fence不能直接重新標成Wall。依原生semantic保留17類及ignore255，不合併類別、不挪用val/development/lockbox做train。

## 執行次序

1. 完成現有批次及完整val、joint、RGB-only demo/benchmark。CPU prior研究完成後仍不能代替GPU；預先固定的bias校正只作一次獨立補充實驗。若依然失敗，保存全候選，不開heldout。
2. GPU工作退出後才啟CARLA。`configs/segmentation_low_wall_preview_v1_1.json`只是候選預覽表：原128錨點原樣保留，另加Town01/03各24個相隔至少8m的道路pose，按native Walls語意及bbox高度hint取候選。不是材質GT或採集批准。先只用新位置前幾個錨點做probe-only RGB/GT，不能把probe計入正式train。
3. 若原train城鎮仍缺木質實心Wall或岩面negative，檢查伺服器實際可用、尚未劃入任何保留split的城鎮；可優先調查Town06/07（以實際可用地圖為準）。先原生inventory及RGB/GT預覽，保存真實OpenDrive/pose，不憑資產名稱斷言材質。新城鎮只供segmentation train擴充，既有Town04 val、Town05 development、Town02 lockbox及其Opt/同圖別名均排除。
4. 找到可用外觀後才凍結正式新版本：每個希望補齊的外觀至少2個不同原生位置，記位置間距與相關性；原八方向rig、640x360、四模態、ClearNoon、逐frame/time/pose驗證、F+1 readback、CRC/SHA、ignore規則均沿用。保留岩面、建築、植被、草路邊界negative，不能只收正例牆面。
5. 新資料另建root及audit/review；不得改原86400PNG或假稱新動態路線。既有T1/T3補景的verify契約不放寬；新增城鎮用明確新版本驗證器，從原plan派生保留城鎮與地圖hash排除表，捕捉／審核／實看批准後才允許新版訓練載入。

## 訓練與驗收

重點是新增外觀及誤報negative，不預設要加高wall權重。先實測新資料支持量及取樣比例，再凍結新config；class weights只由train決定，主動態train維持約70% optimization samples，保留人、rider、兩輪車、號誌與交通桶的支持。新checkpoint須綁原main＋每個supplement及取樣倍率、initializer SHA、source/config/runtime，不能只記舊main。

先做finite/overfit/容量檢查，重置後正式訓練。仍要求16focus mIoU≥.5、每類IoU≥.2、margin F1≥.5；模型選擇按原qualified-first規則。detector逐類P/R及phase actual-box joint也須通過，才一次heldout。新資料和程式完成不代表Phase3完成。

若新增外觀後仍失敗，先比較seen train與val、逐scene FP/FN及標籤外觀一致性，再決定hard-negative/crop/損失或架構實驗。一次只提出有診斷依據的主要變更，完整記錄額外updates，不宣稱未做的matched-compute因果比較。
