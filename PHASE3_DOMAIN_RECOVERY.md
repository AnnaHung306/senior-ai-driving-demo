<!-- 最新實際進度優先於下方歷史準備紀錄。 -->
## 2026-10-18 實際進度

v6分割六輪已完成，原Town01/03補景與新Town06/07各1024視圖、各重複2次；新資料完整audit、四頁32情境視覺批准及實際訓練loader核對均通過。mIoU .6065211771、Fence .6057671054、可見界面F1 .6222117517通過，只有Wall .1564529351未達.2。best SHA 0a3fe3f66fb93be373f2e82218632a4f8cc277bf6067f48bb4683a71f1b2af30；原門檻不變，新heldout仍未評分。detector v6四輪及整批11階段已完成，bicycle／motorcycle／traffic-light Recall不足；phase004 GT crop通過、joint .399064未達.4。原GPU segmentation25／saved detection15／phase31情境共18頁，以及7張bike zoom與8鏡頭demo均實看，完整證據在evidence/phase3/sunny-v3/phase3-recovery-batch-002-completed。CARLA關；新增有界深度A/B尚在跑，來源繼續凍結。

新資料Wall 8,191,123像素、Terrain 31,118,819像素，仍主要是磚石牆與木柵欄／草地／岩面negative，未證實補到實心木Wall或高速路擋土牆。低牆probe亦未補齊此缺口，維持排除train。已用原始semantic raw4/raw5疊圖核對五個validation極值案例：木圍牆與擋土牆確為Wall，不能把Fence重新標Wall。正式plan SHA aff1cc0dc406830104ae50c960f2bc77b9a8d8488ea873e8c43604459e1c44ff；完整來源證據見evidence/phase3/sunny-v3/segmentation-expansion-v2-001-approved。

完成模型的原GPU混淆矩陣逐scene重算：Wall真值2,048,168、預測5,321,592、TP997,032、FP4,324,560、FN1,051,136像素。FP主要真值為other2,150,166、Fence973,196、Sidewalk578,319、Terrain499,916；Town04-00/01/02 Wall IoU .13663/.02127/.20896，不能由整體平均推定每條路線均可用。收據artifacts/reports/segmentation-v6-wall-confusion-diagnostic-001.json核對3600筆與原aggregate完全一致；這是原GPU結果的文件診斷，沒有新推論或材質標籤。

完成v6的Wall-only九候選CPU診斷已跑完3600view：alpha0 Wall .153231至alpha1最高.161307，九候選全未過原門檻；不排正式GPU偏置校正或擴大搜尋。CPU alpha0與GPU原值.156453差異不能當改善。完整負結果在evidence/phase3/sunny-v3/segmentation-v6-wall-only-CPU-study-001。原GPU錯例已實看，木Fence、站牌支柱與岩面仍有Wall誤報，高速擋土牆仍有漏分；後續改法依實際差異，不盲目重複Wall權重或同材質。

完成v6另沿用原先固定28張train-only外觀／negative選例，CPU FP32 Wall IoU .97739、P .98703、R .99011；兩頁六個最高FP/FN案例實看，磚石／水泥Wall大致吻合，局部邊緣仍有誤差。這不是全train accuracy，亦不是獨立或GPU驗收，但與full-val .15645落差仍支持外觀／場景轉移為優先問題。exact選例、工具、收據與原GPU混淆診斷已封存於evidence/phase3/sunny-v3/segmentation-v6-wall-diagnostics-001。

下文保留先前診斷與規劃的時間背景；其中「尚未採集／尚未native驗證」描述準備當時的狀態，以本節及實測收據為準。

使用者要求先參考指定影片及相關文獻後提出改進。已實看關鍵架構／Spatial RNN／Depth-Pose CNN畫面並對照原文，具體方案見[PHASE3_VIDEO_RESEARCH_PLAN.md](PHASE3_VIDEO_RESEARCH_PLAN.md)。64個原train視角／256PNG的depth可行性檢查通過，並非新模型；建議下一個主要方法先以同初始權重、等更新量B0續訓對照B1 depth輔助，GT depth只在train loss，正式仍RGB-only。外觀泛化、輕量context及時序各分支另做，不一次移植整套Tesla／BEV。現有GPU批次與排隊QA的50個source、3個config及4個QA工具已重查零變動。

# Phase3 牆面泛化缺口：資料恢復依據

這是後續實驗的決策依據，並非新資料已採集或模型已通過。已完成但品質未過的 v5 批次、原主資料及品質門檻不變。

第一輪 immutable checkpoint 對事先按 train GT/外觀分層選出的28個已見訓練畫面，CPU FP32 wall IoU .96965、precision .98159、recall .98762。六個最高 FP/FN 的畫面已實看（兩頁），水泥、磚牆大多吻合原GT。這是有偏的選定樣本診斷，不能稱完整train accuracy或獨立泛化估計；但與完整val wall約.08的落差，支持外觀／場景轉移為優先問題，而不是直接增加模型容量或重複既有牆面。

驗證錯例有木質實心圍牆、道路水泥分隔與岩面Wall誤報。原train實看主要磚牆、水泥及隧道；既有補景中的木板Fence不能直接重新標成Wall。依原生semantic保留17類及ignore255，不合併類別、不挪用val/development/lockbox做train。

## 執行次序

1. 完成現有批次及完整val、joint、RGB-only demo/benchmark。完整CPU prior研究的5候選wall只有.07406..07466，原GPU校正候補已取消，不默認重跑。CPU不能代替GPU驗收，這裡只是基於改善幅度調整實驗優先順序；保存全候選，不開heldout。
2. GPU工作退出後才啟CARLA。`configs/segmentation_low_wall_preview_v1_1.json`只是候選預覽表：原128錨點原樣保留，另加Town01/03各24個相隔至少8m的道路pose，按native Walls語意及bbox高度hint取候選。不是材質GT或採集批准。先只用新位置前幾個錨點做probe-only RGB/GT，不能把probe計入正式train。
3. 若原train城鎮仍缺木質實心Wall或岩面negative，檢查伺服器實際可用、尚未劃入任何保留split的城鎮；可優先調查Town06/07（以實際可用地圖為準）。先原生inventory及RGB/GT預覽，保存真實OpenDrive/pose，不憑資產名稱斷言材質。新城鎮只供segmentation train擴充，既有Town04 val、Town05 development、Town02 lockbox及其Opt/同圖別名均排除。
4. 找到可用外觀後才凍結正式新版本：每個希望補齊的外觀至少2個不同原生位置，記位置間距與相關性；原八方向rig、640x360、四模態、ClearNoon、逐frame/time/pose驗證、F+1 readback、CRC/SHA、ignore規則均沿用。保留岩面、建築、植被、草路邊界negative，不能只收正例牆面。
5. 新資料另建root及audit/review；不得改原86400PNG或假稱新動態路線。既有T1/T3補景的verify契約不放寬；新增城鎮用明確新版本驗證器，從原plan派生保留城鎮與地圖hash排除表，捕捉／審核／實看批准後才允許新版訓練載入。

## 額外地圖與 v2 工具準備（尚未產生新訓練資料）

官方0.9.16額外地圖包已完整下載並inspect：7250055782 bytes，SHA709a5cb4d870668519dc27f0e0104e0419cd76adb75da2c232d74ddcc311423f，22592檔／23647 entries，展開19116493167 bytes。官方release連結及文件在additional-maps-preparation-001；未取得獨立publisher checksum。實際有Town06/Town07完整與Opt maps，但尚未用native server確認可載入。

共有11744路徑與原模擬器重疊，其中2686僅檔案大小就不同，還包含517個Engine／plugin內容檔。故**不覆蓋原 artifacts/carla**。prepare_additional_map_runtime.py 在新artifacts/carla-phase3-expansion-v1複製原分發檔（排除Saved輸出）並只在新副本覆蓋已凍結官方包的內容資產；最終重驗全新副本及全部原來源，server binary必須exact。新source/config/archive/baseline完整契約與逐檔恢復；第一個001因content-only守衛拒絕Engine/plugin，未建立副本，失敗保留；002已先逐一驗證全部真22592 paths後再執行。**仍在準備，未native驗證、未啟CARLA**。不得因state檔存在當可用；等completed_content_not_native_validated、source資料全驗後，原GPU所有工作退出，才啟新root的DX12/Epic。

新增prepare_unused_scenery.py、collect_unused_scenery.py、audit_unused_scenery.py是明確v2流程，原collector/auditor/loader保持原Town01/03限制。plan排除所有原split map家族和OpenDrive hash；只Town06/07，rig/ClearNoon/模態/timing/pose/GT規則不變，native name與高度仍只hint。capture另綁新runtime manifests與launch root／binary／DX12/Epic收據，原native metadata與runtime provenance原bytes收入資料root，日後訓練不需live CARLA。probe-only仍拒絕train audit。新Town native inventory、實際RGB/GT、正式plan、full audit、視覺批准與v2 training loader**都還沒完成**；這些工具不是已訓練資料。

139項CPU整批回歸（136pass／3GPU skipped）已通過，其中4項isolated-runtime和3項native anchor選擇測試；原94項GPU preflight才是已執行GPU證據。後續v2 plan額外rig equality小守衛需在正式使用前隨focused／最後統一檢查再核對；不把CPU工具通過當模型品質。

## 訓練與驗收

重點是新增外觀及誤報negative，不預設要加高wall權重。先實測新資料支持量及取樣比例，再凍結新config；class weights只由train決定，主動態train維持約70% optimization samples，保留人、rider、兩輪車、號誌與交通桶的支持。新checkpoint須綁原main＋每個supplement及取樣倍率、initializer SHA、source/config/runtime，不能只記舊main。

先做finite/overfit/容量檢查，重置後正式訓練。仍要求16focus mIoU≥.5、每類IoU≥.2、margin F1≥.5；模型選擇按原qualified-first規則。detector逐類P/R及phase actual-box joint也須通過，才一次heldout。新資料和程式完成不代表Phase3完成。

若新增外觀後仍失敗，先比較seen train與val、逐scene FP/FN及標籤外觀一致性，再決定hard-negative/crop/損失或架構實驗。一次只提出有診斷依據的主要變更，完整記錄額外updates，不宣稱未做的matched-compute因果比較。
