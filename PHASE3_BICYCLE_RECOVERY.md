# Phase3 自行車缺口：第一輪診斷，正式訓練仍繼續

目前v5只完成第一輪，不能提前視為最終結果。完整GPU val mAP50 .712029／AP50:95 .439683；七類P/R合格，bicycle P .549784、R .348422仍未過.5。Town04-00自行車R .62044但P .46196，Town04-01 R .11163、Town04-02 R .25，場景差異大。剩餘三輪及預先凍結選模繼續。

第一輪checkpoint另存immutable snapshot，SHA b2b69ebac6652c43054148cf132d78fbbb504f4b41af5fa608cd0a2b8a2b8237，不假稱completed run。按每個原train/val group的GT最大可見支持與接近256px bbox面積，固定選出16train＋6val，CPU診斷train16/16找到、val3/6找到，三個val小框全漏。這是刻意選例，不是完整train成績或獨立泛化估計；3頁9contexts已實看。

最初診斷腳本漏帶truncated/visible_pixels，截斷分層被默認為not_truncated；已另存舊工具／known_issue、修正重跑002。所有類的AP/P/R/TP/FP/FN精確一致，只有截斷分層修正：train6個截斷／val3個截斷。三頁PNG bytes亦與已實看舊頁完全相同。正式GPU report與訓練不受此診斷腳本錯誤影響。

後續固定CPU RPN-budget研究：pre/post=(500,150)、(1000,500)、(2000,1000)；六個val target在三配置都存在IoU≥.5候選，最終自行車仍3/6命中。原150配置的所有prediction與前一CPU診斷逐欄完全一致；參數／buffers未變。這只在六個有偏樣本上支持優先查ROI分類／回歸和特徵解析度，不能推論整個split或所有失敗因果，更不能把提高budget當已修復。

## 有界備案：P1 CPU原型，未整合／未訓練

第二輪完整 GPU val 仍只有 bicycle recall 失敗：P .959839、R .327846、小框9/388。舊完整GT診斷顯示train small比例10.95%對val53.22%；因此另備PHASE3_DETECTOR_GEOMETRY.md的尺寸增強，考慮先處理資料尺度落差，再決定是否付出P1算力。所有備案目前均未接入正式訓練。

完整729-target CPU RPN調查已完成（第一輪immutable checkpoint）：原pre/post500/150的IoU50存在覆蓋651/729=.893004，小框352/388=.907216；提高至1000/500或2000/1000，皆686/729=.941015、小框371/388=.956186。原Town04-01覆蓋156/215=.725581，較高budget183/215=.851163，故六選例「都有proposal」不能概括全體。所有learned tensors exact未變，首圖組件baseline逐tensor等於真RPN.forward。這是positive-only獨立IoU存在性，不是一對一AP、正式GPU品質或最終分類／回歸召回；更大budget改善候選覆蓋，仍不能宣稱已修復最終自行車漏檢。完整37chunks／tool／contract／report封存detection-v5-first-full-bicycle-RPN-001。

scripts/experimental_p1_detector.py新增stride2 feature，原P2/P3/P4/P5各層權重明確平移index，原body/RPN/ROI所有參數保持；新增P1 projection/output seeded初始化。六層含pool anchors為8/16/32/64/128/256、仍三aspect channels；ROI涵蓋stride2..32。這是本專案待驗假說，不是新的訓練成果。

4項CPU工程測試通過：全部既有參數依映射exact、原較粗features與pool exact、新stride2 shape、新projection有finite非零梯度及CPU完整train/eval路徑、拒絕重複套用。不代表720 GPU容量或品質已通過。若v5最終自己過關，就不執行這個備案。

只有原batch全部退出後，才可整合版本化constructor/config／production loader，明確初始化新層與保留舊層，記完整source及真正完成parent SHA；先做真720輸入GPU容量、finite backward、production reload，再另開run。不能拿舊loader去猜新增FPN形狀，不能同時私改score、labels或原門檻。增加stride2可能顯著增加feature/RPN算力與記憶體，須實測；資料外觀／遮擋覆蓋仍是另一路徑，CPU選例不能證明改架構一定有效。
