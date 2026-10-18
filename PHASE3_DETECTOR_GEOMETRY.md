# 自行車尺寸落差：訓練增強備案，尚未接入

現有完整 GT-only 診斷（sunny-box-size-diagnostic-001）顯示，train 自行車 1470 個中只有 10.95% 框面積小於 32²，validation 729 個中則有 53.22%。短邊小於8px比例分別 .408% / 16.872%。這不是新模型品質，亦不是完整因果證明。v5第二輪 GPU val 的 bicycle P .959839、R .327846，小框只命中9/388；平均 AP 改善沒有解決漏檢。

新增 scripts/experimental_detector_geometry.py 只在原正式來源之外準備成對 zoom-out／水平翻轉功能。它在原畫布內縮小完整 RGB，按真正整數輸出尺寸縮放所有 positive / ignore 框；不裁掉任何原目標。外側 padding 加入 ignore boxes，避免把人造底色當成真實可監督背景。只把 boxes / labels / ignore_boxes 三個 loss 所需欄位傳入 detector，不把原 visible_pixels / truncated 等當成增強後的實測值。原圖片／標註不變，驗證及推論完全不用此轉換。

6項 CPU 檢查涵蓋 RGB／框翻轉一致、原標註不可變、padding 精確覆蓋、空 targets、單位尺度 exact、host RNG 恢復與非法幾何拒絕。其後以 scripts/verify_detector_geometry.py 對 immutable 第一輪 v5 checkpoint 與兩個不同原 train groups 的真實 RGB／GT，做一次 ephemeral CPU optimizer step：loss .92142844、clip前norm8.719962，四loss均finite、RPN／ROI仍有監督；未寫任何正式checkpoint、原parent SHA未變。兩頁原圖／增強後positive與ignore框已實看，框跟隨縮小／翻轉、padding有ignore。完整129項CPU suite（126pass／3GPU skipped）另保存。**仍未 CUDA 容量檢查、未整合 trainer、未正式訓練或取得任何品質改善。** 不把準備完成當 Phase3 完成。

若 v5最終自行通過，就不用新增實驗。若仍是小框漏檢，優先比較這個保留現有架構的尺寸增強與 P1 備案；不要同時加 P1／proposal budget／score 變更，讓結果無法解釋。待全部目前 GPU 工作結束，再凍結具體尺度、比例、warm-start initializer SHA、新 source/config 及正式 update 數，做真 loss／finite／reload 檢查後另開版本。候選尺度 .5/.75/1 是待凍結政策，不是已執行訓練；水平翻轉雖有測試，也可不採用以保持主要改動單一。

仍使用全部八類原 gates 與原 score=.5；不挪用 development／lockbox、不降低召回門檻。牆面外觀資料修復仍須另做實際 RGB／GT 審查。

第三輪v5完整GPU結果仍只有bike recall失敗：P .6648649、R .3374486、小框22/388，其餘七類P/R過。既有完整GT-only尺寸診斷chunks的純幾何投影顯示：train bike小框原10.9524%，scale .75時12.2449%，scale .5時74.0136%。因此原型預設三尺度均勻抽取不一定足以補足尺寸支持；可考慮只對含bike正例的train views，以2/3機率scale .5、其餘原尺度（bike小框期望52.9932%，接近原val53.2236%），其他影像維持原樣。這是使用train／validation診斷提出的政策候選，未正式凍結或訓練，prototype也尚未接入這個class-conditioned採樣。先不疊加flip、P1或proposal budget改動；仍須真GPU finite/overfit/reload與全部逐類val。改過的二維影像不能搭配原相機外參冒稱Phase4幾何樣本。
