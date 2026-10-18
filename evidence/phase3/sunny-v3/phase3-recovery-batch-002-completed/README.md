# Phase3 v6 完整批次：執行與視覺核對完成，品質未通過

原 11 個階段完成。151 項 GPU 回歸通過，det 容量及兩模型初始化完整重播 exact，兩模型 checkpoint reload exact；保存的 3600-frame segmentation/detection 指標檔案一致性通過。

正式分割 mIoU60.65%、Wall15.65%未達20%、道路可見界面F1 62.22%通過；其餘15類IoU均過20%。偵測mAP50 70.46%、mAP50:95 45.10%，自行車／機車／紅綠燈 Recall 未達50%。GT crop phase macro recall94.38%通過，偵測＋phase joint macro39.9064%未達40%。不調降任何原門檻，全tuple development/lockbox未開模型評分。

原 GPU batch replay segmentation25cases/9pages、saved detection15cases/5pages及7display zooms、phase31cases/4pages，以及8-view RGB-only demo，全部實際檢視。詳見 visual_QA/visual_review.json；錯例選例不是新獨立測試。分割每個重播例的 confusion/boundary 都與原GPU儲存值exact。

八鏡頭 warm repeated-bundle inference p50 688.09ms／p95 746.02ms；含RGB decode及三模型，排除sim/render/控制，不宣稱20Hz。權重及完整逐幀結果保留ignored本機，原run state/receipt保存其SHA，未假稱上傳。

watcher首次detector checker漏 --root 的原日誌及修正執行收據另存於 ../sunny-detection-v6-001-completed-validation。修正後同一個checker full3600 canonical GT/AP/PR核對通過；未重訓或改指標。兩個原批次/QA awake helper都已隨parent退出清除請求。

當前另有已獲授權的深度一輪A/B研究，從此completed segmentation parent出發；其結果不包含於本批次，不能把其後研究當成本批次正式通過。
