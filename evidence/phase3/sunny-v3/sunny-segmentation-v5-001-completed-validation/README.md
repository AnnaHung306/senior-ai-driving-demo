# Sunny segmentation v5：訓練完成，品質尚未通過

實際完成6輪、10272updates，選定第6輪，checkpoint exact RGB reload通過。完整原3600 validation/3個Town04 groups的GPU BF16結果：16類mIoU .5900253、wall .0937987、fence .6147463、visible-margin F1 .4944680。原每類IoU≥.2與margin≥.5兩項仍失敗，其餘15類IoU合格。不得稱Phase3完成或開heldout。

best.pt SHA `3fe7bd08d885e481b7435fdf466945f142f8323447cca46bf31c32455835a53f`，權重只在本機 `artifacts/runs/sunny-segmentation-v5-001/best.pt`。主資料及train-only靜態補景來源、採樣倍率、初始化與source/config/data SHA見provenance/config/statistics；不是從零訓練或新增動態路線。

file-only checker已核對checkpoint/配置/凍結source、完整epochs與選模、3600逐frame整數confusion加總及原gate。validation_review三頁八情境已全部實看：漏掉部分木質實心Wall、道路分隔，岩面和grass/sidewalk混淆仍影響可見邊界；白picket/圍欄較舊模型改善。圖為CPU FP32診斷，不替換正式GPU成績。完整逐frame報告留本機，SHA記於state與artifact_check。

未開development/lockbox，也未降低門檻。現有批次接續偵測、號誌及統一val/joint/demo/bench/export；分割外觀泛化修復依PHASE3_DOMAIN_RECOVERY.md繼續。
