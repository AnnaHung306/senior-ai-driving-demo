# Sunny segmentation v6 — 完成訓練，驗證未通過

六輪／10,272次新optimizer更新，從v5最佳權重再微調；不是從零初始化。MobileNetV3 encoder＋17類U-Net decoder，3,589,953參數，CUDA BF16 forward、FP32參數與loss。最佳為第六輪（epoch5），正式重載推論exact。

主資料007訓練9600視圖；原Town01/03靜態補景1024與新Town06/07靜態補景1024，各重複採樣2次。每輪13696採樣視圖，main70.09%；static anchors不是新動態路線。新補景audit/review與rawbytes來源見../segmentation-expansion-v2-001-approved。loss/幾何/photometric/架構/原門檻維持不變，optimizer/RNG重設，sanity後重建；不是matched-compute因果對照。

完整3600張validation的16類mIoU60.65%、Wall IoU15.65%未達20%、Fence60.58%、可見道路界面F162.22%通過50%；其餘15focus類IoU均通過20%。仍不得將Phase3標記完成。此模型尚未對development/lockbox作模型評分。

權重只保留本機ignored路徑 `artifacts/runs/sunny-segmentation-v6-001/best.pt`，SHA `0a3fe3f66fb93be373f2e82218632a4f8cc277bf6067f48bb4683a71f1b2af30`；沒有假稱已上傳或加入Git。GPU trainingwall4922.844s，peak allocated4046.51MiB/reserved4662MiB。151項實際GPU回歸全過；後新增QA3＋bias診斷2項僅focused CPU，不合稱156GPU全跑。

16 類 focus mIoU 60.65%；可見道路界面 F1（2px）62.22%。

| 類別 | GT pixels | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| other_annotated | 413857841 | 89.99% | 97.07% | 92.50% |
| road | 198638171 | 95.23% | 97.66% | 97.45% |
| sidewalk | 31957757 | 59.79% | 63.17% | 91.80% |
| pedestrian | 399691 | 30.42% | 59.99% | 38.16% |
| rider | 604233 | 55.37% | 77.91% | 65.68% |
| car | 2651790 | 68.57% | 80.90% | 81.82% |
| truck | 1342377 | 69.31% | 87.52% | 76.91% |
| bus | 17321696 | 95.65% | 97.76% | 97.79% |
| motorcycle | 446570 | 56.13% | 71.19% | 72.62% |
| bicycle | 201966 | 39.72% | 68.54% | 48.57% |
| road_line | 5694147 | 76.45% | 83.62% | 89.91% |
| traffic_light | 212553 | 57.46% | 80.36% | 66.84% |
| wall | 2048168 | 15.65% | 18.74% | 48.68% |
| fence | 8168548 | 60.58% | 73.08% | 77.97% |
| pole | 8926242 | 44.34% | 49.25% | 81.66% |
| temporary_obstacle | 432081 | 67.05% | 93.38% | 70.39% |
| terrain_or_ground | 136292335 | 78.73% | 89.49% | 86.75% |


Wall逐scene IoU：Town04-00 13.66%、Town04-01 2.13%、Town04-02 20.90%；場景差異仍大，不能只用整體均分宣稱泛化。原CNN輸出仍有大量Wall誤報。僅理想晴朗白天合成資料；未測試廣泛天候/資產/真實道路泛化，沒有受控物理遮擋率驗收或可靠獨立試驗信賴區間。

這是逐鏡頭2D語意與可見界面，不是可安全通行區、遮擋後補全、3D融合或AI駕駛。actual GPU原batch錯例重播已排在完整批次結束後執行，現在尚未完成該視覺QA，不將CPU診斷當原GPU輸出。
