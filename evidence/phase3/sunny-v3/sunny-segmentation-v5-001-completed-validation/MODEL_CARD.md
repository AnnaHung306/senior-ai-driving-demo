# MobileUNet segmentation v5-001 — 驗證未通過

從v4 best微調6epochs／10272新updates，選定第6輪，重新載入RGB推論exact一致。17類／16focus；ClearNoon、RGB-only推論，BF16卷積與FP32參數/loss/output。

主train9600views＋核准train-only靜態補景1024獨特views每輪重複4次；sampledviews13696/epoch，主資料70.09%。主資料Town01/03 train、Town04 validation；靜態錨點不是新動態路線。encoderLR5e-5／decoder.00015、batch8、CE＋.5Dice及train-only paired augment維持原定義；重置optimizer/scheduler，不宣稱matched-compute因果比較。

完整GPU validation3600views：focus mIoU **0.59002533**，visible margin F1 **0.49446801**。Wall IoU及margin兩項失敗，其餘15focus類IoU合格。Phase3未通過，沒有development/lockbox成績。

| 類別 | GT pixels | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| other_annotated | 413857841 | 0.885319 | 0.952594 | 0.926123 |
| road | 198638171 | 0.899594 | 0.967684 | 0.927456 |
| sidewalk | 31957757 | 0.487799 | 0.501169 | 0.948146 |
| pedestrian | 399691 | 0.297888 | 0.567105 | 0.385560 |
| rider | 604233 | 0.560576 | 0.778593 | 0.666883 |
| car | 2651790 | 0.703707 | 0.873315 | 0.783709 |
| truck | 1342377 | 0.707694 | 0.847625 | 0.810851 |
| bus | 17321696 | 0.955939 | 0.972736 | 0.982257 |
| motorcycle | 446570 | 0.573949 | 0.688217 | 0.775623 |
| bicycle | 201966 | 0.380689 | 0.672052 | 0.467544 |
| road_line | 5694147 | 0.762389 | 0.872079 | 0.858383 |
| traffic_light | 212553 | 0.502017 | 0.793692 | 0.577357 |
| wall | 2048168 | 0.093799 | 0.114465 | 0.341898 |
| fence | 8168548 | 0.614746 | 0.685330 | 0.856504 |
| pole | 8926242 | 0.457485 | 0.528679 | 0.772583 |
| temporary_obstacle | 432081 | 0.692369 | 0.918687 | 0.737568 |
| terrain_or_ground | 136292335 | 0.749768 | 0.923697 | 0.799270 |

權重只留本機 `C:/code/senior_AI/artifacts/runs/sunny-segmentation-v5-001/best.pt`；SHA `3fe7bd08d885e481b7435fdf466945f142f8323447cca46bf31c32455835a53f`。source `edbdf26f4a4acd63f71b8034163ebf5eebf72e607c039e71a054a73217677132`、config `db034d5d761b8aeb3fcc40aa1cb8c3ea6f565f83a4db4c0d1f8bb72f0fc18d04`；main及補景audit/review/plan指紋見provenance。新run訓練牆鐘5252.77s，峰值allocated4046.51MiB/reserved4662.00MiB。這不是八鏡頭效能或20Hz閉環測試。

3頁8個val錯例已實看，CPU FP32重建只供診斷；正式指標來自原GPU BF16報告。木質實心Wall漏檢、岩面誤報與草地/sidewalk混淆仍存在。分割confidence未校準；可見road/line界面不是遮擋後自由空間、導航安全區或3D重建。短序列、少量共用資產與相關錨點，沒有可靠廣泛泛化信賴區間。
