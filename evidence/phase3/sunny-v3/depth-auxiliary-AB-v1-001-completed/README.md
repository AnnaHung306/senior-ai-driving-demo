# 深度輔助分支 A/B：小量篩選完成，暫不納入

已實際完成32步GPU檢查，以及control／depth各一輪（各1712次更新、13696個採樣train views），每組皆完整驗證3600張。GPU檢查通過不是模型品質通過。小幅整體收益未達事前0.2個百分點的研究篩選值，這版先停，不加第二輪、不搜尋係數、不正式採用。

| 指標 | 原parent v6 | Control續訓1輪 | Depth續訓1輪 | Depth − Control |
|---|---:|---:|---:|---:|
| 其他15類focus mIoU | 63.652573% | 64.069403% | 64.150772% | +0.081369 pp |
| 全部16類focus mIoU | 60.652118% | 61.096831% | 61.237003% | +0.140172 pp |
| Wall IoU | 15.645294% | 16.508261% | 17.530464% | +1.022203 pp |
| 可見道路界面F1 | 62.221175% | 61.465750% | 61.584406% | +0.118655 pp |

排除Wall的整體額外收益只有 **+0.081369 個百分點**。原parent至depth共+0.4982pp，其中對照續訓已帶來+0.4168pp，不能把全部提升歸因於depth。以下逐類都是完整validation segmentation IoU，並非獨立detector AP／Recall。

| 類別 | Control IoU | Depth IoU | 改變（pp） |
|---|---:|---:|---:|
| other_annotated | 90.1400% | 90.1561% | +0.0160 |
| road | 95.1658% | 95.0939% | -0.0720 |
| sidewalk | 60.5855% | 61.3989% | +0.8134 |
| pedestrian | 31.5035% | 32.5984% | +1.0949 |
| rider | 55.1833% | 54.6605% | -0.5227 |
| car | 68.7174% | 68.9984% | +0.2809 |
| truck | 70.6649% | 70.7544% | +0.0895 |
| bus | 95.6860% | 95.6914% | +0.0054 |
| motorcycle | 55.4187% | 54.8550% | -0.5637 |
| bicycle | 39.8196% | 39.4860% | -0.3336 |
| road_line | 75.4166% | 75.9310% | +0.5144 |
| traffic_light | 58.6966% | 58.4619% | -0.2347 |
| wall | 16.5083% | 17.5305% | +1.0222 |
| fence | 64.1484% | 64.5626% | +0.4142 |
| pole | 45.0521% | 45.0545% | +0.0024 |
| temporary_obstacle | 65.9358% | 65.6956% | -0.2402 |
| terrain_or_ground | 79.0469% | 79.0192% | -0.0277 |

行人+1.0949pp、人行道+0.8134pp，但bicycle−0.3336pp、motorcycle−0.5637pp、rider−0.5227pp。兩組都只有Wall未達原分割門檻，其他15類及margin過原門檻；這沒有消除現有detector／joint的不足，也沒有使Phase3通過。

| 場景 | Control固定15類mIoU | Depth固定15類mIoU | 改變（pp） |
|---|---:|---:|---:|
| town04-00 | 62.6123% | 62.7594% | +0.1471 |
| town04-01 | 57.7362% | 57.4727% | -0.2635 |
| town04-02 | 49.6499% | 49.7895% | +0.1395 |

Town04-01退步，其餘兩場景略升，收益並不一致。場景級類別support不同，此固定15類診斷不是把三場當獨立試驗，也不是整體mIoU的等權分解。

GPU與公平性核對：depth-only梯度到head及共用encoder均有限非零；eligible卷積BF16、head FP32，32步實際peak allocated4184.33MiB／reserved4870MiB，8-view雙輸出reload exact。32步更新已丟棄，兩組均重新從parent初始化。兩組初始完整GPU報告exact parent，最終seg checkpoint reload exact。

同起始權重、seed、13696採樣訓練影像、LR／seg loss／paired幾何、photo增強及更新次數；整輪indices／幾何／實際增強RGB與semantic bytes SHA完全相同：`ee0473be0181ac1789eb5086557fcb437f1f57d4343b0852c62ec4b2b634b7e1`。深度只加6945參數head與.2輔助loss，GT depth只在train，推論仍RGB-only。等更新量不是等FLOPs；control也載入深度以保持共用資料路徑，但不向網路輸入GT。

深度loss使用.1..120m normalized log與nearest上採樣，以避免CUDA bilinear backward的不確定性，與前CPU toy bilinear設定不同。深度train平均loss.0357741；這不是完整val depth RMSE／AbsRel證明。

完整兩臂各3600筆檔案重算confusion／逐類與場景指標／boundary counts exact，canonical validation index與parent GT support exact，44個source SHA與parent所有固定文件／weights一致；見file_integrity.json。三頁六情境以原GPU batch重播，confusion／boundary exact，全部實看；另看case01與04個別panel。gallery是每場景最大gain/loss選例，不是代表性抽樣；見assistant_review.json。

只測一個seed、每組一輪，沒有統計顯著性或穩定跨場景收益證據；不能推論深度技術永遠無用。這版已完成工程可行性及首輪收益篩選，現有收益不足以優先納入。原品質門檻未改，development/lockbox未開模型評分，未自動增加輪數。所有GPU程序已退出，awake請求已清除。

權重及完整逐幀結果只保留本機ignored，SHA如下（本Git archive沒有權重）：

- control final.pt `eaf4c4cad9679aeb0e38a9a1910cc0cd3cd2d3775a575fa89cb4d8ac5641158d`；validation_frames.jsonl `4a8eb4df965f5da56d1c48f46c40418da390179e90be9fccce27f3523bbd544b`。
- depth final.pt `12cd9e8bbaceb495b91a1b91c0e810e446b57d157845f55bb4ef30eb374fef88`；validation_frames.jsonl `f6b4e5e4469b1bf20119c80e384507b3cd6bf045c7010a384dd8147176dda457`。

[三頁比較圖](validation_comparison/catalog.json)／[完整逐類比較](comparison.json)／[工程GPU檢查](GPU_preflight.json)／[實看決策](assistant_review.json)。
