# Phase 3 訓練與驗收協議

本協議限定理想晴朗白天 ClearNoon。原始資料、版本化配置、來源快照與實測收據優先於說明文字；訓練完成不等於品質通過，Phase 4 不以中間成果啟動。

## 資料與隔離

主資料 `data/sunny-cross-town-v3-007`：18 條30秒路線、2700同步 bundles；21,600 RGB與相同數量的semantic、axial depth、instance PNG，共86,400張。全圖CRC/SHA、sensor時戳/actual capture pose、taxonomy、可見框幾何、類別/群組支持通過。assistant抽查收據另綁audit與實際檢視影像SHA，不冒稱逐圖人工認證。

| 分割 | 城鎮 | 路線群組 | RGB views |
|---|---|---:|---:|
| train | Town01、Town03 | 8 | 9600 |
| validation | Town04 | 3 | 3600 |
| development | Town05 | 3 | 3600 |
| lockbox | Town02 | 4 | 4800 |

同路線的時間、方向、相位影像皆高度相關；共用車輛/燈具資產，城鎮切分不等於資產OOD。不得把像素、連續frame或相位補收當成獨立路線，沒有可靠廣泛泛化信賴區間。

RGB是網路唯一輸入。semantic/instance/depth、lamp world OBB與actor state只在資料標註/訓練/評估使用。由規則導航與GT collision supervisor產生資料，不是learned driving。不可讀未通過audit與綁定視覺抽查的資料。

八類detector：car、pedestrian、bicycle、motorcycle、truck、bus、traffic_light、temporary_obstacle。可見instance tight xyxy框、正例優先；未解析或低支持前景進ignore，RPN/ROI匹配後採樣前實際套用。兩輪車合併同verified ID的rider/body；臨時障礙目前限核准橙白交通桶。17類語意映射見`sunny_data.py`，unknown raw21不能自動當障礙物，raw0忽略255。

## 已觀察問題與模型修正

Tiny實驗`sunny-segmentation-v3-001`保留三完整epochs與checkpoint後刻意停止；最高val focusmIoU29.37%，不記completed。第二輪selected train/val壁面IoU97.35%/0.17%、柵欄79.54%/0.75%；這是特定支持最高樣本的診斷，非全split平均。完整val也顯示這兩類很弱，單看loss下降不足以證明泛化。

`sunny_training_v4_1.json`改為官方COCO MobileNetV3 encoder完整tensor transfer、FrozenBN、多尺度U-Net decoder與全解析度RGB branch（3,589,953 parameters）。自行訓練17類decoder並微調encoder；CUDA BF16 autocast、FP32參數/optimizer/loss/輸出。strict deterministic fwd/back與exact reload實測通過。batch8 synthetic allocated約4056MiB、reserved4624MiB；不是accuracy或整套模擬記憶體保證。

train-only scale .75/1/1.25、隨機crop/pad255與horizontal flip .5，RGB/target同步、target nearest，photometric .8..1.2。validation不變。這是二維分割增強，沒有把縮放後的影像當成原calibration的三維訓練樣本。

detector依`sunny_training_v3_9.json`使用COCO MobileNet與高層FPN/RPN部分transfer，重新訓練專案head與低層pyramid，stride4/8/16/32/64照顧小物件。Windows runtime無Triton，採上游deterministic eager ROIAlign訓練；eval用fused forward，保持固定sampling_ratio2與ROI預算。

## 最佳化、恢復與來源

各run凍結config原始bytes、所有實際來源SHA、source snapshot、完整資料/審核指紋、Python/Torch/TorchVision/NumPy/Pillow/CUDA/cuDNN/driver/device。checkpoint含model、AdamW、cosine schedule、各RNG與完成epoch。只允許來源/config/data/runtime完全一致的epoch resume；文件Git revision單獨變動不阻擋恢復。

開始正式訓練前先80步兩view overfit sanity，損失比需≤.8，之後重建模型與seed，避免偷用sanity更新。finite loss/gradient強制檢查、clip5；detector全無監督的ignored批次不更新optimizer/scheduler。有標註空目標背景仍訓練。

分割loss：train-frequency inverse-sqrt weighted CE（1..20）＋.5 foreground Dice；兩項均遮罩ignore255，Dice只計當批有支持的非other類。v4 encoder LR5e-5、decoder5e-4；8epochs/batch8。detector8epochs/batch4、LR2e-4/warmup250，精度FP32。seed20261016。CARLA與GPU訓練分開執行；無CPU暗中替代。

validation選模：detector最大AP50:95、segmentation最大focusmIoU、phase最大macro recall；同分取最早。重新載入最佳checkpoint後必須重做RGB-only推論並完全相同。不得用lockbox選epoch、閾值、增強或分類映射；未過val不進heldout。

## 號誌及實際偵測框

四類RGB crop網路：unknown/unobservable、red、yellow、green。native phase只是GT來源；至少兩個semantic-lamp像素有相容可見燈色才給已知狀態。背面、遮擋或無法解析者unknown，不以隱藏phase冒充可見性。

主lockbox Yellow只有4crops/1group，未滿原quota。`light_lockbox_supplement_v1.json`在phase/detector模型評分前凍結17個Town02原路線視角，新增Off/R/Y/G共68RGB、204三模態PNG，只作測試。沿用原group，保留同pose Off對照；已知相位另須≥2相容色像素比Off亮>30，通過獨立CRC/geometry/phase audit與實際視覺抽查後才可用。這不是新的動態episode，dynamic與controlled結果分開報告。

先驗證GT-crop，再以所選detector的validation真實框檢查joint，兩checkpoint、validation frames與joint report均綁SHA。固定detector score≥.5、IoU≥.5一對一匹配，漏偵測列第五欄，錯分與偽號誌計入precision。matched red→green rate不能以大量漏判稀釋；val joint失敗不開lockbox。

## 品質條件與交付

| 任務 | 必須條件 |
|---|---|
| detection | mAP50≥.5，8類各自Precision/Recall@score.5≥.5；缺類失敗 |
| segmentation | 16非other類focusmIoU≥.5、每類IoU≥.2；可見road/line↔sidewalk/terrain界面2px F1≥.5 |
| phase crop | 4類齊全，macro recall≥.8、每類recall≥.5、unknown≥.8，red→green≤.02 |
| phase joint | 4類齊全，含漏偵測的正確狀態macro recall≥.4，matched red→green≤.02 |
| phase coverage | train各類≥20、val/dev/lock各類≥10，且各類≥2原groups |

visible-box AP用101 recall點、IoU .5:.95、max100，不冒稱完整COCO evaluator。逐類AP/P/R/IoU、逐scene、小框/截斷recall及FP/FN/phase PNG錯例都保留，不能只報平均。lockbox根據凍結模型只作最終一次評估；若根據結果改模型，該資料轉development並另補新lockbox。不得事後放寬門檻。

最終三模型必須在只有八RGB與輸入manifest的獨立根目錄完成真實推論；本機保留權重、SHA與來源快照。demo預選validation Town04-00中間frame，不宣稱其為獨立測試。個別模型與八鏡頭整套warm timing分別量測；不包含simulator/render/sensor delivery、圖檔輸出與車輛控制，不能宣稱20Hz閉環。

完整交付需正式權重、保留測試逐類指標、錯例庫、模型卡、來源/資料收據、RGB-only八方向圖與實測效能。road只是語意道路，可見邊界不補遮擋區；沒有Phase4的跨鏡頭融合/BEV/深度/3D occupancy或AI駕駛控制。


## 2026-10-18 v5修復批次

原v3 detector與v4 segmentation皆完整訓練但val失敗，final detector-box phase joint也失敗；凍結紀錄不覆寫。新增train-only靜態seg補景003：Town01/03、128虛擬道路錨點、1024獨特view、4096PNG，full audit及5頁38情境實看通過。只是static2D補景，main007不改，不能算新動態groups或送進detector/phase/heldout。每輪9600main+1024static各重複4次=13696sampledviews，主資料70.09%；資料與重複量、原始support、audit/review/planSHA都進provenance。

v5分割從v4 best再微調6輪，batch8/10272新updates，encoderLR5e-5、decoder.00015；架構、loss與映射不變。v5 detector從v3 best再微調4輪、batch2/LR5e-5，原640x360RGB/GT不變，內部resize720x1280以改善小框表示；load_model按checkpoint config重建resize，輸出仍原始座標。optimizer/scheduler重設，80步sanity後再次重建；不是嚴格matched-compute或純資料因果ablation。

新配置在任何v5評分前凍結qualified_first_then_score：只用val，先選所有既定門檻通過的epoch，再最大原AP50:95/mIoU；若都沒過則最大均分並記失敗，同分最早。每epoch記全部gate、checkpoint/resume帶best_qualified，避免低於單類門檻的高均分蓋掉合格模型。所有門檻數值不變。

遮擋的專項說明：現有動態畫面包含遮擋，偵測報告按實際可見框評分，另列小框及影像邊界截斷recall，錯例須實看遮擋／小物件情境。`visible_pixels_per_projected_bbox_area_not_visibility`只是投影矩形填充比例；細長自行車、空隙與視角同樣影響它，不能命名為物理遮擋率。若要量化25%/50%/75%遮擋的性能，須另外凍結同target/camera pose的無遮擋reference與受控occluder配對，驗證真實mask與depth前後關係，另行報告static controlled範圍，不能由目前bbox填充率捏造此結果。尚無這種受控遮擋率驗收或廣泛遮擋魯棒性保證。

號誌v1-003同超參數重跑，是為凍結正確的720 detector production loader；舊phase/source的360硬編碼不能拿來做新det controlled補收推論。新phase保存config原bytes與data SHA。統一batch先完成94項實際GPU回歸/容量，再順序訓練與joint；任何val失敗整個tuple不開heldout。正式模型還必須實際RGB-only8view、3種暖機效能與final QA，code工具或資料批准不代表Phase3通過。
