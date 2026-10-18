# Phase 3 晴天範圍與正式完成條件

2026-10-16 使用者指示：「可以不用考慮雨天情況，強制模擬器永遠晴天也可以。我們在理想環境中進行模擬」。從本版本起，訓練、驗證與後續理想環境模擬固定 ClearNoon；既有雨夜失敗保留歷史，不再列為本階段必須改善的品質門檻。對雨夜、真實道路的泛化沒有主張。

Phase 3 尚未完成。目前凍結的 configs/dataset_sunny_v3_8.json 採18個30秒路線群組，train=Town01/Town03、validation=Town04、development=Town05、lockbox=Town02。每秒儲存5 bundles，每bundle八方向RGB、semantic、depth、instance共32張，已實收2700 bundles / 86400 PNG；18場採集與全圖獨立稽核均通過，另有assistant GT抽查綁定收據。單行道路可採同向車隊，沒有真實人行道的場景不造出行人路徑；各split的逐類正樣本及群組覆蓋另行檢查。

訓練資料必須通過原始影像CRC/SHA、時間與capture pose、actor taxonomy、真實可見框、ignore前景、類別與群組覆蓋，以及GT抽查。未經驗收的fixture與失敗採集不可批准訓練。卡車 european_hgv、公車 fusorosa、施工交通桶 trafficcone01 在五城鎮32-sensor試驗中分別實測raw15、16、21。raw21只對已核准且ID一致的施工交通桶給正例，其他Dynamic物件仍ignore。Town04首次靜態fixture放置碰撞失敗保留，修正fixture先在空中生成再停用物理放置；這不是無碰撞駕駛證據。

偵測使用TorchVision 0.25.0+cu128的MobileNetV3 Faster R-CNN，官方COCO起始權重SHA256=907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5，重新初始化本專案分類頭後自行微調。RGB是唯一模型輸入。RPN與ROI都在真實匹配後、採樣前套用ignore，正例優先；ignore框必須隨影像resize。單元測試使用真實RPN/ROI採樣器驗證，不只測獨立NumPy函數。上游版本相容表：[PyTorch官方](https://github.com/pytorch/vision)；模型來源：[TorchVision官方原始碼](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/detection/faster_rcnn.py)。

偵測逐類輸出AP50、AP50:95、固定score=0.5的Precision/Recall，max100 predictions/image，101點插值、10個IoU門檻。自有visible-box evaluator不冒稱完整COCO（沒有crowd/area標準分解）。缺類為null並使完整類別驗收失敗，不能把缺類算滿分。分割另外保留逐類IoU/Precision/Recall與各group指標。模型輸出圖與FP/FN案例必須是實際checkpoint推論。

新資料採集與heldout模型評分前凍結品質門檻：偵測mAP50>=0.50，各物件類score0.5 recall>=0.50；分割focus mIoU>=0.50、各focus IoU>=0.20。validation選模，development用於診斷，lockbox只在選模完成後開啟一次。未通過就明確記錄失敗，不能以事後調低門檻宣稱完成；若根據lockbox改進，該集合轉development並補新lockbox。

完成仍需：完整批准資料、實際detector與擴充分割訓練、可重載權重與版本指紋、RGB-only推論、heldout逐類評估、錯誤庫與品質門檻結果。沒有上述證據就不進Phase4。Phase3完成也不等於已有3D/BEV、自主規劃或安全駕駛閉環。

人工式視覺檢查（Codex assistant）確認 blueprint trafficcone01 的實際外觀是橙白交通桶，不是尖錐；模型類別維持 temporary_obstacle，不能主張已涵蓋各種施工錐或所有臨時障礙物。新分割GT只依verified_obstacle_actor_ids對應已核准資產。見evidence/phase3/sunny-v3/assets-visual-review.json。

訓練前補強驗收：configs/sunny_training_v3_6.json 在任何新模型評分前另要求每一偵測類別於固定score0.5的precision>=.5，保留原mAP/recall/IoU門檻。AP可能在GT已全命中後忽略較低分大量FP對操作點的影響，因此不能只靠AP。分割靜態映射已依CARLA0.9.16官方ObjectLabel.h核對：raw4=wall、5=fence、6=pole、11=sky（歸other）；新增測試直接對照carla.CityObjectLabel，草稿誤列11為wall在訓練前已修正，沒有產生錯版權重或成績。

號誌補充：configs/light_state_v1.json、light_labels.py、LightStateNet。燈具由semantic7+軸向depth+實際lamp world OBB對應；相位標籤以simulator phase為來源，只有至少2個已確認燈具像素符合互斥的發光色範圍才標red/yellow/green，其餘unknown_or_unobservable。這是有明確可見性條件的合成GT，不是人類相位認證；不能從背面偷讀隱藏相位。class/group覆蓋不足不得訓練該head。後續必須分GT-crop分類與實際detector-box聯合召回、red→green錯誤率，不能只報完美GT框分類。

歷史005版採集為data/sunny-cross-town-v3-005、configs/dataset_sunny_v3_6.json；六個幾何、seed、split、criteria、rig完全相同的已通過場景由004以PNG hardlink及獨立metadata副本沿用，report.reuse_lineage保留來源hash。它們算該資料集的六個groups，不是新增獨立樣本；失敗場景不沿用。Town03-02僅將無法在彎道生成的卡車移到同一對向車道的直線段，剩餘12場先全部原生spawn預檢通過。

模型訓練最終草案configs/sunny_training_v3_8.json：MobileNet P2/P3/P4/P5+pool（strides4/8/16/32/64）照顧小物件；保留官方backbone、高層FPN、RPN卷積/ratio通道，新增低層FPN與任務head再微調。分割增加terrain/ground成17類，loss為weighted CE+0.5 foreground Dice，避免薄線與稀少body只由大量背景主導；ignore255同時遮罩兩項loss。只輸出可見road/road-line↔sidewalk/terrain界面，另報2px tolerance boundaryF1（門檻.5）；不把車輛遮擋輪廓當道路邊界，也不補不可見邊界。所有新增門檻均在任何新模型評分前設定。

Ignore-aware訓練也處理整批ROI/RPN均無監督樣本的情形：空CE不產生NaN，回傳可微零loss；全無監督批次跳過optimizer與schedule，記錄跳過數。完全空白的有標註背景negative仍正常訓練。

004/003兩次Town01相同seed的交通互動差異很大：003卡車0.56m未過門檻，004同位置73.99m且新紀錄沒有zero-speed-limit ticks，不能把改善歸因於缺值fallback。原生模擬的交通時序/互動未宣稱bitwise重現；各次失敗、實際控制、位姿與採集hash保留，零碰撞/位移門檻沒有降低。

原生CARLA client在跨城鎮時曾assert _episode!=nullptr，Python finally無法攔截；伺服器仍正常載入新town。supervise_sunny_collection.py僅在所有既有episodes已通過且cleanup成功、下一場尚未開始時有界重連新client；任何半成品/失敗scene都停下診斷，不盲目覆寫重試。執行狀態以artifacts/reports/sunny-cross-town-v3-005-execution.json的child poll及timestamp確認，不能只看舊log假設仍在採集。


2026-10-17 實際執行更新：canonical 為 data/sunny-cross-town-v3-007，audit完整86400 PNG通過；105物件GT panels、18語意targets及四split16個相位cro​​ps由assistant實際檢視，另存 visual_review.json 綁定audit與review圖SHA。已正式開始 configs/sunny_training_v3_9.json 的17類分割訓練；不是舊8類權重的重跑測試成績。

主 lockbox 黃燈只有4個crops／1群組，沒有通過獨立相位覆蓋。configs/light_lockbox_supplement_v1.json 在phase／detector模型評分前凍結17個既有Town02 heldout視角的Off/R/Y/G控制補收；它只補phase測試，不作訓練、不算新增動態路線。仍用原route group，新增同pose Off亮度對照。configs/light_state_v1_1.json維持原門檻，main dynamic與controlled joint分開保存後再報pooled phase驗收。補收尚未執行／批准；主audit通過不等於phase可以忽略缺類。


Tiny 三輪實驗（sunny-segmentation-v3-001）已保留後刻意中止，最高 val mIoU29.37%，不是完整訓練通過。train/validation抽查證實靜態類過擬合；轉為configs/sunny_training_v4_1.json：COCO MobileNetV3 FrozenBN encoder＋3.59M multiscale decoder，8epochs/batch8、BF16卷積/FP32參數與loss、train-only paired scale/crop/pad255/flip。CUDA真實正反向、exact reload及batch8約4GB容量已通過，71 tests通過。新run v4-001重新正式訓練中，原品質門檻沒有改動。號誌另需在任何heldout評分前先以actual detector boxes通過joint validation，matched-red→green錯誤率仍不得超過2%。

2026-10-17後續實測：Mobile分割v4-001已完成8輪/9600updates，val focus mIoU .5084561但wall .0619479、fence .0072838、visible margin F1 .4073334使品質失敗，未開heldout。27source/權重/設定/逐張混淆核對通過；8個錯例及4個歷史固定train/val probe已實際看圖，材質/類別分布差異另存evidence。號誌v1-001全輪後JSON收尾NumPy bool失敗已保留；只修型別，新v1-002從頭重跑16輪/832updates並成功completed，best/last tensors與原run逐一相同。GT crop macro recall .9438117、red→green10/753=.0132802通過；11類型錯例已看圖，actual detector-box joint尚待做。queue56349已開始detector v3-001，15:43:44凍結source；不能把啟動當成訓練完成。新分割/號誌模型卡與registry已按真實狀態更新，整個Phase3仍未完成。


2026-10-18統一查核：detector v3-001已完成8epochs/19200updates，best第4輪，mAP50=.60210175、AP50:95=.36368405；行人recall=.499072、自行車=.307270、traffic_light=.381757未過。號誌用該final detector實際框joint macro=.360204<.4，matched red→green=0；GTcrop通過仍不足以驗收，所有新模型heldout未開。

凍結17視角號誌補收現已原生完成於data/light-lockbox-supplement-v1-002，68組／204PNG；audit及兩個原路線四相位QA實看通過，unknown38/red14/yellow14/green14，各2groups。它只補heldout coverage、不供訓練、不代表phase品質已過。第一次001缺幀失敗保留。

景物補收計畫configs/segmentation_supplement_v1.json只含原train Town01/Town03，128道路錨點×8方向，靜態虛擬rig，不捏造動態episode。原生inventory與白色picket預覽已實看，仍待完整補景audit/review後才訓練。v5從各自v4/v3 best微調，保留所有門檻；validation-only選模優先所有門檻均過，再按原任務均分選模，同分最早；若無合格仍選均分最高並記失敗。每epoch保存全部門檻結果，避免較高均分但漏類的權重蓋掉真正合格的候選。
