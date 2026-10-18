## 2026-10-18 實際GPU A/B結果：本版暫不納入

以下原方案保留實驗前背景；使用者只授權有限小測，實際執行的是32步GPU檢查＋兩組各一輪，不是下文兩輪提案。GPU容量／BF16／depth-only encoder梯度與重載皆過，兩組各1712更新、13696採樣train views、相同parent及實際增強RGB／semantic bytes，完整3600張validation比較。

其他15類focus mIoU control64.069403%、depth64.150772%，只多0.081369個百分點，未達事先0.2pp研究篩選值。全部16類mIoU61.096831%→61.237003%，Wall16.5083%→17.5305%仍低於原20%；margin F1 .614658→.615844。ped+1.0949pp、sidewalk+.8134pp，但bike−.3336pp、motor−.5637pp、rider−.5227pp，Town04-01場景退步。三頁六情境均實看，原GPU batch confusion/boundary重播exact。

工程可行已通過，這版整體實益不足，完成首輪篩選後停止，不開第二輪或係數搜尋。單seed單epoch無統計顯著性／穩定性證據，也不能推論深度方法永遠無用。parent至depth增益含對照續訓效益，不能全部歸因depth。nearest loss與先前CPU toy bilinear不同；本輪評segmentation IoU，沒有完整val depth RMSE／AbsRel benchmark。原門檻維持，heldout未開，獨立detector未改。

[完整報告與原圖](evidence/phase3/sunny-v3/depth-auxiliary-AB-v1-001-completed/README.md)包含44個source、GPU檢查、兩臂指標與實看收據。

# 影片與原始文獻對照：Wall 泛化及深度輔助訓練

2026-10-18。範圍仍為 ClearNoon、RGB-only 正式推論、Phase 3 二維感知。本文件是改進方案與可行性證據；沒有宣稱已訓練下列新模型，也沒有變更正在執行的 v6 批次。

**最新小測：**使用者後續要求先只做有限深度分支可行性，不大量嘗試。已另跑一次CPU FP32原型：4張fit＋4張其他原train view、24updates、新增6945參數，深度fit loss .11706→.04277、其他train view .13544→.04399，depth-only梯度可回傳encoder、初始RGB語意與原模型exact、checkpoint重載exact。兩頁實看仍有條紋／模糊／遠景錯距，GPU/BF16及Wall跨城鎮改善未驗證；尚未正式整合，也未自動啟動下文的3424-update對照。收據與圖見[evidence](evidence/phase3/sunny-v3/depth-branch-CPU-probe-001/README.md)。

## 已實際檢視的內容與限制

指定[影片](https://www.youtube.com/watch?v=--cGYatMESY)為大刘科普的18:45技術解說。已用瀏覽器檢視關鍵畫面及燒錄字幕：1:52 RegNet／BiFPN／多任務head、5:37影像到車體座標、7:30時間資訊、9:22 Spatial RNN、11:15離線標註流程、15:15 Depth CNN／Pose CNN重建圖。

播放器未提供可擷取的CC／逐字稿。以上是實際關鍵片段檢視，不能寫成完整聽讀逐字稿，也不能把評論者的說明當成Tesla公開完整實作。15:15圖與Zhou等人CVPR2017原論文Figure 2/3相符，已另外查閱原文。下列PAD-Net、SegFormer、MixStyle等是本次額外研究，不冒稱全都出現在影片中。

## 對本專案問題的判斷

完成v6的原GPU完整3600view：Wall IoU .15645294、P .18735596、R .48679210，FP4,324,560、FN1,051,136像素。FP真值主要為other2,150,166、Fence973,196、Sidewalk578,319、Terrain499,916。三條validation路線Wall IoU .13663/.02127/.20896，表現差異大。

沿用之前已固定的28張train外觀／negative選例，v6 CPU FP32 Wall IoU .97739，兩頁六個最高FP/FN情境已實看；這是有偏的seen-train診斷，不能當完整train accuracy或與GPU分數作嚴格matched-compute比較。它與完整val的落差仍支持優先調查場景／外觀轉移，而不是預設模型沒學會訓練標籤。

新Town06/07資料已實際採集、audit及視覺批准，但主要新增磚石Wall與草地／岩面／Fence negatives，尚未證實補到實心木Wall或高速擋土牆外觀。新增樣本量不等於補足外觀缺口。原raw4=Wall、raw5=Fence已對五個validation極值案例核對，不改標籤來求過關。

深度提供距離與表面幾何，推測可減少部分地面／路緣誤認Wall；但Wall與Fence均可為直立表面，單一平面、法向或距離規則不能決定語意。other亦含直立建築，不能把它當全是地面。這些是待驗證假說，不是文獻已證明本案有效。

原始證據：`evidence/phase3/sunny-v3/sunny-segmentation-v6-001-completed-validation`、`segmentation-v6-wall-diagnostics-001`、`segmentation-expansion-v2-001-approved`。

## 方法如何轉成可行實驗

| 方向 | 原文支持的機制 | 本案採用方式與限制 |
|---|---|---|
| RegNet／BiFPN | RegNet研究網路設計空間；BiFPN做可學權重的雙向多尺度特徵融合 | 現有MobileUNet已融合多尺度skip，不能把「加FPN」視為從零補缺。先保持backbone，若需再比較輕量雙向融合或context head。換RegNet本身沒有解決Wall外觀轉移的保證。 |
| 深度與語意多任務 | PAD-Net展示RGB輸入可利用深度／法向／輪廓等預測任務改善表示 | 優先試同一RGB encoder加訓練用depth head；GT depth只作loss target，推論丟棄此head。這是借用幾何輔助監督觀念，並非聲稱重現完整PAD-Net蒸餾架構。 |
| 深度／姿態自監督 | Zhou等人以預測depth與pose重建鄰近影像；Monodepth2處理部分遮擋及違反相機運動假設的像素 | 我們已有經稽核模擬深度，先做直接輔助監督。若之後做自監督，需處理移動車／行人、遮擋、無視差、尺度及光照；不能直接照搬靜態剛體假設。 |
| 外觀泛化 | MixStyle混合train特徵的統計量；TLDR研究避免單一來源紋理過擬合並保留有用紋理 | 原型已有CPU檢查，尚未正式整合／GPU訓練。另立style-only對照，僅用核准train，不把val／heldout當未標註target domain送入UDA訓練。 |
| 較大範圍語境 | SegFormer融合階層式多尺度特徵與局部／全域資訊 | 如果depth或style小實驗不足，再比較B0等小型模型或輕量context head。先量VRAM、loss／gradient、權重轉移與RGB reload；不能用論文別的資料集mIoU預言本案成績。 |
| 時序記憶 | 影片的Spatial RNN；NetWarp用相鄰RGB光流對齊中間特徵 | 後續先做每鏡頭3張因果序列，處理短暫遮擋與閃爍，不能只平均錯誤mask。跨鏡頭BEV／空間RNN屬Phase4；不作為Phase3 Wall問題的直接替代品。 |

## 已完成的深度資料可行性檢查

另做有限、CPU/file-only、零模型推論的train檢查：八個原train groups各八鏡頭、每個group/camera固定第一張，共64view；選例在讀depth前決定。256個RGB／semantic／instance／depth PNG的SHA與原inventory核對，train index、audit／review／inventory及每個manifest／calibration均綁定。沒有開啟heldout模型評分。

沿用原auditor的packed RGB解碼 `(R+256G+65536B)*1000/(2^24-1)` 米，而非把彩色depth圖當灰階。使用提案用mask（raw0及sky排除、finite、.1..120m）後，有效11,190,183／14,745,600像素，約75.89%。這個range只供可行性檢查，尚未凍結正式depth target。

| 所選GT類別 | 有效像素 | 距離p01／p50／p99（m） |
|---|---:|---|
| Wall | 1,268,814 | 3.78／6.58／72.67 |
| Fence | 198,575 | 3.83／8.72／74.33 |
| Terrain | 485,910 | 5.24／12.58／111.62 |

這些距離明顯重疊，不能用「近的是Wall、遠的是Fence」之類規則分類。分位數僅描述固定64張train子集，不代表全split分布、深度模型準確度、Wall改善或完整訓練管線已完成。exact tool／contract／report保留於`artifacts/runs/phase3-depth-supervision-preflight-001`及同名evidence封存。

## 建議先做的最小對照

先等現有v6訓練、整合與排隊GPU錯例QA完成，保持凍結來源。取得完整CPU偏置診斷後再決定是否有必要正式GPU重驗偏置，不根據部分資料選alpha。下一個主要新方法優先為depth輔助；不要一次換backbone、loss、資料與時序。

1. **B0：原模型繼續訓練對照。** 從同一v6 best SHA初始化，固定相同train資料、batch、資料順序、幾何／photometric增強、optimizer、LR schedule及3424次更新（兩個13696-view epochs）。這是新實驗的等更新量對照，不能拿既有六輪總分冒充等計算對照。
2. **B1：只增加depth輔助監督。** 初始化的segmentation所有tensor與B0逐一相同；新增低解析度depth head，例如stride4的輕量conv，保持原17類CE＋Dice與RGB-only分割輸出。先用train-only小批次凍結valid range、log-depth參數化與輔助loss係數，再做與B0相同3424更新。新增head初始化使用獨立或保存／恢復的RNG，避免悄悄改變資料與增強順序。額外head有計算成本，等更新量不冒稱等GPU時間或等FLOPs。
3. **依結果才加第二方向。** depth若降低Terrain／Sidewalk誤報但Wall/Fence混淆不改善，另外做B2 style-only，沿用相同B0條件；depth＋style合併須另版，不混寫B1結果。若兩者皆無實質改善，先檢查長範圍context及確實缺少的外觀，再比較輕量context／SegFormer-B0。

兩輪是有界的篩選實驗提案，不預設必須跑滿所有分支，也不聲稱它們已啟動或正式通過。下一版config／source須在首個候選GPU評分前凍結。若篩選未見收益，記錄負結果與分支停止理由，避免盲目再加epochs或只補相同磚牆。

## Depth輔助的工程要求

- 原train index沒有depth target欄位，現有SunnySamples亦未載入depth；需正式版本化target compiler／loader，綁原inventory與各補景來源，不能只把檔案存在當已整合。先沿main+已核准補景；檢查每root所有訓練深度的支持量與有效range。
- 有限訓練用示意：`L = 原17類CE + .5 Dice + λ_depth L_depth`，`L_depth`採valid-mask的log-depth robust loss。λ、range、head／normalization為待train-only pilot凍結的設定；先不加normal與boundary兩種新loss，避免同時改太多。空mask必須有finite零損失，遠端／sky／padding不造假GT。
- depth PNG先解碼米值後再同步crop／resize／flip；不能對packed三通道做連續插值再解碼。縮放不把公尺值乘上影像scale，邊界避免插值跨物件。target／valid mask／語意／RGB必須同位；對需要K的後續幾何，記錄像素仿射並同步更新K。
- surface normals需由正確軸向depth、pixel-center convention、內參反投影後求3D法向；不是把depth灰階梯度當世界法向。水平flip的法向方向亦需正確變換。先不把此步納入B1，現有可行性檢查沒有驗證法向target compiler。
- GT depth只供訓練loss／離線診斷；正式分割forward只收RGB。另以僅含RGB＋manifest的根目錄驗證，移走label目錄仍能推論。若日後採predicted-depth融合，必須額外記實際深度網路的參數／延遲／誤差，禁止用GT depth代替其輸出。
- 外觀資料按真raw4/raw5與多原生位置確認。若要材質隨機化，需確認RGB與語意／depth仍對應、兩種語意都保留形狀／結構多樣性，先probe與audit，不能將wood Fence直接改成Wall。train-only hard negatives要同時包含建築、岩面、路緣和柵欄。

## 如何判斷真的改善

保持原16類mIoU≥.5、每類IoU≥.2及margin2px F1≥.5，所有gates同時通過才合格。固定3600view完整GPU BF16評估與qualified-first選模，保存全部候選，不因Wall過線就忽略其他類退步。列出Wall P／R／IoU、FP真值來源、逐scene／camera、近中遠距離與薄邊界；後者是診斷分層，並非事後改寫原門檻。

深度實驗另報valid pixels、AbsRel／RMSE及分距離誤差，不能把訓練depth loss下降當Wall通過。反覆val選模也可能過擬合，限制候選與事先固定設計；development／lockbox不作材料挖掘、pseudo-label訓練或hyperparameter選擇。所有val gates及detector／phase joint都過，才依原協議一次heldout。

檢查真實GPU peak VRAM、finite loss／gradient、target alignment、deterministic replay、checkpoint/config/source/data SHA與RGB-only重載；不要在正在跑detector的GPU上並行做新容量／訓練／benchmark。現有v6分割peak allocated約4047MiB/reserved4662MiB只是原模型證據，不能當新增head／Transformer容量已保證。

時序另立未來實驗：目前資料約5Hz，不能假設影片所述高頻輸入。序列不跨group／scene／camera／split、每段重設memory、保留cold start評分、不讀未來frame。online使用RGB估計光流／姿態或明確允許的感測量；不能把simulator GT ego pose當未聲明的正式輸入。遮擋／動態物件須mask或gate記憶更新，避免傳播同一錯誤。先報單幀與因果序列對照，再討論跨鏡頭BEV。

## 已查閱的原始文獻與實作

- [Zhou et al., CVPR2017：Unsupervised Learning of Depth and Ego-Motion From Video](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.html)：影片雙CNN／view synthesis圖的對應原文。
- [Godard et al., ICCV2019：Digging Into Self-Supervised Monocular Depth Estimation](https://openaccess.thecvf.com/content_ICCV_2019/html/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.html)：minimum reprojection、auto-masking等改進，仍需評估本案假設。
- [Xu et al., CVPR2018：PAD-Net](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_PAD-Net_Multi-Tasks_Guided_CVPR_2018_paper.html)：深度／語意等多任務及中間預測蒸餾；本文提案是較小的輔助監督實驗。
- [Radosavovic et al., CVPR2020：RegNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.html)。
- [Tan et al., CVPR2020：EfficientDet／BiFPN](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html)。
- [Zhou et al., ICLR2021：MixStyle官方實作](https://github.com/KaiyangZhou/mixstyle-release)：本機原型的FP32統計量等是實作改編，非原文結果重現。
- [Kim et al., ICCV2023：TLDR](https://openaccess.thecvf.com/content/ICCV2023/html/Kim_Texture_Learning_Domain_Randomization_for_Domain_Generalized_Segmentation_ICCV_2023_paper.html)：來源紋理過擬合與泛化；不把其benchmark數字移植成本案預測。
- [Xie et al., NeurIPS2021：SegFormer](https://arxiv.org/abs/2105.15203)、[官方實作](https://github.com/NVlabs/SegFormer)：官方舊安裝組合不直接覆蓋現有venv；若採用需獨立相容性／權重格式驗證。
- [Gadde et al., ICCV2017：NetWarp](https://openaccess.thecvf.com/content_iccv_2017/html/Gadde_Semantic_Video_CNNs_ICCV_2017_paper.html)：因果相鄰影像與特徵warp的可參考起點。
- [van Dijk & de Croon, ICCV2019：How Do Neural Networks See Depth in Single Images?](https://openaccess.thecvf.com/content_ICCV_2019/html/van_Dijk_How_Do_Neural_Networks_See_Depth_in_Single_Images_ICCV_2019_paper.html)：深度模型亦可能依賴影像位置捷徑，不把預測深度當可靠真值。

目前改進方向的結論：先以可追溯的深度輔助監督對照驗證幾何是否減少特定誤報，再分別處理外觀與上下文；將多鏡頭／時序／BEV分階段引入。所有效益仍待本機完整GPU實驗，影片和論文不能替代驗收。
