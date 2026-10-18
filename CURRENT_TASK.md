## 最新終局 21:35：深度A/B完整完成，首輪收益不足，本版停止不納入

最新使用者「照你建議的是看看吧」授權的小GPU檢查＋兩臂各一輪已完整完成。**exec78878 exit0、parent23304已退出、awake22688 released/request_clearedtrue。沒有剩餘GPU job、watcher、automation或CARLA。不要重啟舊queue或自動第二輪。** 原batch79387正常quality_failed、postQA45681 exit0且18pages/7bikezooms/8viewdemo全實看，file watcher59456漏--root已補明確root執行3346核對通過；monitor cells671/709/713皆結束。原批證據已commit d15238d（前5f3d28c）。

新32步CUDA preflight真實2原train groups/batch8：depth-only head梯度.04453888／encoder.01972393有限非零、eligible conv BF16/head FP32、peak4184.33MiB/reserved4870、8view雙輸出reload exact。32步更新丟棄。control/depth各從parent0a3fe重新初始化，初始完整3600GPU報告exactparent；各1712updates/13696採樣views，RGB/semantic/order/geometry整輪SHA ee0473be0181ac1789eb5086557fcb437f1f57d4343b0852c62ec4b2b634b7e1 exact。

完整val其他15類mIoU control.6406940277337431、depth.6415077225594156，額外+.00081369482567＝+0.081369pp，未達事前.002＝0.2pp研究篩選值，positive_signalfalse。all16 .610968314→.612370030；Wall .165082608→.175304638仍fail；margin .614657502→.615844057。ped+1.0949pp/sidewalk+.8134pp，但bike−.3336pp/motor−.5637pp/rider−.5227pp；Town04-01 fixed15scene−.2635pp，另外兩scene略增。parent nonWall.636525727→control已+.4168pp，不能把parent→depth+.4982pp全算深度。

兩臂原gates都僅Wallfail，未開heldout。control final SHA eaf4c4cad9679aeb0e38a9a1910cc0cd3cd2d3775a575fa89cb4d8ac5641158d；depth final12cd9e8bbaceb495b91a1b91c0e810e446b57d157845f55bb4ef30eb374fef88，RGBseg checkpoint reload exact。權重與完整逐幀結果本機ignored沒入Git。nearest loss不同於CPU toy bilinear；depth trainloss.035774087不是val depth RMSE證明，獨立detector沒改。

file_check.py CPU實跑exit0：兩臂各3600筆matrix／逐類／scene／boundary重算exact、canonical index/parent GTsupport exact、44sources/parent files/weights/reports及3pages SHA核對過。不是GPU新增tests或品質pass。原GPU batch6QA cases confusion/boundary exact，3pages全部實看＋case01/04個別panel實看；ped silhouette稍改善、rock Wall/Other及bike/rider/motor混淆仍在。assistant_review.json completed_screening_no_adoption。完整工具／GPU／指標／44source／圖與實看收據封存 **evidence/phase3/sunny-v3/depth-auxiliary-AB-v1-001-completed**；README/研究plan/domain/registry已更新，待Git bytecheck與commit。

決策：工程可用，但此版整體收益太小且類別／scene有退步，按有限小測範圍停下，不加第二輪、不搜尋係數、不正式採用；不是深度方法永遠無用或統計顯著性證明。Phase3仍qualityfail。最終回覆簡報64.069→64.151%/+0.081pp、部分改善/兩輪類退步、Wall17.53仍未過、已停止額外訓練及完整報告連結。所有實跑/實看/封存已做完，保存後即可結束本次有限實驗。

## 最新接續 20:38：原v6全批與實看QA完成；深度A/B已接GPU，務必等實跑完

原batch79387已exit1，**正常品質未過終局 quality_failed，不是工程failure**，11階段都完成；segWallfail、det bike/motor/light Recallfail、phase004 crop.9438117pass但joint.3990643924未達.4。det v6四輪19200updates completed、best epoch0 ba6d2f4269a7f3f95e4b92d7daff1f330f4203a4619e1f5902c752aa371f7155，mAP50 .7046156223／mAP50:95 .4510487365，checkpoint reload exact。det初輪metrics選例snapshot SHA不同於formal best bytes，正式用completed state，不混用。

原file watcher59456漏帶det checker必要--root而argparse exit2→watcher exit1；原log保持，已用相同未改checker＋明確--root data/sunny-cross-town-v3-007、CUDA_VISIBLE_DEVICES=-1重跑exec3346，exit0，full3600canonical GT/ignore/AP/PR exact。recovery receipt artifacts/runs/phase3-artifact-checks-v6-001/recovery.json；原state waiting為失敗時殘留，不再活。不要再啟watcher。

原postQA45681已完成artifact，**已實看seg9pages/25cases、det5pages/15cases、7bikezoom、phase4pages/31cases，以及3WallFP個別panel和8viewdemo**，visual_review.json已寫原QA root。seg原GPU每例confusion/boundary exact。主要Wall/Fence/Other／rock confusions、ped/bike/motor遺漏、det bike↔motor/ped誤判／小物漏檢、phase unknown顏色誤判。phase no_light_object代表IoU unmatched，不保證畫面真的沒燈。joint matched red→green0不表示GTcrop沒有red→green錯誤。原batch與QA awake均released/targetexit/cleared。整批rawreports/source/export/logs/QA/實看收據封存evidence/phase3/sunny-v3/phase3-recovery-batch-002-completed；det另完整modelcard與failure/recoverylog封存sunny-detection-v6-001-completed-validation。README/domain/registry已更新實際結果，待commit；不改rawQA/catalog/state或parent來源。

**新depth A/B唯一GPU job exec78878／parent23304／ownawake22688持續，已自動通過前批來源守衛並於20:31接續，state verifying_and_loading_data（尚無GPU_preflight.json）**。現在全資料SHA／cache需數分鐘，沒有正式updates屬正常。不能只回覆排隊或等資料就結束；等32步GPU preflight→control/depth各1712updates完整val→exact aug digest/QA→實看並給實際結論。現在原GPU已退出，不要啟其他GPU、CARLA或額外tests，新增44與原50sources及config仍凍結。若實驗fail，保存查因，不盲目重啟。**未自動第二輪**。此前cell671監測已在20:28train-phase轉換時完成，無monitor執行。

## 最新授權與進度 18:40：已同意GPU小檢查＋一輪A/B，等待GPU但勿只排隊就結束

使用者最新「照你建議的是看看吧」已授權先32步GPU/BF16檢查，再同parent的control／depth各1輪，視結果才考慮第二輪。此前「不自動擴大」的舊限制已由此明確授權更新，但**第一輪比較之前不擴到第二輪、不改原品質門檻、不開heldout**。本輪要等實際執行與完整val／QA後回覆，不能把排隊當完成。原det v6第二輪進行（18:37 updates6100，18:39 6500/19200），原batch79387／file checker59456／postQA45681仍活，CARLA關。

新增唯一有界深度對照job **exec78878**：scripts/run_depth_ablation.py --config configs/depth_ablation_v1_1.json --output artifacts/runs/depth-auxiliary-AB-v1-001 --wait-for-batch artifacts/runs/phase3-recovery-batch-002 --wait-for-qa artifacts/runs/phase3-v6-final-qa-001。state waiting_for_existing_batch_and_QA、GPU_work_startedfalse。原生parent23304 creationFILETIME134367935912441472，ownawake22688開始18:39:51，6h至約00:39，parent退自清；不是automation／永久powerplan。**勿重複啟、別改它的config或frozen source**。它會等整個main terminal＋postQA completed_artifacts_pending_assistant_visual_review（GPU subprocess已退出）＋所有舊source/QA/output SHA守衛，才require_idle_gpu開始；最多等6h。若前批failure或任何source/config改動會raise保留，不自動retry。

新增scripts/depth_ablation_components.py：DepthStream依三audited roots讀rawdepth+semantic，SHA核對、train-only、128LRU約141MiB；不把整split floatdepth塞RAM。paired_depth_geometry與原paired_geometry相同Python draws及RGB／semantic運算，另同步nearest depth／valid；padding255/0/false，不縮放公尺。nearest depth loss避免CUDA bilinear backward依賴；**這與先前CPU probe bilinear不同，明記於新design**。4項focused CPU tests過、log artifacts/reports/depth-ablation-components-CPU-tests-002.log；6個真train views跨main/oldscenery/newexpansion的CPU loader／aug/RNGexact也過，artifacts/reports/depth-ablation-real-loader-CPU-001.json。沒有新GPU預檢已跑，不混稱新增GPU tests。

新runner凍結本身／component／原experimental_depth_branch／test／holdawake＋parent所有source，parentbest/config/state/prov/val/frames/stats皆SHA綁定。**queued之後這些都不能再改**（所有SRC、原train/eval/tools及兩套QA source照舊凍結）。深度head constructor原CPU fork會manual_seed_all；runner另包全CUDA+CPU fork並assert RNG無改，再每arm training前seedreset，防GPU公平性錯誤。初始完整3600GPU report需每arm exact parentv6。32 preflight使用兩個原dynamic train groups各4view、batch8，depth-only gradient到encoder、eligible conv BF16（GroupNorm輸出可FP32）、peak≤6144MiB、8view reloadexact；32updates丟棄，重新base0a3fe與隨機head作真正兩臂。

兩組各1epoch/1712updates，與parent同encoder5e-5/decoder.00015、cosine1epoch/warmup100、CE+.5Dice；depth另head.003、aux.2 normalizedlog／.1..120m。全epochhash indices/geometry/actualaugRGB/semantic，必須兩組exact；相同更新量不是等FLOPs。state進度每100，arm train_progress.jsonl，control/depth final.pt包含base/aux/optimizer/schedule/RNG但status **completed_research_arm**，不冒充正式採用。完整val、原gates、nonWall15研究mIoU、每類/scene/margin、checkpointreload後比較。篩選gain≥.002且margin≥control-.01／無新nonwallgatefail只是review訊號，不是quality waiver。選每scene最大gain/loss共最多6contexts，原GPU batch重播matrix/boundaryexact後3pages；需實看。終局completed_first_epoch_AB_pending_assistant_review。**第二輪不自動啟**；有CANCEL marker可按使用者明示停止，無自動budget增長。

CPU準備/新design/contract/rawsources tar封存evidence/phase3/sunny-v3/depth-auxiliary-AB-preparation-001，明確標GPU尚未跑。新codes/config/test＋上輪nonWall解讀README/CURRENT_TASK待本機commit。下一步持續原批與新queue，舊postQA完成後務必看其seg/det/phase全部pages與RGBdemo；新AB在GPU時可看圖片但不能再啟GPU。若新GPUpreflight失敗，保留failure查實際原因，不大量換設定或假稱通過；若比較無收益就停本分支並回報。

## 最新分析：depth對其他IoU仍未證實，不自動擴大

使用者只問研究價值／還需多少輪／假設不考慮Wall是否幫其他IoU，這不是移除原Wall品質門檻的操作指示。重算CPU小測unused_train_probe4張（原模型曾見過、只未參與24步）的15類non-Wall focus mIoU：.7734519193→.7677747363（-0.5677pp）；全部16focus .7792070685→.7739737309。pole .655033→.671695與barrel .788752→.797571升；ped .791086→.772237、bike .731573→.719956降，road .989766→.989628近不變。depthloss下降≠其他IoU提升，而且沒有no-depth續訓控制，不能歸因。已補prototype evidence README解讀，不改rawreports。

回覆方向：工程接法已驗證；對當前IoU的實益未知，無可靠成功百分比。目前正式segv6其他15類及margin都過，若假設不要求Wall則depth對當前Phase3不是必要項目；depth也不會自動幫另一獨立detector的bike Recall。若後續要試，先一次GPU小檢查，再兩組各1輪／最多事前固定各2輪（總2–4）判斷方向，不保證成功、不自動啟動或排程。使用者仍要求少量可行性，不擴大到完整訓練。原batch／postQA仍保持來源凍結、GPU只detector。

## 最新接續 18:19：使用者要求有限depth分支小測，已完成，勿自動擴大

最新授權範圍是「模擬器畫面為主、不考慮實際應用或擴及；先帶入depth分支看可不可用，不大量嘗試，再考慮正式納入」。已完成一次隔離CPU原型，**不要自動啟3424updates/B0B1/fullseg訓練或額外GPU排程**。既有原detector batch79387照舊在跑，18:19第一輪4200/19200updates，postQA45681/file checker59456依舊等原批次，不因小測而取消。CARLA關，所有50main frozen source/3configs/4queuedQAtools再核對0差異，三個prototype來源也exact。新code只scripts/experimental_depth_branch.py、scripts/probe_depth_branch.py、tests/test_experimental_depth_branch.py，不碰任何SRC或正式train/eval。

CPU實跑exec76090已exit0：artifacts/runs/depth-branch-CPU-probe-001。completed v6 best0a3fe...初始化，4fitviews（Town01-00/Town03-00各front/front_left）＋4unused原TRAINviews（Town01-01/Town03-01），固定第一張、所有32PNG/index/manifests/calibs/SHA驗證。共用原MobileUNet dec4 stride4 feature，原forward不變，call期間temporary hook捕捉feature且finally移除；head48→16→1/sigmoid6945參數，FP32；head初始化fork_rng不消耗主RNG。原seg CE+.5Dice+.2depth normalized-log smoothL1，baseLR1e-5/head.003、batch1、無augmentation，24updates，CPU2threads僅23.907s。**只CPU，GPU/BF16容量／隨機幾何整合未測**；不搶現有GPU、不排自動後續。

fit depth loss .117061818→.042774559（ratio .36540146），unused TRAIN .135440683→.043985286；fit AbsRel .80207→.36636/RMSE20.137→13.525m；unused TRAIN AbsRel .92671→.32700/RMSE21.021→14.290m。depth-only gradient head.0583202／shared encoder.0294945，finite非零；8張initial CPU RGB logits exact原父模型、depth bypass相同、checkpoint雙輸出reload bitwise exact、原parentbest bytes不變。proto checkpoint f70eecc8163f13a10ebf926edc4a38534e0c0bf8a38f9e355b351467cc5ce3d4保留ignored，不假稱在Git。沒有val/development/lockbox；所選TRAIN Wall分數雖存於report不可當depth改善因果或跨town品質。

兩頁四個情境已實看並寫visual_review：道路遠近梯度／前背景有學習，但紋理印記／道路條紋／模糊邊緣及遠景錯距仍見。**depth顯示各欄套GT valid mask，黑sky不是模型預測遮罩**；forward仍只有RGB，GTdepth只loss。3項focused CPU tests已過，rawlog artifacts/reports/depth-branch-cpu-tests-001.log及evidence。原型report/contract/steps/before-after/32PNG SHA/sources/testslog/2pages/visualreview/sourceisolation已封存evidence/phase3/sunny-v3/depth-branch-CPU-probe-001；研究plan已加實測補充。可判接法能跑、能回傳梯度及短程學習；還不能判正式值得納入或Wall通過，依使用者小測要求停止擴大，後續若要納入才一次小GPU檢查。

先前CPU Wall-only exec25474亦已completed全3600，alpha0 .153231→alpha1最高.161307，九候選全未過，不值得自動排GPU偏置校正／擴大搜尋。完整負結果contract/report/tool/parentGPU保存在evidence/phase3/sunny-v3/segmentation-v6-wall-only-CPU-study-001；CPU alpha0與GPU原.156453差異不能當模型改善。舊prior／calibration watcher繼續維持取消。

本輪新prototype/test/CPUevidence/研究補充/CURRENT_TASK待本機commit。最新已commit d7e2410為前輪研究／64view depth資料可行性；655fbac為v6seenfit28及原GPUconfusion。最終回覆本輪要簡單說小原型可行，4fit+4其他train/24updates、loss降約63.5%、新增6945params，非正式Wall改善、GPU未測、不擴大訓練，提供小測README。後續接續原Phase3仍先查實際batch/stages及postQA，不能修改凍結來源或把user新scope當放行大實驗。

## 最新接續 18:09：使用者要求先看影片／文獻並提出Wall改進方向

最新使用者明確要求參考 https://www.youtube.com/watch?v=--cGYatMESY 的深度影像處理、框架及訓練，先檢視後提出可行方案，避免Wall問題反覆。這一輪重點是研究與方案，不在進行中的v6改模型。原GPU batch exec79387持續，18:06 det第一輪2700/19200updates；segv6已完但Wallfail。postQA45681與file checker59456仍等待。CPU Wall-only study25474約3100/3600，還沒讀完整grid，不根據partial選alpha。**保持所有model／train／eval／frozen QA source不變，CARLA仍關。**

已用computer-use技能／CUA背景browser實際檢視影片關鍵畫面與燒錄字幕，沒有可擷取CC，不假稱完整逐字觀看。1:52 RegNet/BiFPN/multihead、5:37車體座標warp、7:30時間、9:22 SpatialRNN、11:15離線標註、15:15 DepthCNN/PoseCNN；後者Figure2/3與Zhou CVPR2017原文實核對。primary web另讀PAD-Net、Monodepth2、MixStyle、TLDR、SegFormer官方、RegNet/BiFPN、NetWarp、單目depth捷徑研究，沒有用二手文章作技術依據。瀏覽器tab1暫停，agent-created會turn結束清理，非user原tab。

新增 PHASE3_VIDEO_RESEARCH_PLAN.md 完整改進與對照設計：保留RGB-only、優先小型train-only depth auxiliary；B0同v6init續訓 vs B1只加depth、各3424updates提案，head初始化RNG要隔離、不能假稱equalGPUcompute。depth有助ground/vertical幾何但不能單憑距離或法向判Wall/Fence，反覆相同磚牆不等於外觀缺口補齊。style-only、context/SegFormer與因果時序後續另分支，BEV是Phase4，不能用GT depth/ego pose作弊。depth正確packed米解碼、resize不縮放米、valid masks/K/normal方向、動態遮擋／低視差／5Hz與序列邊界、原gates/heldout隔離、全部GPU驗證寫清楚。**尚未整合新depth loader/head、沒有新GPU訓練，不要把提案說已完成。**

有限可行性實測已完成CPU/file-only（未載模型/GPU），artifacts/runs/phase3-depth-supervision-preflight-001/tool.py/contract/report/input_source_verification.json。原8train groups×8cams固定第一張64view，256PNG及index/manifests/calibs/原inventory與audit綁SHA；.1..120m/排sky/raw0提案mask有效75.8883%，selected Wall1268814px/Fence198575/Terrain485910，距離分位重疊不能threshold直接分類。只是選定train子集，不是full-depth audit或depth模型品質。exacttool與原bytes已封存evidence/phase3/sunny-v3/phase3-depth-supervision-preflight-001。依賴3source追加核對與v6parentexact。主batch50source/3config、queuedQA4tools、CPUstudytool再次0變動，receipt artifacts/reports/phase3-video-research-source-freeze-001.json。

此新plan／depthpreflight evidence／domain doc／CURRENT_TASK待本機commit。上一個655fbac已保存v6seenfit28例與原GPU混淆診斷。完成本輪研究答覆時需清楚說「已檢視關鍵片段與原文」，提供可行優先順序與方案連結，不誇大成影片完整逐字觀看／新模型已訓練。原背景批次繼續，下一次接續先讀當時實際stages/results，再QA與合理恢復；不要直接修改凍結source或重複啟watchers。

## 最新接續 17:50：狀態問答後仍繼續原Phase3任務

使用者最新只問「現在做什麼／六輪做什麼」，已在commentary簡短解釋六epochs是同樣train+val循環、不是六獨立測試；沒有停止原完成Phase3目標。主GPU exec79387與原postQA exec45681、file checker59456持續，不另啟GPU。detv6初始完整3600val已與v5正式validation整份JSON／bytes SHA30d8...完全一致，receipt batch002/detection_initialization_replay.json；sanity ratio .19087849通過並重設。17:49實際第一輪700/19200updates，loss約.208、lr約4.98e-5，無錯誤。全部batch凍結source與config再查零差異。不要以初始化重播當新改善。

新增文件診斷 artifacts/reports/segmentation-v6-wall-confusion-diagnostic-001.json：3600個原GPUframes按scene重聚合exact，Wall TP997032/FP4324560/FN1051136，FP主要other2150166/Fence973196/sidewalk578319/terrain499916；這只是既有GPU輸出的檔案重算。

新增有限CPU fit已完成，exec83605退出0，artifacts/runs/wall-training-fit-v6-001。沿用原train-only外觀／negative選例固定28個indices，原選例兩reports SHA與oldcontract exact；v6completed checkpoint 0a3fe...，CPU FP32 Wall IoU .97738998/P .98702681/R .99010945。兩頁六個最高FP/FN情境已實看，磚石／水泥Wall大致吻合，少量邊界誤差。這是biased seen-train選例，不是全train accuracy或GPU驗收。report/tool/contract/oldselectionreports/2pages/visualreview與原GPUconfusion已封存evidence/phase3/sunny-v3/segmentation-v6-wall-diagnostics-001。高seenfit低val支持外觀／場景泛化優先，不證明單一原因或某方法必成。

Wall-only CPU study exec25474仍跑全3600，17:49約1600，未根據partialgrid決策；別改script、別把CPU當正式通過。PHASE3_DOMAIN_RECOVERY.md與TRAINING_PROTOCOL已補真實v6／各種檢查範圍，歷史準備狀態清楚標時間。沒有更動任何SRC/model/training/frozenQA code。後續仍待det4輪→phase004/joint→conditionalheldout/demo/bench/export→queuedGPUQA實看；不要因問答／此存檔而結束。

## 最新接續 17:36：segv6完成、detv6進行中，仍勿結束

**目標仍盡量完成整個Phase3，不是回報完六輪就結束。** 唯一GPU主批 **exec79387 / parent25188**（FILETIME134367844292432653）不變，batch002所有model source/config依然凍結、CARLA關。segv6 **六輪10272更新已正式completed**、best_epoch5，bestSHA **0a3fe3f66fb93be373f2e82218632a4f8cc277bf6067f48bb4683a71f1b2af30**；last3b32c59a56283da92614566f5fed539f268bfdf3342cc4f88403ade66dcd60e6。mIoU **.6065211770678386**／Wall **.15645293506956517**（P.18735596/R.48679210）／Fence.6057671054／margin **.6222117516687405**，**onlyWallfail**，其餘15focus類及margin都過。selected reloadexact，GPU allocated4046.51MiB/reserved4662，trainingwall4922.844s。validationSHA0aad5062137087b317eae07504d30113a73182451a6b89041da71de1dba5eaf1，framesSHAf94c3d58e762caf62980cf63c52766249ae6d2ae39f019da3b9c8dad8145fb44。3600frame文件一致性checker已passed（不是品質pass），artifacts/reports/segmentation-v6-artifact-check-001.json。完整modelcard/config/provenance/epochs/reports/source_raw_tar在evidence/phase3/sunny-v3/sunny-segmentation-v6-001-completed-validation，README/registry已更新但尚未commit。GPU錯例仍等整批結束postQA，不假稱CPU圖是原GPU。

**主批17:31:13已自動train-detection**，啟動器8840、實際Python **14408**（父8840），17:36CIM確認存活CPU26.23s/WorkingSet607MB，GPU208MiB。仍full main metadata/86400PNG SHA驗證、尚未runDir或optimizer更新，這個驗證成本數分鐘，不要因為無progress就重啟。stage log為batch002/train-detection.log；之後cache/80sanity/reset/初始val/4epochs19200updates，當前模型initializer ec142...v5、720resize、bike-positive p2/3 halfzoom、不改score/gates。其後phase004/joint/demo/3bench/export同批自動接續；seg fail故全tuple新heldout保持未評分。

**CPU/file checker exec59456**已seg passed後等detcompleted，state artifacts/runs/phase3-artifact-checks-v6-001/state.json。**post-GPU visualQA exec45681 / parent23100 / ownawake30184**仍waiting整個batchterminal，GPU空出才replay segmentation原batch+saved det+phasecontacts。兩個job別重複啟。QA source工具（review_sunny_segmentation.py/run_phase3_final_qa.py/review_saved_detection.py/holdawake）已自16:50凍結別改，不然queuedjob會拒絕；全部SRC/train/eval/export/batch/config也保持原contract。主awake至22:07、postawake至22:50、都自parent退出清除，不是排程/永久powerplan。

**新增CPU-only有限診斷 exec25474**：artifacts/runs/segmentation-v6-wall-only-CPU-study-001，log artifacts/reports/segmentation-v6-wall-only-CPU-study-001.log，17:35已300/3600views。**CUDA_VISIBLE_DEVICES=-1**，checkpoint0a3fe...completed parent且僅Wallfail、無heldout才能啟；每個selected RGB/semantic/instance驗SHA、main/index/trainingstats綁SHA。script scripts/study_segmentation_prior.py新增 --wall-only：九個事先固定alpha0,.125,...1，臨時logits只扣Wall通道12的alpha*log(train CE classweight)，其他16通道保持原值；沒有改checkpoint、模型選擇或正式GPU驗收、沒有開heldout。這是查Wall誤報與class-weight偏好關係的診斷，不能直接宣稱機率已校準或Phase3通過。2項focusedCPU bias-mask/invalid-support tests過；所以已實跑仍151GPU＋QA3focusedCPU＋此2focusedCPU，**不能說156GPU全跑**。原舊all-class prior/calibration watcher仍取消，這是新模型的Wall-only CPU診斷，不要重啟舊GPUwatcher。

CPU study source也已載入/記SHA別改到它退出。此script/test在model source whitelist之外，沒有改任何SRC/model code；目前新study2tests、完成seg archive/README/registry/CURRENT_TASK待commit。若CPU結果有實質改善，真正採用仍必須等**整個GPU batch＋postQA都退出**後，在新版本上完整GPU候選評分／exact alpha0 replay／重載／來源與GT/params檢查，不能以CPU結果當通過。舊calibrate_segmentation_prior.py現只重算main+舊scenery，**不支援newtraining_expansions／Wall-only**，不能直接拿來跑v6或無聲忽略新data。任何新準備先確認來源契約/所有門檻與heldout隔離，不能為完成Phase3下調門檻。

中間epoch快照保持ignored local：segmentation-v6-first-epoch-diagnostic-001 c148...（第一輪）及segmentation-v6-epoch-snapshots-001（第三輪last、第二輪selected299d...、第四/五輪last與reports）。不是final；真正v6final用上面的0a3fe。繼續等det結果，最終全QA/模型卡/收據/registry一併封存，不能只啟動就結束。

## 最新接續 16:52：三輪過半，最終QA已串接（仍不要結束）

**GPU批次 exec79387/parent25188不變，只有segv6在GPU，CARLA關。** segv6第三輪更新5100/10272，16:51進入第三輪fullval（progress5100停住正常）；已完成第一/二輪，第二輪selected mIoU.6013262154、Wall.1383076656、margin.5801831016，onlyWallfail。SRC/batch sources再次0差異。det v6/phase004/joint/demo/bench/export後續待同批自動跑，全部val過才heldout，勿手動開。CPU/file checker exec59456仍等待seg/det completed，沒有GPU。

新增**一次性post-batch QA exec45681**，script scripts/run_phase3_final_qa.py，output artifacts/runs/phase3-v6-final-qa-001，目前statuswaiting_for_entire_GPU_batch_to_finish。原生parent **23100** creation **134367870389417172**，ownboundedawakehelper30184最多6h至22:50，parent退自清；和原batch6h awake重疊確保交接，未改powerplan。**別重複啟動、別再改review_sunny_segmentation.py/run_phase3_final_qa.py/review_saved_detection.py/holdawake或其contract工具，否則post-QA sourceguard拒絕。** 此job必須等整個batch terminalstate+exportSHA、allbatchsourcesexact才啟GPU，然後分割GPU原batch重播→savedGPUdetgallery→savedphasecontact4pages，輸出只pendingassistantreview，沒有training/新heldout/排程或calibration。完成後務必實看pages與demo，不把code/圖片產生當通過。

review_sunny_segmentation.py新增 --device cuda --all-classes（defaultCPU保留）。CUDA取原batchsize與index分界，核對每個replaybatch RGB SHA／selectedGT SHA，模型源/權重/config/validation來源全綁；selected confusion及boundary counts需exact符合savedGPU才產圖，否則raise不假稱原輸出。選worstWall/Fence/margin各scene＋16focus各類＋Wall最大FP；margin overlay改用同metric的ignore鄰域。新增3項focusedCPU tests全過（batch界線/短尾、ignore鄰域、類別與Other-only FP選例）；**151actualGPU是此前整批，新增3只CPUfocused，不能說154GPU已跑**。phase_contacts對既有corrected32張phasepng hash/尺寸與4contact試做通過，證據artifacts/runs/phase3-final-QA-preflight-001；真正v6 GPUreview尚未跑，不新增prototype模型。

CURRENT_TASK與上述QA工具/3tests/evidence尚待本機commit；模型training source完全未變且baselineinit與v5報告exact已保存。第一輪c148...immutable中間checkpoint在下文指示root保存，非v6final。繼續等整批結果，不回答狀態就結束。先前取消的prior/calibrationwatcher不要重啟；若v6完成仍fail再看實際結果，任何新GPU恢復要等post-QA全退出才做。

## 最新模型進度 16:42（active batch不變，仍勿結束）

使用者剛要求簡單說明目前工作及六輪內容，已在commentary解釋6epochs不是6個不同測試；每輪13696sampled train views/1712updates、固定3600val，繼續原工作未停止。**唯一GPU exec79387/parent25188，file-checkwaiter exec59456；全部source/config仍凍結、CARLA關，不啟其他GPU。**

segv6已完成2/6epochs、正式3424updates，目前第三輪progress3600。第一輪mIoU .586257621／Wall .134311121（P.15259/R.52857）／marginF1 .594962337，onlyWallfail；第一輪immutable模型copy保存在artifacts/runs/segmentation-v6-first-epoch-diagnostic-001/first-best.pt，SHA c148af6a2cd691b629211afcdc9b6dc53e31499e73d762acad324b52f94cca6d，config/provenance/first-report/receipt齊；它只是中間checkpoint，非完成模型。FP主來Other2787678、Fence1097228、Sidewalk1003640、Terrain918583。不要混成v6final。

第二輪mIoU **.6013262154438384**／Wall **.13830766560271235**（P .1659045103/R .4539896141）／margin **.5801831016244913**，目前selected，仍onlyWallfail，其餘15focus IoU及aggregate/margin過。每輪約715秒；首次完整initial_validation與v5父代validation整份JSON exact，收據phase3-recovery-batch-002/segmentation_initialization_replay.json（非新改善）。80步sanity ratio .4250342 pass/reset，151/151GPU tests全過、actual2train-groups CUDAfinite及720capacity exactreload已過。後續det v6/phase004/joint/demo/bench/export仍待同一批自動接續；不能只回答狀態就結束，等整批與QA。

## 最新接續 16:16：正式 v6 批次已啟動，務必等完（優先於全部歷史）

**使用者要盡量一次完成Phase3，不要在此結束。** 唯一GPU batch **exec session79387**：`scripts/run_phase3_recovery_batch.py --segmentation-config configs/sunny_segmentation_v6_1.json --detection-config configs/sunny_detection_v6_1.json --segmentation-run artifacts/runs/sunny-segmentation-v6-001 --detector-run artifacts/runs/sunny-detection-v6-001 --light-run artifacts/runs/sunny-light-state-v1-004 --output artifacts/runs/phase3-recovery-batch-002`。原生parent **25188**、creation FILETIME **134367844292432653**；awakehelper23960綁parent6h至約22:07，active，parent退自清。**所有SRC與batch contract來源/config已凍結（最新0差異），訓練中不要改SRC／train/eval/export/batch／config，不啟CARLA、不加其他GPU任務或舊calibrationwatcher。** Git **0fe1a6e**（前a396669）。

已完成**151/151實際GPU tests、0skip、0fail**，12.126s；720det容量也passed，allocated3377.50MiB/reserved4226MiB，exact production reload；含兩真實原train groups一個halfzoom/一個unitzoom，GPUfinite loss.4594409/norm5.67114，dummy更新不帶正式train。此容量report根scope仍legacy稱Synthetic，nested real_train_geometry_preflight才是實際真資料補充；別在activebatch改源。

**當前stage train-segmentation**：子啟動器31616、實際Python **26368**（nativeCPUmemory讀取看後者，前者0CPU正常）。主資料＋supplement完整SHA驗證後，快取decoder約6min，16:15已完成13696train/3600val載入；old1024repeat2/new1024repeat2均2048sampled、各128staticanchors。80步overfitratio **.42503420364 PASS**；sanity已重置。初始GPUvalidation還在跑，16:16尚無initial_validation.json／正式updates；不要把快取階段當hang，也別把sanity當正式模型品質。接著6epochs/10272newupdates，預估約1.5h，再det4epochs/19200updates約3h；時長是前輪估算，不是保證。**不能僅啟動就結束**，等兩model＋phase004/joint＋conditionalheldout＋demo/bench/export，再看實際QA及逐類結果。任何valfail仍不開heldout、不降低gate。

另有**exec session59456**一次性CPU file-check等待程序：artifacts/runs/phase3-artifact-checks-v6-001/contract.json/state.json，log artifacts/reports/phase3-v6-artifact-checks.log。先等seg completed跑scripts/check_sunny_segmentation_artifacts.py→artifacts/reports/segmentation-v6-artifact-check-001.json，再等det completed跑check_sunny_detection_artifacts.py→detection-v6-artifact-check-001.json；CUDA_VISIBLE_DEVICES=-1、6h bound、無inference/heldout或排程。不要重複啟動。

新formaldata v2-001已audit＋4pages32contexts實看批准、實際loader exec70869exit0讀入1024/repeat2=2048且supportexact。auditSHA326cd9fda7d36d598b4cad880b483a3bb962fbed2bf192e003687eb23afcc18c；reviewSHA30ab42a79ae6ff032aaceadb9091130c1f3b93c7f973a688e1646f644d01956f。資料meta/source原bytes tar.gz、audit/review/plan/QA/nativecleanup與actualLoader收據已Git封存 `evidence/phase3/sunny-v3/segmentation-expansion-v2-001-approved`。Wall8191123pix、Terrain31118819pix；外觀仍主要磚石牆，不能說補到木實心Wall或高速擋土牆。

低牆距離漏選已實查：49native低牆距道路spawn47..69m，故newCLI80m候選；4anchors32views probe的2pages8contexts實看仍前景woodFence＋遠薄Wall，**不加入新train**。native-server所有capture/extra-probe/audit exec13481/57290/99020已exit0，server exacthandle停止，awake23744released/requestclearedtrue，GPU閒置讀數316MiB後才啟此batch。

下一步：持續監看進度，適時把初始val與父代selected做exact aggregate核對；各模型completed後統一checker、val錯例需actualGPU QA（不要用CPU BF16差異圖冒稱原GPU），最後phase004 jointfix結果／RGBdemo／3bench／modelcards／README/registry更新。若全val過batch會按凍結規則一次heldout；若fail繼續根據真結果恢復，別宣稱Phase3完成。原v5各結果與legacy jointprecision→v2addendum均已a396669保存，不覆寫。

## 最新接續 16:05（優先於全部歷史）

**勿結束，使用者要盡量完成整個Phase3。** 新formal `data/segmentation-expansion-v2-001` 已1024views/128anchors/4096PNG，全auditpassed，4頁32contexts實看批准。auditSHA326cd9fda7d36d598b4cad880b483a3bb962fbed2bf192e003687eb23afcc18c，reviewSHA30ab42a79ae6ff032aaceadb9091130c1f3b93c7f973a688e1646f644d01956f；Wall8191123pixels（349views>=512）、Terrain31118819（1021views），0retry、maxpose7.9699e-5、maxresolvedclassdisagree4.34e-6。RGB/GT實看含磚石牆、岩面/草地、woodrail/picket/meshFence、路界；仍沒有確證新木實心Wall/高速擋土牆。Audit exact source隨root/audit_source保存。

新plan v2_3把nativehint距離40→80m，因49個1.4m矮牆距native roadspawn47..69m，全被原40m略過。低牆probe `data/segmentation-expansion-v2-town06-low-probe-001` 4anchors32views完成，2pages8contexts實看只是前景woodFence+遠處薄磚石Wall，不是新木實心Wall，因此不加新formaltrain。review在artifacts/runs/town06-low-preview-review-001。exec13481/57290/99020都exit0、所有capturecleanuptrue。

**CARLA已關**：exact handle核對PID8064、creation134367812249032916、fullimagepath後TerminateProcess/Wait；stopreceipt artifacts/runs/phase3-expansion-server-001/stop.json。沒有CarlaUE4 process，VRAM316MiB。nativeawake應自動released，待核對。下一步可GPU。

新seg配置 `configs/sunny_segmentation_v6_1.json` SHA b2a3ca820bc73bd26267b2be09cde25d630bfd41a99aaf227cc473bcec6d770d，從v5best3fe7...，原scenery1024repeat2+新1024repeat2、main9600/13696=70.09%，6epochs/10272updates，架構/loss/幾何/photometry不改，encoder5e-5/decoder.00015。det v6配置前述ec142parent+conditionalbikehalfscale2/3，4epochs/19200updates。phase004重跑以凍結jointignorefix。**尚未啟GPUbatch002**，先等real-loader exec70869完成（002預期actual1024讀入repeat2=2048，support exact）；舊001wrapper SimpleNamespace無__len__錯誤非loader錯誤，已改fixture。

原commit a396669已保存v5全批與修正/新source，815evidence raw bytes對Git全部exact。之後newlow-preparer/config、新segconfig和實際formaldata證據待封存commit。然後`run_phase3_recovery_batch.py`用segv6/detv6/light004/newoutput phase3-recovery-batch-002，一次full151 GPU檢查+720capacity（含2真實train groups成對縮放finiteCUDA）+兩主要模型+phase+joint+heldout conditional+RGBdemo+3bench+export。**不能在只啟動batch時結束**；繼續等完成、看QA與quality。全部SRC凍結直到此batch退出，不再新增未用備案。6h awake由batch自己綁parent，舊prior/calibrationwatcher已取消，不重跑。

## 最新接續 15:54（優先於下方歷史）

仍持續完成Phase3，勿結束。原v5批次完畢品質失敗；此刻CARLA PID8064 exact FILETIME134367812249032916仍活，沒有DL GPU。Town06第二preview exec80942已exit0、review57751exit0：64views，6pages全實看，1272073Wall pixels/54676Fence，多位置磚石高低牆，未證明木實心Wall/擋土牆；原raw4/5身份audit兩頁5案例實看確認木圍牆/擋土牆確為Wall，先前主要診斷未逆轉。Town06兩次probe review已寫，probe不可train。舊batch八RGB surround也已實看，收據寫並收入evidence。

**正式capture exec99020仍進行Town06**，plan configs/segmentation_expansion_diverse_v2_2.json SHAaff1cc0dc406830104ae50c960f2bc77b9a8d8488ea873e8c43604459e1c44ff，64anchors/town、root data/segmentation-expansion-v2-001。**exec13481已啟動批次等待：Town06 passed/cleanup後才prepare Town07→capture Town07→audit**，log artifacts/runs/phase3-expansion-server-001/complete-acquisition.log。別重複切圖／capture／audit，等13481退出後實看QA並bound review批准。report預設statusfailed/cleanupfalse代表進行中，不是已失敗；只有cleanuptrue及error才是終局。所有源快照獨立保留，不改收過probe。

正式v2 loader src/senior_ai/segmentation_expansion.py已接train/evaluate/export，綁main/unused-map-policy/plan/audit/review/allbytes/fullmatrix/F+1/probe排除，只分割train。原v1限制不變。src/senior_ai/detection_augmentation.py已接conditional bicycle-positive halfscale p2/3，同步所有GT/ignore、paddingignore、無flip，epoch記actualcounts。configs/sunny_detection_v6_1.json已凍結（從v5best ec142...、4epochs/19200newupdates），尚未CUDA finite/capacity/正式訓練。verify_sunny_capacity.py增加兩真實原train groups SHA-bound GPU geometry finite check與原3 synthetic/reload。**CARLA停後才跑GPU**。

新增6項CPU tests；full006=151 total/148pass/3GPUskip/0fail（005三項因使用系統sandbox temp cleanup拒絕，改workspacefixtures後focused及full006過，非模型失敗）。還沒新fullGPU。新segconfig需audit/reviewSHA後才建：parent3fe7...、舊1024viewsrepeat2+新1024repeat2=4096static、main9600/13696=70.09%，6epochs/10272updates，同既有架構loss/aug，先data-first不加未用MixStyle。phase004需重跑帶jointfix。新batch002 sourcefreeze後不再改SRC；全val過才heldout，失敗不lowergates。

最新README/registry已更新v5完成，batch原legacy precision的v2recount addendum保留。新sources/QA/evidence仍待commit；evidence/training-expansion-preflight-001只CPU/probe證據，非formaldata已批准。下步：等採集稽核→QA→訓練→統一驗證，不能只停在工具完成。

# 目前任務：批量完成晴天 Phase3（2026-10-18）

## 最新接續：v5全批完成，正在實際補景（優先於所有下文）

**使用者要求持續盡量完成Phase3，勿在此結束。Phase3仍品質失敗。** 原exec32462已結束：所有11 stages return0、最終state=quality_failed（wrapper exec回1；Python設計quality failure回2，不是stage crash），無failure.json。原parent19772／det6608均已退出，awake4644 released/request_clearedtrue；functions monitors367/372都結束。現在**沒有DL GPU訓練，CARLA正在使用GPU，別啟任何CUDA訓練／benchmark。**

### 正式v5結果與驗證

- segv5 best SHA3fe7bd08d885e481b7435fdf466945f142f8323447cca46bf31c32455835a53f：6epochs/10272updates，mIoU .59002533、wall .09379872／margin .49446801未過，fence .61474629及其餘15類IoU過，selected reload exact。已實看3頁8情境，原證據保留。
- detv5已4epochs/19200updates完成，selected第3輪（epoch2），mAP50 .7085198797911534／AP50:95 .4531501647391283，只有bike R .3374485596707819未過（P .6648648649、小框22/388），其餘7類P/R過。第4輪AP .4500293較低所以沒選。best SHA **ec1427214d5fc3e51b9bbb2fed652fa762e97edeb34cfc23a989bdc879c8575f**，last d9bfade775f0e38a58977c68fecf251ea1936df1cf6c025cdb264abb20448407。3409.32MiB allocated／4418reserved、reload exact、RPN/ROI ignore各19200。validation SHA30d8ebc4dbd4b2b7c230b3936817af498936a077286d86e52b4f55e4e095f59b；framesSHA23fca48988331744cb57689f5b4d041cc54d7eb39cf224cf66dd1c432e4de550。
- det file-only全量checker exec54597已exit0：3600 canonical index/GT/ignore/float32 metadata、full/by-scene/size/truncation AP/PR全部重算exact，quality只bike R fail。artifacts/reports/detection-v5-artifact-check-001.json。scripts/review_saved_detection.py已用saved GPU predictions產生run/validation_errors（exec17516 exit0），**5pages/15contexts＋3小框zoom已實看**，visual_review.json/input_inventory_binding.json/tool_used.py保留；並非新CPU推論。近處bike→motor/person、遠小/遮擋/截斷漏檢、truck局部重框、barrel→bike均見，不能只說全是小框。初版gallery外接核對image_inventory SHA正確；目前tool已內建該check，tool_used.py才是此次exact source。
- phasev1-003完成16epochs/832updates/7.516s，best SHA **c2709f15f14b193b68d0cb35d5cec37713f5e1015dff60f42885b85afb5f0846**，GTcrop macro .94381170398／所有crop gates過。actual-det-box joint macro **.46670520113725555**／matched red→green0，joint gates也過。**所有新heldout仍未開**（seg/det val fail）。
- batch rgb-only demo＋3 benchmarks＋export完成：artifacts/runs/phase3-recovery-batch-001/export/summary.json，phase3_completefalse，SHA cccfe2c0b16db97b9281cd62f4a7917d815a939d7579084f8109b62d155722c2。8view integrated warm p50 699.08485ms／p95 741.78373，排除sim/render/control，不是20Hz。**surround.png還沒實看，需看並寫review；新det/phase完成證據、README/registry/modelcards尚需整理封存，不能只報工具完成。**

### 發現並修正真的joint報表bug（canonical source已變，舊模型快照不改）

原joint把ignored/unscorable區域也列false-light，331 vs detector FP42不一致。新增src/senior_ai/light_joint_policy.py，修evaluate_light_state.py（positive優先，wrong-class/duplicate over任何known正例仍FP；只有背景且ignore才排除），supplement也帶ignore_boxes，gallery不選ignored當FP。3 focused CPU tests過。**這使canonical source不再是8ec；不要用舊batch --resume，也別拿current evaluator去假裝匹配phase003 source。下一輪重跑phasev1-004（小網路7s＋資料檢查）凍結修正源，不盲用phase003舊frozen evaluator做最終heldout。**

scripts/recount_light_joint_ignore.py已CPU重算既有saved records（exec5653 exit0）：289ignored移除後42FP，精確符合det AP50；原matrix/GTrecall/red→green全部不變，macro仍.4667052、quality仍pass。原run reports不覆寫。輸出artifacts/runs/phase3-joint-ignore-recount-001，**4張contact／32個修正後error contexts已全實看**，visual_review已寫。需把此addendum和original batch一起封存，precision改善不是模型改善。最新全CPU suite仍139 total/136pass/3skip；之後新增checker3＋ignore3，尚未完整145合跑；GPU完整仍只有先前94。

### 現在native server與即時工作

- 新隔離root artifacts/carla-phase3-expansion-v1內容驗證已完成（收據/manifest在002與Git6410f90），原root保持原bytes；native啟動也已成功。**shipping PID8064，creation FILETIME134367812249032916**，launcher31472，DX12/Epic port2000。identity與launch在artifacts/runs/phase3-expansion-server-001。awake nativehelper23744（launcher26436）綁此確切instance，最多2h至**17:14 Taipei**附近；有需要可另建有界helper，別改global power plan。操作/停止前重查CIM與creation，不能盲殺PID。
- client/server都是0.9.16，ClearNoon cld5/sun45/rain0。Town07和Town06都已真正完整load成功（不是只看available_maps）。Town07 OpenDrive **be730652769e993e27f748ba5793dba9e49d299205146a7a8db0bcd17af877bf**，116spawns／279Fence／0Wall。Town06 **a90b613175fa837fdc128ddfaf5ddad25797b128fdbd8cd945974bc21a60496e**，436spawns／922Fence／618Wall。
- metadata scoutruns train-expansion-scout-town07-001／town06-001都completed；prepare_reserved_world.py只load一次、fresh client實看after exception，今回兩次正常成功，source也待commit。
- Town07 probe已完成：data/segmentation-expansion-v2-town07-probe-001，4anchors/32views/128PNG，plan configs/segmentation_expansion_town07_preview_v2_1.json SHAb3f585279663af8ec533a7ca8a1e34d9754d15377c783d81a456b13821b8869a。scripts/review_scenery_probe.py實檢CRC/frames/pose/classes，output artifacts/runs/town07-preview-review-001，**3pages/10contexts已實看**：0Wall pixels、105894Fence；木製兩道欄杆、白picket、岩坡/草地/林地。岩坡是terrain/ground，不能把Fence改Wall；可當後續negative來源但probe不可train。visual_review明確training_approvedfalse。成功F+1當時只source-method綁定，沒有explicit counter（報告0）。
- **唯一active capture：exec48269**，collect_unused_scenery.py --town Town06 --probe-anchors4 --plan configs/segmentation_expansion_town06_preview_v2_1.json --output data/segmentation-expansion-v2-town06-probe-001。plan SHA89e8b81ea902aa9aa4ec107d1f13145740f4a7dfd8a911af8c464175ca5ed95d，64候選錨點。**先等此job退出，別改其source/config。** 本版collector已增加每筆成功readback_frame=F+1與report.explicit_success_readback_framesTrue；auditor新正式v2強制檢查（舊Town07 probe不供train，不修改）。

### 接下來真正要做（不要再無限寫未用備案）

1. 等48269完成，跑review_scenery_probe.py對Town06 root輸出new artifacts/runs/town06-preview-review-001，實看RGB/GT材質，決定是否有新Wall-positive＋rock negatives。Town07只有negative，若合收Town06+07，正式audit的Wall/Terrain quota按完整集合驗，不能單用Town07零牆冒稱補牆。
2. 視preview再凍結正式新plan（目前兩個只是preview候選），完整capture另root（不重用probe）、audit_unused_scenery.py、實際QA/training_approved review，全都通過才train。原v1 Town01/03 collector/auditor/loader限制完全不放寬。新v2 train loader **尚未實作**，必須正確綁main＋所有static roots/SHA/倍率與sampling，main約70%，不當新動態路線或新的independent episodes。
3. 關閉並核對此owned server、awake清除後才GPU tests/capacity/training。det下一輪優先對bike-positive train views 2/3機率scale.5、其餘原尺度（幾何prototype已CPU real finite，但未整合／GPU）；不先疊P1/flip/budget/score變更。parent為正式v5 ec142...。seg parent3fe7...，新data先於MixStyle備案；六epochs及data採樣待實收support後凍結。phase重跑004帶修正版joint源。所有原品質門檻不降低，全部val過才一次heldout。
4. v6 batch需解決各模型source provenance／frozen evaluator相容，不能為重用run就刪source checks。原完成run不能resume。current source已有joint修正，與v5父代不同屬正常新實驗，initial inference/loader/reload須exact核對。

最新Git6410f90；上述v5 QA／new sources／Town6/7 probes／joint修正等尚未commit。資料roots/artifacts/runs都ignored，本機證據要封存明確scope與SHA。所有scope仍只有理想晴天Phase3二維，不開Phase4。

## 最新接續：14:49（優先於全部歷史）

**補景runtime準備exec3278已exit0！** 002 state=completed_content_not_native_validated，31427原base檔全部clone、22592 addon CRC/SHA、42275新runtime檔final驗證，原base所有來源重新hash完全未變。只新clone替換5724共用不同assets、新增10848；patched shipping SHA仍b0370d6dcd57b95299d12005cebad1dc6187804e83aeb51a184ccbc4b581fea8。contractSHAa13ec3ecdad270ae52be4adf8cca411742486d4afa048de8bec2c9dbe9c8a080，runtime_inventorySHAd36ae28ee4609509067fa430426600f5d66aa13b98bca97b956f47f009270bf7。來源／全manifest壓縮原bytes與完成收據已封存evidence/phase3/sunny-v3/additional-maps-isolated-runtime-002。**native尚未啟，仍等GPU批次全退出才開新root；此時不要重跑prepare job或把content完成當map可用／新資料已收。**

原GPU exec32462仍最後第4輪，約17700/19200updates（14:47），source contract再次0差異；後續phase003／joint／demo／bench／export還沒完，不結束本回合。monitor cell372活，主要nativeIDs仍19772／det6608（執行前重查）。PHASE3_DETECTOR_GEOMETRY.md另記純GT尺度投影備案：只bike-positive views，2/3的.5縮小／其餘原尺度，小框期望52.99%，未整合或正式訓練；不先加P1／flip／proposal數。待現有全批結果與實際補景後才凍結下一輪。

**原GPU仍執行、不要結束回合。** 第三輪14400已正式驗完AP50:95 .4531501647391283／mAP50 .7085198797911534；七類P/R通過，bike P .6648649／R .33744856、小框22/388，仍唯一bike recall失敗。最後第4輪約15800/19200updates（14:32）；phase003及其後全批步驟仍待接續。磁碟job exec3278已完成31427base clone，正overlay22592addon（約16512），尚未final hash／原base不變核對；仍不能啟CARLA。

新增scripts/check_sunny_detection_artifacts.py：對completed run做weights/source/config SHA、全部3600 canonical index/GT/ignore/float32序列化欄位、逐類/逐scene/size/truncation完整AP/PR重算exact、選模與update counts。已在舊detv3完成全量CPU/file-onlyrecheck exit0（exec76394），結果只有舊ped/bike/light R品質失敗，沒有新推論／heldout。證據detection-artifact-recheck-v3-001，3項focused tests全過；最新全suite仍139，不是142。**v5 completed後務必用此工具再查新run**，不能在training state先跑。原legacy v3尚未抽出perception_quality.py，checker允許該舊模組缺席但metric/data source仍exact、門檻照原config，不放鬆品質。

目前read-only monitor **functions cell372**（每45s讀磁碟status）；舊367已terminate，只停監看不影響nativejobs。沒有校正watcher／新GPU模型。git最新e1581a2後還有上述新checker/recheck待commit。

**持續完成整個Phase3，不要結束為完成。** 原唯一GPU batch exec32462/native19772（det native6608）仍活；det第三輪約14300/19200updates，14400後才完整val。第二輪仍只有bikeR .327846未過，其餘七類P/R過。seg v5完成但wall/margin未過；phase003／joint／統一val／RGB demo／3種bench／export待原batch串接，全部新heldout仍未開。

官方額外地圖curl exec46320已exit0，完整檔仍命名AdditionalMaps_0.9.16.zip.partial，size7250055782／SHA709a5cb4d870668519dc27f0e0104e0419cd76adb75da2c232d74ddcc311423f；inspect exec85244已exit0、22592files／23647entries／19.116GB展開。源與manifest gzip在additional-maps-inspection-001。**11744共用路徑，2686size不同，517個Engine/plugin assets，不得覆蓋原artifacts/carla。**

**新的唯一磁碟準備job exec3278**：scripts/prepare_additional_map_runtime.py --destination artifacts/carla-phase3-expansion-v1 --output artifacts/runs/additional-maps-isolated-runtime-002。14:17已hash凍結contract並開始clone，6001/31427base files附近。它不啟CARLA／不碰GPU；完成後還要overlay CRC/SHA、重驗新runtime與原base全部bytes，state必須completed_content_not_native_validated；僅內容完成仍不代表native可啟動。**此job運行中別改prepare_additional_map_runtime.py／install_additional_maps.py／src/senior_ai/core.py／sunny_data.py**（自身contract凍結）；原GPU source全部SRC仍保持8ec5771 bytes。

舊準備exec71660/native30868已exit1；001在preflight拒絕Engine/plugin，尚無contract、未建立destination，狀態／exact tool保存；**不要重啟舊PID或用--resume001**。002現先驗所有實際22592 paths，再做hash/clone；sourceSHA3ae833b7d700c2f5de5087ff8a1380bba09d18868a648bd5d23300449e4e5d73。原stockpatched binary b037...不變，原CARLA仍關；下一次須等原GPU批次所有工作退出，再用start_carla.ps1 -CarlaRoot C:/code/senior_AI/artifacts/carla-phase3-expansion-v1啟DX12/Epic，只先native availability／scout，若runtime實際載入不相容保存失敗回原root，不可說已有新資料。

新v2備妥prepare_unused_scenery.py／collect_unused_scenery.py／audit_unused_scenery.py，原v1 collector/auditor/loader bytes完全不變。新plan只unused Town06/07，排除所有原split家族/Opt/OpenDrive別名，保留rig/ClearNoon/4模態/F+1actual frames、pose及raw class規則；source/runtime metadata隨data保存。capture核對新root launch／binary／DX12/Epic；probe不能train audit。**仍無新native inventories、RGB、正式plan、audit/review批准，v2 train loader未整合。** 下一步先probe實看木質實心Wall／水泥分隔與岩面negative，不以asset名稱斷言材質。PHASE3_DOMAIN_RECOVERY.md更新完整路徑。

最新完整CPU suite exec60663已exit0：**139 total／136pass／3GPU skipped／0fail**，14.969s；active batch source差異0，證據cpu-regression-20261018-004。包含4 runtime與3 anchor tests；source guard其後只加rig==immutable config，正式使用前focused/最終檢查。原94 GPU preflight才是目前實跑完整GPU證據。139CPU≠139GPU。Native工具新source/preflight證據已保存，不要為等待訓練繼續無限寫未用備案；完成實際data／training與品質恢復。

## 13:55歸檔與地圖準備補充（仍在同一批次）

commit ac713fd修正真正的Git可恢復性問題：411/573個舊evidence檔案因CRLF normalization，Git blob和實際SHA-bound bytes不同。`.gitattributes`新增evidence/** -text，重新存原始bytes；573全部對committed object核對exact，證據git-evidence-byte-preservation-001與scripts/check_evidence_git_bytes.py。只保留原bytes，沒有重算模型分數；active source contract再次全部exact。

installer最新使用staged已驗CRC/SHA檔案→同目錄.pending副本（實際suffix.partial）→Windows non-overwriting rename，防中斷留下半份.umap。6項CPU工程測試全過，包括現有不同檔必須在任何安裝前失敗、保留原binary及模擬第二檔發布中斷後用新receipt run恢復；最新證據additional-maps-installer-preflight-003。001硬連結受sandbox拒絕的失敗已保留，002是5項版本，勿用001當成功或實際安裝。未做132項整批suite；最新完整仍129 CPU／3GPU skipped。

額外地圖curl exec46320仍下載（13:53約5.26/7.25GB），原batch exec32462det第三輪11300updates附近；不要重複下載／重開訓練。尚無native Town06/07 availability或RGB預覽。Phase3尚未通過，繼續原batch與必要修復。

## 最新接續點：13:48（優先於下方歷史）

唯一原GPU batch exec32462/native parent19772仍跑；det native6608已進第3輪（epoch2，約10400/19200updates），第2輪9600驗證完成AP50:95 .44521858、mAP50 .71195058，仍只有bicycle recall失敗（P .959839/R .327846、小框9/388）。其餘七類P/R仍過。所有SRC／原batch contract檔案hash再核對完全未變。seg v5已完成但wall/margin未過；phase003/joint/demo/bench/export尚待原batch自動接續，全部新heldout仍未開。不要提前結束。

CPU完整729自行車RPN調查exec25563已exit0：原budget覆蓋651/729=.893004、較高budget686/729=.941015；小框352/388→371/388。這是第一輪CPU候選存在率，不是最終P/R；完整證據detection-v5-first-full-bicycle-RPN-001。先前6選例不能概括全體。CPU survey無須重啟。

GT-only全train小自行車比例10.95% vs val53.22%，故新增experimental_detector_geometry.py備案（未整合SRC／trainer）：成對zoom-out/flip，padding全ignore、loss只帶boxes/labels/ignore。6項CPU幾何檢查全過；verify_detector_geometry.py對真2個train groups＋immutable firstepoch parent做1次ephemeral CPU optimizer finite、兩頁實看對齊，沒有正式checkpoint。exec71390已exit0。證據detector-geometry-preflight-001，PHASE3_DETECTOR_GEOMETRY.md記限制；可考慮先於P1處理尺寸落差，不可把備案當改善。

完整129項CPU suite：126pass/3GPU skipped、0failure，13.993s（cpu-regression-20261018-003）；原94GPU preflight仍是唯一已整批實跑GPU結果。新installer3項＋geometry6項包含其中；安裝器後續新增fullmap／binary守衛不影響active sources。

**官方AdditionalMaps_0.9.16.zip正在下載，exec46320**，檔案artifacts/downloads/AdditionalMaps_0.9.16.zip.partial，HEAD7250055782bytes，13:45約2.75GB／尚未安裝。官方release及文件證實Town06/07為額外內容，本機Content/Carla/Maps當時沒有它們；這不是server availability查詢。下載不要重啟或盲resume，先查session/size。新install_additional_maps.py只inspect或content-only新檔合併，拒絕現有不同bytes、路徑逃逸與executables，不能覆蓋patched server。先inspect實際archive內容再決定install。shippingSHA b0370d6dcd57b95299d12005cebad1dc6187804e83aeb51a184ccbc4b581fea8，官方未驗publisherchecksum，來源證據additional-maps-preparation-001。可在CARLA關閉時準備assets；**仍須等原GPU批次全退出才啟CARLA／RPC採集**。

使用者要求token恢復後盡量一次完成Phase3，允許批量執行及統一驗證；持續工作，不按小段落結束。**Phase3尚未通過，全部新模型heldout未開。** 詳細歷史與恢復見CHECKPOINT_PHASE3_2026-10-18_BATCH.md。

## 已完成且批准

- 主資料data/sunny-cross-town-v3-007保持immutable：18個30秒動態groups，86400PNG，train Town01/03、val04、dev05、lock02。
- 舊segv4完成但wall/fence/margin失敗；舊detv3完成8輪，mAP50 .60210，ped/bike/light召回.49907/.30727/.38176未過.5。
- phasev1-002 GTcrop macro .94381、red→green1.33%通過；搭final detv3的actual框joint macro .360204<.4，matched red→green0，joint失敗已實看28錯例並保存。
- 號誌heldout覆蓋補收data/light-lockbox-supplement-v1-002完成68組204PNG，audit與兩個4-state QA實看通過；unknown38/R14/Y14/G14，各2原groups。不供train，不代表模型品質過關。
- **靜態分割補景data/segmentation-scenery-v1-003已完成1024view/128anchor/4096PNG，獨立audit及5頁38情境實看通過，training_approved review已寫，實際verify_scenery也通過。** auditSHA06703e9bf4d88f3788f1cab169cd2eb04c9b2b4ee165300bdcbca7a0919ca68f；reviewSHA607b9c5cc24710dfc582085abb22003d93eb6dd363c06babf2bd92e2d7c3dd54。
- 補景只原train Town01/03；Town03沿用001成功512view，Town01重收512；失敗344/59view和probe全部不供train。只有static virtual rig，不能算新動態路線或3D資料。QA實見Town03 anchor114的木板斜撐Fence、白picket、metalmesh、石牆/草地/道路negative。
- 同步採逐幀snapshot buffer、future只留給其真實F、F+1後取完整四模態；每tick wall.05、最多24候選。207不完整候選丟棄，pose最大4.52e-5；沒有錯幀重標。raw0 instance植被缺標分開報告，resolved類別一致性仍2%門檻，語意targets及模型gates不變。
- CARLA已停（owned27540/30584均退出）；server awake helper23920已released/request_clearedtrue。exec14255採集及23190 audit均exit0，勿重開舊PID。

## 立即執行的唯一批次

```powershell
.venv/Scripts/python.exe -u scripts/run_phase3_recovery_batch.py --segmentation-config configs/sunny_segmentation_v5_1.json --output artifacts/runs/phase3-recovery-batch-001
```

**已啟動唯一批次：exec32462；native parentPID19772（venv launcher31940）、awake helper4644，6h綁parent確切creation。** 10:27起跑分割child native26248（launcher23100）。11:26已完成3輪，正進第4輪；不要重複啟動。用scripts/status_phase3_batch.ps1共享讀取落盤狀態；程序/GPU實況另查native CIM/nvidia-smi，PID可能退出，使用前重查身分。

- v5 seg：從v4 best SHAa703...微調6epochs/batch8，decoderLR.00015/encoder.00005；9600main +1024unique static各重複4次=13696sampledviews/epoch、10272updates；70.09%main/29.91%static，不把重複算新場景。
- v5 det：從v3 best SHA9cbed...微調4epochs/batch2/LR5e-5，內部720x1280、輸入及框仍640x360；min/max resize必須由checkpoint config重建。
- v1-003 phase：同設定重新訓練，凍結正確新版det loader（舊phase/source硬編360無法正確做新det的controlled測試）。新phase config保存原始bytes/dataSHA。
- 新選模事先凍結：優先val所有原門檻通過，再比原AP50:95/mIoU；沒有合格仍挑均分最高並記失敗；每epoch保留全部門檻結果。
- batch序列：完整GPU回歸→真實production loader容量/finite更新檢查→seg→det→phase→actual boxes joint→統一val gates。全部val過才一次heldout；之後真實8RGB demo/個別與整合3種warm benchmark/export。val失敗仍可出真實未合格報告，不能把export存在當完成。
- batch有OS鎖防重複、source/config/data/input contract、stage receipts、6h防閒置helper。中斷用同來源--resume；不得刪marker重開heldout。無完整epoch的失敗run換版本；完成run不能resume。

## 驗證與限制

**batch完整GPU suite已94/94通過、零skip；高解析det容量/finite正反向、真正production loader重載exact一致已通過。** batch2峰值allocated3350.97MiB/reserved3924MiB；dummy3updates僅容量測試，不是正式模型。新模型品質尚未測，不可假裝通過。主門檻不降低，只有ClearNoon，不進Phase4。

舊run要從自己的source快照執行evaluator，dataset-root用絕對路徑；sourcesha比文件Git revision更具體。Native CARLA RPC需escalation；不用過去DX11/Low。status讀檔用ReadWrite|Delete共享，避免Windows atomic replace鎖；不要用舊log當程序仍活著。

## 11:26中間結果與診斷（不是完成）

- seg第1/2/3輪 best mIoU .568366/.583960/.587430；wall .075309/.078031/.080723，始終未過.2。第1輪fence .584425、margin .502724；第3輪margin .487057未過.5。其餘15類IoU通過。後續仍按預先凍結6epochs完成，不把中間結果當驗收。
- 初始完整val與v4選定best的JSON逐bytes一致，兩者SHA f68d0df2b0f8724c2864925999b88130ef2125a63c7ea9963cee9eb575357a36；overfit ratio .423368通過，非新成績。
- 第一輪best另存不可變diagnostic snapshot：sunny-segmentation-v5-first-epoch-snapshot-001/best.pt，SHA4e1a0d7ff0e2dac78c006073221ee402c9615c20a7475999d22bbe64ac67fb94，不偽裝completed。
- evidence/phase3/sunny-v3/wall-diagnosis-001保存實看17頁：9個val錯例3頁、24個train形狀候選8頁、16個train外觀分層6頁。val漏檢含木質實心圍牆及水泥道路分隔，岩面誤報明顯；先前「只是低水泥牆不足」推論不完整。train實看主要磚牆/水泥/隧道；沒有憑這16例就斷言全train沒有木牆，亦不可把木質Fence重新標Wall。
- 全9600train掃描：4688低牆形狀hint、7805中性非牆hint；形狀是depth×pixel高度proxy，不是物體高度，顏色不是材質GT。外觀QA在每秒/相機/group去重候選1374張上分層，非新獨立資料。
- 新CPU研究exec30015：scripts/study_segmentation_prior.py，output segmentation-train-prior-study-001；不碰GPU/source contract。只第一輪immutable checkpoint，完整3600validation，事先凍結alpha0/.25/.5/.75/1、logits-alpha*log(train頻率CEweights)。探索類別偏置；沒有改任何正式推論、訓練、品質門檻或heldout。CPU FP32不能代替正式GPU BF16驗收。先查progress/report再決定，不要重啟重複研究。

## 11:53執行中補充（務必先讀，避免競爭）

- seg已進最後第6輪，9900/10272updates，最新selected仍第4輪 mIoU .589854、wall .075970、fence .591737、margin .489201，wall/margin失敗。未完成final reload/完整val，不可先用completed checker。
- CPU研究到3200/3600；尚無report.json。完整結果要實讀，不把先前半批結果當結論。
- **一次性等待接續器exec67677/native32944已於11:58有意取消，未啟GPU校正。** 先查CIM命令列，再以同一native process handle核對creation FILETIME134367682905851523並停止；operator_stop.json證實退出，沒有sunny-segmentation-v5-prior-001，awake29360已released/request_clearedtrue。不要重啟此舊PID或只看遺留waiting state；原GPU batch19772完全保持運行。
- 正式校正工具尚未GPU執行；新5項prior/3項artifact selection/3項native process identity focused測試都通過，證據prior-calibration-preflight-001。原batch先前GPU94全過，新增11項尚未與完整GPU suite合跑。canonical batch sources及model fingerprint於11:46再核對仍完全相同，無中途改SRC。
- 修正file-only segmentation artifact checker尊重qualified-first，避免高平均但不合格epoch蓋過低平均已合格epoch；舊v4完整3600frame artifacts重檢通過，模型品質仍原3項失敗。此checker不在active source contract。
- 見evidence/wall-seen-train-fit-001：第一輪snapshot對28個已選seen train contexts CPU wallIoU .969654/P.981587/R.987617；2頁highest FP/FN已實看。這是有偏樣本非完整train accuracy，支持優先修復外觀泛化，不能當新驗收。
- candidate configs/segmentation_low_wall_preview_v1_1.json僅預覽表：Town01/03各新增24 native道路pose、原128anchors不變、min8m；未採集/未批准train。舊candidate v1及其exact source在artifacts/runs/low-wall-preview-v1-superseded，v1_1才是目前候選。PHASE3_DOMAIN_RECOVERY.md有先preview原train、若必要才新增未使用城鎮的資料路線，不能只重複既有磚/隧道牆。
- 最新工具/證據commit e500b16；phase3未通過，不可結束為已完成。繼續等目前批次→統一val/joint/demo/bench/export，按實際結果處理缺口。

## 11:59覆蓋更新

CPU研究exec30015已exit0，完整3600val×5固定alpha完畢。alpha0 mIoU .569206/wall .074063/margin .503958；alpha1 mIoU .579824/wall .074656/margin .501732；全部候選只有wall門檻失敗。這是第一輪snapshot的CPU診斷，不能代替GPU或外推成第6輪校正結果；但改善幅度不足以優先再花GPU測同一先驗法，因此取消閒置watcher，工具仍可供另一次有依據的實驗。證據segmentation-prior-CPU-study-001，full report SHA ee6b85c0be4427c2a6a54d8f20b18ad1a3382744341522999f7cb677fe0ea78b。

seg v5正式6epochs/10272updates已全走完，第6輪選定mIoU .5900253284664799、wall .09379871927291071、fence .6147462883597083、margin .49446801091434905；仍wall/margin兩項失敗。此刻state仍training，正做最終selected reload及full val輸出，等completed再跑artifact check與新錯例。只有原batch等待接續det/phase；無GPU校正watcher或CARLA。後續資料修復按PHASE3_DOMAIN_RECOVERY.md，主train不能當新獨立驗收。

## 12:03正式完成分割，已接續偵測

seg state已completed、exact reloadtrue，best SHA3fe7bd08d885e481b7435fdf466945f142f8323447cca46bf31c32455835a53f；validation SHA f33385596c75cb12e4022bbba972c33bdeba5d80943f5ceff4a15d40fc4a9350、frames SHA98c4ac93ac6595fb033a334b148775b8659b0a68b94e0c84adb7ee15c44316a0。完整file checker通過，wall/margin品質失敗；最終3頁8例已實看並封存sunny-segmentation-v5-001-completed-validation證據。峰值allocated4046.51MiB/reserved4662，training_wall5252.77s。

原唯一batch於12:01:07接續train-detection，launcherPID27684。run尚未建立時先查原batch stages/native程序，不要再啟；完整86400PNG校驗在建立run之前。batch parent19772仍是唯一協調器。CPU研究與取消的接續器均已結束，沒有新GPU模型或CARLA。接下來持續完成det/phase、統一val/joint、RGB demo/bench/export，再按缺口補資料與新訓練；Phase3尚未通過。

## 12:32偵測正式訓練與離線準備

- det nativePID6608／launcher27684，exec仍32462，12:09完整86400PNG校驗完，overfit ratio .072984通過；重置後initial高解析val mAP50 .539253/AP50:95 .314878，不能當改善。12:14左右正式訓練起，12:32 epoch0/2000updates，規劃4epochs/19200updates；meanloss .19784、無failure。首輪約4800updates後才完整val，保持單一GPU job。
- Completed seg v5模型卡已產生，README與evidence/phase3/registry.json更新。所有新模型heldout未開。原生server仍關，只有原batch awake helper4644（最晚16:26附近）有請求；若批次超時仍需查具體PID再補有界helper，別改電源計畫。
- 新scripts/analyze_detector_operating_points.py已對舊v3全部3600val的原score=.5八類TP/FP/FN exact重現；固定探索grid內bike/light無同時P/R≥.5的點，證據detector-v3-operating-point-diagnostic-001。**未採用新閾值**，正式/display仍.5；待新v5完成可用同工具查confidence vs容量。
- MixStyle-inspired prototype與6項CPU測試在scripts/experimental_feature_style.py/tests/test_experimental_feature_style.py；PHASE3_FEATURE_STYLE_EXPERIMENT.md及design JSON記錄。eval/RNG/statekeys/finite通過，但**未整合src/trainer、未GPU驗證、未訓練**，不能把design當train config。原始論文不是本專案seg品質證明。先按資料修復，保留為有界替代實驗。
- 新scripts/scout_unused_training_town.py與configs/segmentation_training_expansion_scout_v1.json只預留Town06/07 metadata候選。5項split/Opt別名/OpenDrive重名保護CPU測試及真main policy兩次--dry-run通過；**無RPC/無新inventory/無RGB，native maps是否可用尚未查**。須先等GPU全部退出，再另client準備full map，scout自己不load/tick/render。原v1 Town01/03 collector/loader完全未放寬；若擴城資料須另凍結v2 capture/audit/review/train loader，不能直接匯入。
- 現在完整suite預期116項：原94 GPU曾全部過、新11prior/selection/identity＋6feature prototype＋5unusedTown focused CPU已各自通過，**尚未整批合跑116GPU**。canonical active batch全部來源於12:28再核對仍完全不變。native監看functions cell234僅read-only loop，已停止；不影響exec32462。

12:35已另外整批跑完116項CPU回歸：113passed、3明確GPU skipped、0failure，10.506s；只在獨立測試shell/subprocess設定CUDA_VISIBLE_DEVICES=-1，原native GPU訓練不受影響。證據cpu-regression-20261018-001；仍不是116項GPU全過。det epoch0/2400updates仍正常，Phase3及全部新模型heldout狀態不變。

## 13:20偵測第一輪成果及查錯（批次仍繼續）

- det第1輪完成4800updates，mAP50 .7120289789350768、AP50:95 .4396834044368154；7類全部P/R合格，只有bicycle recall .3484224965706447失敗（P .5497835497835498）。pedestrianR .90909、traffic_lightR .53167、motorcycleR .50651。首輪2638.531s，maxgrad34.09是clip前；無fully-ignored skip。第2輪於13:20已到7600totalupdates，仍exec32462/native6608，不改4epoch設計。
- 第一輪checkpoint已存sunny-detection-v5-first-epoch-snapshot-001，SHAb2b69ebac6652c43054148cf132d78fbbb504f4b41af5fa608cd0a2b8a2b8237，只有immutable intermediate snapshot，不是completed。完整GPU bicycle size-strata：small31/388=.07990、medium196/308=.63636、large27/33=.81818；truncated37/154=.24026。
- CPU GT固定選例16train＋6val：train16/16找到、val3/6，3個val small全漏，3頁9contexts已實看。原001診斷漏truncation metadata已保留known_issue及原tool，修正另跑002；全部類AP/P/R/TP/FP/FN exact一致、PNG逐頁SHA相同，只有truncation分層修正。**正式GPU報告沒受影響**。現用sunny-detection-v5-first-bicycle-fit-002，不用001的truncation統計。
- 6val CPU proposal study：pre/post500/150、1000/500、2000/1000，都6/6有IoU50 proposal，最終Bike仍3/6；baseline所有prediction逐欄exact重現，weights/buffers一致。只支持在這些有偏樣本先查ROI分類/回歸/feature，非完整split因果證明。兩診斷及firstepoch metadata封存detection-v5-first-epoch-diagnosis-001。
- PHASE3_BICYCLE_RECOVERY.md與scripts/experimental_p1_detector.py準備stride2備案；4項CPU測試及**真firstepoch訓練參數**完整transfer通過：20,151,660→20,746,092（新P1 594,432params），原coarse features exact。**未整合production loader、未720 CUDA驗證、未訓練／評分P1**；若v5本身通過就不必用。原batch source contract仍完全不變。
- 最新整批CPU suite120項於13:19完成：117pass/3GPU skip/0fail，12.049s，證據cpu-regression-20261018-002。這不是120項GPU全過；原94 GPU preflight仍是已實跑的GPU證據。兩個診斷CPU jobs已exit0，目前只有原batch GPU工作。
