# 目前任務：批量完成晴天 Phase3（2026-10-18）

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
