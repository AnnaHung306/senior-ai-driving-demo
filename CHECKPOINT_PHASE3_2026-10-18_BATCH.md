# Phase3 批量修復與統一驗證（2026-10-18）

使用者要求已從小工作切換為一次盡量完成Phase3；本檔只是恢復紀錄，不是結束工作或驗收通過。

## 已確認

- 舊三訓練均完成、全部新模型heldout未開。detv3 mAP50 .60210175/AP50:95 .36368405，行人recall .499072、自行車 .307270、號誌 .381757失敗；segv4 mIoU .508456但wall .061948/fence .007284/marginF1 .407333失敗。
- phasev1-002 GT-crop通過；用final detv3框做joint macro .360204、matched red→green0，joint失敗。28錯例已實看、保存evidence/light-state-v1-002-joint-validation。
- frozen號誌覆蓋補收已完成於data/light-lockbox-supplement-v1-002，68組204PNG；獨立audit passed，unknown38/R14/Y14/G14，每類2原groups。兩個QA已真正查看並寫visual_review；僅供heldout，不作train，主動態與controlled仍分開。001缺幀失敗保留。
- Town01/03原生scout都完成且按檔壓縮進evidence/training-town-scout-001。Town01有原生picket候選；已實看新增6個白色尖柵欄/庭院RGB與semantic5遮罩，名稱提示沒有被當材質GT。

## 靜態補景與同步修復

- 凍結configs/segmentation_supplement_v1.json，SHA8d2cab60f1287a195e77c4a84435db53070e99360c9ed10399d75dd25e657350；只原train Town01/03，128道路錨點×8方向=1024view/4096PNG，虛擬rig、無ego/NPC，不是動態路線。
- root001 Town03完成512view。Town01 root001僅344view後futureF+1連續失敗，cleanuptrue；不能訓練。
- 曾推測奇偶重試保留造成失敗，加入額外一tick後root002 Town01仍在59view失敗，故奇偶不是完整原因。002已失敗、不可訓練；其Town03只是原001逐位元複製，不計新增。
- 新aligned_triplet改逐幀snapshot pipeline：保留每sensor已接到的future樣本，轉到它自己的snapshot再等F+1，四模態必須同一真實F/time。先前receive_frame遇future便丟掉它，不能用它判下一個frame。新測試以RGB缺101但保留102、其他三sensor都有101/102，最後明確返回102且world在103；沒有重標future成101。無commonframe有界失敗。
- 最新原生短測exec70596：artifacts/runs/scenery-synchronization-probe-001，8anchors/64view。probe不可用於訓練。先檢查native完成、PNG/實際pose/time/frame，再建root003僅重用Town03、完整重收Town01；不要盲目開第四個全量重試。
- 新collector採集前存source_snapshot，report綁所有sourceSHA；Town03舊成功capture的未變動來源是在修正前事後存檔，receipt有明示，不假稱採集當下寫檔。
- full auditor拒絕probe、矩陣缺漏、CRC/shape/timing/pose錯、semantic-instance>2%偏差、source snapshot錯及wall/fence/terrain支持不足；生成QA。必須實際看QA並另寫綁auditSHA的training_approved visual_review，才能訓練。

## 新訓練與統一驗證工具（尚未啟動）

- configs/sunny_detection_v5_1.json：從detv3 best微調4epochs/batch2/LR5e-5，內部720x1280、原輸入和框仍640x360。make_detector/train/predict載入都必須按config重建resize；未變映射/anchors/gates。
- configs/sunny_segmentation_v5_1.json：從segv4 best微調6epochs/batch8/decoderLR.00015、encoder.00005；新靜態補景重複4次，主動態9600/13696=70.09%，靜態1024獨特views重複4次=29.91%，總10272updates。**目前config補景root仍指002，需等probe/003正式audit後改正才freeze。**
- source/provenance明確綁main與training_supplement audit/review/plan/獨特views/groups/repeats；保守cache budget檢查；只train segmentation可append，不可加到det/phase/val/heldout。
- 新v5 validation-only selection：優先全既定門檻合格，再比原任務均分，無合格仍挑均分最高並記失敗；每epoch記全門檻，checkpoint含best_qualified，resume保留。
- run_phase3_recovery_batch.py序列：全部GPU回歸→det容量→seg訓練→det訓練→phase v1-003重跑同設定以凍結新版det loader來源→actual框joint→統一val gates→全通過才heldout→RGB-only8view→3類warm benchmarks→export。val失敗可仍輸出真實demo/效能與不合格報告，不能宣稱Phase3完成。
- 新phase v1-003有必要：舊phase/source的load_model硬編360，不能正確評估新720det的controlled supplement；新phase保持相同crop任務/超參數但凍結正確loader。新phase保存config原始bytes、data/configSHA到state。
- batch有OS檔案鎖拒絕重複runner、contract綁來源/config/輸入/資料、逐stage receipts與log、整個batch最多6h防閒置helper，parent退出自動清除。若中斷只能exact-source --resume；無完整epoch的失敗run換版本；已開heldout marker不得重開。
- start_carla guard加入batch/capacity程序名；不與GPU訓練/評估並行。所有Native RPC需escalation，預設DX12/Epic，不用過去DX11/Low。

## 最新測試

CPU suite012：92項、89pass/3CUDA skipped；另batch OS lock focused測試通過，總測試現應93項（尚未全量GPU）。新同步6項focused tests007通過。GPU容量/統一訓練/新模型成績均尚未做，不能把code或計畫當成果。

下一步先確實完成短測與補景audit/看圖，關CARLA、commit這一批來源，再真正啟動batch持續到品質驗收；模型錯例/最終demo也要實看。權重與raw留本機ignored，Git只收程式、configs、審核與選定QA/報告。原86400主資料完全不改。


## 10:10之後覆蓋更新

Paced buffer probe002已完成64view/256PNG，獨立verifier通過，max pose4.37784e-5、raw mask差異1.7096%，43個不完整候選已丟棄；probe001未paced仍失敗，紀錄保留。root003正式Town01 exec14255正在採集，參數每tick牆鐘.05、每view有界24候選；Town03512只沿用001原成功版本。CARLA owned27540，臨時awake exec8068/helperPID23920已綁server確切creation FILETIME，server停即清除或最多1h。

Town03全512先檢發現max raw difference2.00651%；實際最差影像4623pixel全是semantic9植被對instance0未標註，已看4panel/像素對照。不能將未標註0當有效錯類：resolved_class_alignment只比較兩側非零，保留全raw差異/unknown支持/29x29matrix，仍2%界線，不改semantic targets或模型gates。新單元測試通過。修正後全Town03預檢另外存static-label-availability-001。正式audit仍須對全部1024/4096做CRC/time/pose/intrinsics/weather/path/coverage/sourcechecks與QA。新segconfig現在指root003，但無audit/review仍不可訓練。

Native完成後才能full audit（不可先寫不完整audit，因immutable），實看各QA頁，建立單獨training_approved綁定review，再關CARLA、commit來源，啟動：

```powershell
.venv/Scripts/python.exe scripts/audit_segmentation_supplement.py data/segmentation-scenery-v1-003
# 實際view_image並寫review後才啟batch
.\scripts\stop_carla.ps1
.venv/Scripts/python.exe -u scripts/run_phase3_recovery_batch.py --segmentation-config configs/sunny_segmentation_v5_1.json --output artifacts/runs/phase3-recovery-batch-001
```

新GPU容量工具會從真正production load_model重新載入dummy3updates的synthetic checkpoint，包含720resize非tensor設定，這些dummy權重絕不是正式模型。batch會重做完整GPU tests；尚未啟動GPU訓練。最新focused scenery6項全通過，目前總數預計94（GPU全套待實跑）。


## 正式補景及批准完成（10:20左右）

root003 Town01完成512、exit0/cleanuptrue；完整1024view/4096PNG audit passed，auditSHA06703e9bf4d88f3788f1cab169cd2eb04c9b2b4ee165300bdcbca7a0919ca68f。5頁38contexts已實際看過，reviewSHA607b9c5cc24710dfc582085abb22003d93eb6dd363c06babf2bd92e2d7c3dd54，verify_scenery全檔再核對通過。資料與QA摘要壓縮證據在evidence/phase3/sunny-v3/segmentation-scenery-v1-003；full audit原bytes gzip可還原。static原生木板斜撐圍欄真的出現在Town03 anchor114，原主train四Town03 routes完全沒有Fence，這次補到新context；不宣稱有兩個獨立木板地點。

full source/data/classpairs/intrinsics/weather/path/timing/pose檢查通過，207不完整候選被拒，maxresolved8.7097e-6；完整raw差異及unknown支持仍保存。原生server27540/30584已taskkill自有程序樹退出，awake23920已releasedtrue。不要重啟CARLA，進GPU batch。最新完整批次命令與狀態以CURRENT_TASK.md為準。
