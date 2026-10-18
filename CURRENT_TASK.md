# 目前任務：批量完成晴天 Phase3（2026-10-18）

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
