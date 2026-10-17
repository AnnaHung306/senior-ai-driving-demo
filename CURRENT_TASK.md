# 目前任務：可中斷的小檢查，繼續晴天 Phase 3

最新狀態：**CHECKPOINT_PHASE3_2026-10-17_SMALL_CHECKS.md**（2026-10-17 12:26 Asia/Taipei）；完整流程與限制見 **CHECKPOINT_PHASE3_2026-10-17.md**。舊程序與舊成績不能當最新狀態。

- 使用者允許繼續，但本輪做小工作，每項落盤，避免中斷丟失成果；沒有要求停止原背景程序。
- 分割 v4-001已完成8輪/9600updates，exec68540 exit0，best第7輪。檔案SHA、27個source、3600張validation逐張混淆合計核對通過。
- val mIoU50.85%達總平均門檻；wall6.19%、fence.73%、路緣F1 .40733仍失敗，**不能說品質通過**。未開heldout。
- 更新：queue57077已exit1。號誌v1-001跑完16輪/832updates後，JSON收尾因NumPy bool失敗，原run與failure保留。已修正單一布林型別，5項相關回歸通過。新queue exec56349正在跑v1-002相同設定，成功且state completed後接det v3-001；勿重複啟動。
- 新檢查器 `scripts/check_sunny_segmentation_artifacts.py` 純CPU讀檔、輸出atomic、可立即重跑。收據與薄弱類別混淆診斷在 `evidence/phase3/sunny-v3/health-20261017/`。
- RGB-only示範輸入9檔SHA與schema通過，號誌config仍綁補收plan；沒有做推論/採集。
- CARLA off，ClearNoon；黃燈補收、joint validation、heldout、三模型RGB-only demo/benchmark及最終模型卡導出仍未完成。
- Phase3未完成，不進Phase4。先查實際程序及新log再接續；勿以歷史PID殺程序或重新啟動相同run。

- 新增可中斷續跑的CPU validation錯例工具 `review_sunny_segmentation.py`，exec74874已exit0，輸出8例/3pages，已實際看完並保存evidence，標記inspected_with_failures。CPU FP32診斷不取代GPU BF16驗收。

- 15:40實際Get-Item確認v1-002目錄15:35:59已建立，無檔案表示尚未寫coverage；不能用Get-ChildItem空輸出推定目錄不存在。新Python PID33276累積讀取約320MB/81665次，有前置I/O活動。

- 最新15:44：號誌v1-002已completed且GT crop gate passed（macro recall94.38%、紅→綠1.33%），最佳epoch8。queue56349已接det v3-001，run/source已凍結，尚未epoch；所有後續phase命令改v1-002。號誌/固定景物錯例都實際看圖並保存evidence，joint仍未做。
