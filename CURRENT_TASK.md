# 目前任務：可中斷的小檢查，繼續晴天 Phase 3

最新狀態：**CHECKPOINT_PHASE3_2026-10-17_SMALL_CHECKS.md**（2026-10-17 12:26 Asia/Taipei）；完整流程與限制見 **CHECKPOINT_PHASE3_2026-10-17.md**。舊程序與舊成績不能當最新狀態。

- 使用者允許繼續，但本輪做小工作，每項落盤，避免中斷丟失成果；沒有要求停止原背景程序。
- 分割 v4-001已完成8輪/9600updates，exec68540 exit0，best第7輪。檔案SHA、27個source、3600張validation逐張混淆合計核對通過。
- val mIoU50.85%達總平均門檻；wall6.19%、fence.73%、路緣F1 .40733仍失敗，**不能說品質通過**。未開heldout。
- queue exec57077仍活，號誌 v1-001程序已呼叫；12:25尚無run/log內容/epochs，不能當成已訓練。Python實際命令已核對。det v3-001仍排其後，不重複開run。
- 新檢查器 `scripts/check_sunny_segmentation_artifacts.py` 純CPU讀檔、輸出atomic、可立即重跑。收據與薄弱類別混淆診斷在 `evidence/phase3/sunny-v3/health-20261017/`。
- RGB-only示範輸入9檔SHA與schema通過，號誌config仍綁補收plan；沒有做推論/採集。
- CARLA off，ClearNoon；黃燈補收、joint validation、heldout、三模型RGB-only demo/benchmark及最終模型卡導出仍未完成。
- Phase3未完成，不進Phase4。先查實際程序及新log再接續；勿以歷史PID殺程序或重新啟動相同run。
