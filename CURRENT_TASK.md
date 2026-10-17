# 目前任務：先存檔，接續完成晴天 Phase 3

最新完整段落存檔：**CHECKPOINT_PHASE3_2026-10-17.md**（2026-10-17 10:11 Asia/Taipei）。請先讀此檔；舊歷史見Git，不要拿舊程序或舊成績當目前狀態。

- 使用者要求token快用完先段落存檔再繼續；**沒有要求停止訓練**。
- CARLA off，固定ClearNoon；主007資料18場/86400PNG完整audit與assistant視覺抽查已過。
- 真正在跑：segmentation v4-001，exec68540，8epochs/batch8 MobileUNet BF16/FP32。第一輪val mIoU44.36%、marginF1 .36546，仍未通過wall/fence等門檻。最近epoch1/2200updates（會繼續更新）。
- queue exec57077等v4 completed後自動phase v1-001 configlight_state_v1_1→det v3-001 configsunny_training_v3_9。存檔時兩個後續run還沒開始；不要重複啟動。
- Tiny v3-001已刻意abort、保留3輪29.37%與嚴重過擬合證據，不是完成。新v4來源snapshot commit046ec3b，勿強行用後續修改source resume/eval，可用run/source凍結腳本及absolute --dataset-root。
- phase lockbox黃燈不足，補收204PNG plan已凍結但**未採集**；訓練全部退出後才啟CARLA補收、audit、實際看QA、寫bound review。
- 正式heldout前先通過actual detector-box phase validation joint，再一次heldout；原門檻不降低。最後需三模型實際RGB-only8view demo、benchmarks、錯例、模型卡與registry/README。
- 存檔前完整suite **72 tests passed**，log phase3-sunny-tests-009.log；last.pt SHA已核對，完整/partial進度分別存於evidence checkpoint receipt。
- Phase3仍未完成，不進Phase4。詳checkpoint的全部參數、執行順序、坑與品質條件。
