# 深度輔助訓練：有限資料可用性檢查

CPU/file-only，原train八groups×八cameras固定第一張64view，讀depth前決定選例。256 PNG與train index／manifest／calibration／inventory／原audit綁SHA。只檢查decode、尺寸與提案mask支持；沒有模型、GPU推論、訓練或heldout評分。現有loader尚未整合depth target。

提案mask下有效75.89%像素；Wall／Fence／Terrain距離分布重疊，不能由距離決定Wall類別。實作、原始report、contract及import source追加核對保留。完整改進與對照設計見專案PHASE3_VIDEO_RESEARCH_PLAN.md。此檢查不等於depth模型或Wall IoU已改善。
