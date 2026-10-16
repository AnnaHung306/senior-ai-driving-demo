# Phase 3 失敗證據

1. CUDA NLLLoss2d mean / sum reduction 被 strict deterministic 拒絕；官方 v2.10.0 source 確認 atomicAdd。改 per-pixel reduction 後 weighted-mean 等價與 RNG 恢復測試通過。
2. segmentation-v1-001 在 80-step train-only overfit 尚未完成前遇到非有限 AMP 梯度。loss-scale 初始 backward 的 65536 / 1024 對照均有限，說明不能只測第一步；完整 overfit 才攔下後續溢位。原 source snapshot 仍在 artifacts/runs/segmentation-v1-001/source，對應 Git 13df777。未產生正式模型、未打開 test。
3. v1.1 另開 segmentation-v1-002，initial scale 1024、最多八次 backoff / same-batch retry，只有有限梯度能更新；人工 Inf 測試確認壞更新跳過。不得把原 run 改寫成成功。
