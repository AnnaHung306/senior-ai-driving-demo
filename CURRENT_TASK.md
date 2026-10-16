# 目前任務

更新：2026-10-16（Asia/Taipei）。原始長期需求見 PROJECT_SPEC.md。

Phase 0、Phase 1 初始規則 Baseline、Phase 2 v2 基礎資料驗收已通過。**Phase 3 進行中：訓練基礎與第一輪實際 segmentation baseline 完成；整個 Phase 3 尚未通過。** 不進入 Phase 4。

## 本輪可追溯成果

- 實驗：artifacts/runs/segmentation-v1-002；source commit=a1eff7b；config=configs/segmentation_v1_1.json。PyTorch 2.10.0+cu128 / Python 3.12.14 / RTX 4070 Laptop。CARLA 關閉，訓練 704.96 秒、16 epochs / 4800 optimizer updates。
- Tiny U-Net 從零訓練，488,120 parameters，640×360，batch 4，seed 20261016，七個 focus 類 + other_annotated；raw 0 ignore。只 train 更新與算 weights，validation 選模，test 最後一次完整評估。
- 最佳 epoch 12：validation focus mIoU 27.50%，test 26.22%；常數多數類基準 0%，train-prior 解析期望基準 2.69%。沒有設定事後品質通過線。
- Test 晴天 scene-04=49.04%，夜雨 scene-05=11.83%；場景幾何不同，不能把差異單獨歸因於天氣。夜雨道路 IoU 僅 1.13%，自行車總 recall 20.21% / precision 8.10%，行人 / 汽車誤報也明顯。不能接上駕駛控制。
- 保留 14 張按固定錯誤規則選的 RGB / GT / prediction 對照、1200 筆 validation+test per-view 指標、84 筆逐類 FP / FN triage entries。這些不是獨立錯誤場景數。
- CUDA overfit 80 steps loss ratio=0.3716；正式 run 無 AMP backoff。checkpoint 重載 forward 一致；32 tests 通過，含 RNG 恢復後下一步更新一致、人工 Inf 跳過壞更新、resume 來源不符拒絕、test 重開拒絕、完全無 labels 的 RGB-only 推論。
- 訓練後再次重驗 7200 PNG hashes、639 metadata files、source/config/criteria/rig，全部未變；engineering_audit passed。實際最佳模型也在僅 RGB+calibration fixture 成功推論，輸出在 rgb_only_demo。
- Peak allocated / reserved VRAM=1382.4 / 1538.0 MiB。Batch1 network p50 / p95=9.07 / 9.85ms；warm-cache 載圖到 mask+confidence=25.64 / 32.04ms。八方向單 bundle 順序處理 217.09ms，不含模擬器／寫檔，不是 20Hz 八鏡頭即時達標。

完整協議：PHASE3.md；model card / 數值 / provenance：evidence/phase3/segmentation-v1-002；best / last checkpoint、source snapshot、對照圖、review.html 留在原 run。資料與權重未進 Git，複製工作目錄時須另存。

## 失敗與修正

CUDA NLLLoss2d 的 mean / sum atomic reduction 被 deterministic 拒絕；改 per-pixel loss 再 tensor reduction，保持 weighted-mean 定義。segmentation-v1-001 在完整解析度 train-only overfit 出現 AMP 梯度溢位；未碰 test，原 config / source / failure 保留。v1.1 固定 initial scale 1024，有限次 scale backoff / 同 batch retry；只有有限梯度更新，不關閉保護。證據：evidence/phase3/failures。

## 下一個單一任務

Phase 3 資料補強的前置 pilot：依 configs/detection_label_contract_v1.json，實測 RGB / semantic / depth / instance 共 32 sensors 五幀同步、已知 actor 的 instance ID / visibility / near-plane / truncation，保留人工 QA 與峰值資源，再凍結可見性門檻與 v3 capture plan。先回歸既有 Phase 0；不可直接投影所有 3D boxes 當偵測 GT。

configs/dataset_expansion_v3_plan.json 明確為未收集計畫：至少 18 個獨立 route groups、較長 episode、跨天氣／時間、空目標負樣本、兩輪車與大型車／號誌／障礙物；保留新的 lockbox。已看到 Town01/02/03/04/05/10HD map assets，但尚未 RPC / render 測試新城鎮。當前 test 已揭露；若據此改模型，只能視為 development test，不能再冒充未見的正式驗收集。

## 不要重犯

1. Town01 已載入時沿用地圖，不冗餘 reload。新城鎮的載入與幾何須另做 probe，不能沿用 Town01 座標。
2. PixelReader background patch SHA256=b0370d6dcd57b95299d12005cebad1dc6187804e83aeb51a184ccbc4b581fea8；保留 stock 備份與 exact-build guard。使用 DX12 / Epic；不要回到已失敗的 DX11 / Low。
3. sensor readback 按原 snapshot frame 配對，不拿下一 world tick 冒充當前 sensor frame。
4. NPC cleanup 先取消 autopilot，world restore 前保持 TM sync，最後才關 TM sync。
5. 新 actor 首次 tick 前 get_location 曾為 (0,0,0)，導航必須在暖機 snapshot 後初始化。
6. GT 只在訓練／評估，推論只用 RGB；BasicAgent 是規則。semantic road 不是可導航區域，other 不是空白安全空間。
7. 6 個短場景不是 2400 個獨立樣本；validation 1 group / test 2 groups 無可靠泛化信賴區間。沒有 AP、偵測器、號誌狀態、3D / BEV、導航介面或 AI 駕駛閉環。

目前模型訓練、CARLA 與資料程序均已結束，沒有監控排程或其他背景工作。
