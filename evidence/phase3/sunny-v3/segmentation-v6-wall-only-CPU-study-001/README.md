# v6 Wall-only固定偏置：完整CPU負結果

先前已啟動的CPU診斷現已完成全3600val，九個alpha事先固定0,.125,...1，只扣Wall通道的train-frequency偏置，原checkpoint未改。CPU alpha0 Wall IoU .153231，alpha1最高.161307，九個候選全部未通過原gates。僅小幅變化不足以支持自動排GPU校正或擴大搜尋；不以CPU當正式GPU驗收。

CPU FP32 alpha0與正式GPU BF16 parent .156453不同，不能把兩種device結果差異當模型改進。原report保存完整grid，原門檻及heldout隔離不變。未啟任何新校正／訓練。
