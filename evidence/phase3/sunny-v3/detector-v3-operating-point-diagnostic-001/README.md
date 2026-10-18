# v3 偵測分數診斷：沒有採用新門檻

只讀取完成v3模型已保存的3600個validation視角，不執行新模型推論或接觸heldout。按原max100、IoU.5、一對一matching與正例優先ignore重新計數；原score=.5的8類TP/FP/FN逐一與正式報告exact相同後，才列出事先固定的探索grid。

在precision≥.5的已測grid中，bicycle最高recall .33196、traffic_light .42230，均未達.5；pedestrian有探索點能達P/R .57486/.57699。這不是對連續所有分數的最優界限，但支持舊模型自行車／號誌漏判並非只調顯示門檻即可宣稱解決。**原部署、展示及gate仍固定score=.5**，不採用探索點、不宣稱新模型能力或概率已校準。高解析v5為另外的正式訓練實驗。
