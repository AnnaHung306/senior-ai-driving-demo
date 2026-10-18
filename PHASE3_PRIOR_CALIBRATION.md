# Phase3 分割後處理校正實驗

此方法尚未在正式完整 GPU 模型上執行，不能算修復成功。原批次與來源完全不變；校正是另外的新 run。

現有 loss 含 train-frequency weighted CE，可能使少數類相對背景出現偏差。先固定 alpha 0、.25、.5、.75、1，研究 `logits - alpha * log(class_weight)`。class_weight 沿用原訓練的 inverse-sqrt frequency、1..20 clamp；不從 validation/heldout 的類別比例估計。此假說不保證解決跨城鎮外觀缺口。

CPU 探索使用第一輪不可變 snapshot、完整原3600 validation，contract 在推論前保存。結果只供診斷。正式工具只接受 completed、完整val artifacts及exact reload通過、未開heldout、但val品質仍失敗的parent。完整重新核對原資料與核准補景，獨立載入全部train重算支持量，再以原模型原17類的 classifier bias 實作校正。

GPU正式流程：alpha0 必須精確重現parent完整GPU驗證；所有候選以真實MobileUNet BF16 forward重新評分，不以CPU/float32相減代替；選模沿用「全部原gate合格優先，再mIoU，最早tie」。輸出checkpoint須exact reload、選定val重跑一致、除head.bias外所有tensor與parent逐一相等。校正過程沒有optimizer更新，receipt分開記0 calibration updates與parent inherited training updates，不捏造epochs/last.pt。

```powershell
# 只能在目前整批GPU程序退出後執行；不可與det/phase訓練並行。
.venv/Scripts/python.exe -u scripts/calibrate_segmentation_prior.py --parent artifacts/runs/sunny-segmentation-v5-001 --root data/sunny-cross-town-v3-007 --design configs/segmentation_prior_calibration_v1.json --run artifacts/runs/sunny-segmentation-v5-prior-001
```

如果校正也未過，保留全部候選結果與原始失敗，不開heldout。若全部seg gates通過，再搭配已完成且合格的detector/phase實際框joint結果重做統一val；全部通過才一次heldout。任何最終model card要明確記載calibration parent、方法、alpha、0新訓練updates及繼承updates，不能沿用「本run再微調」的說法。當前export尚未加入這種operation的敘述，必須於原batch退出後補齊，才能把校正run作最終導出。

原資料、分類與門檻保持不變。校正不能證明欠缺外觀已補齊；若完整GPU仍低wall IoU，下一步仍須訓練場景覆蓋與誤報negative的資料修復。
