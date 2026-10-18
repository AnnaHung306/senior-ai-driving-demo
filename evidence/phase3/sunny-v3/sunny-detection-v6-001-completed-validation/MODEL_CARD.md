# Sunny detection v6 — 訓練完成，品質驗證未通過

從 v5 最佳權重開始，四輪共 19,200 次 optimizer 更新；720/1280 resize、batch 2、LR 5e-5，含 bicycle-positive half-scale paired geometry。不是從零訓練，也不是與 v5 等算量的受控因果對照。

完整 3,600 張 validation：mAP50 70.461562%、mAP50:95 45.104874%；最佳第一輪（epoch 0），後三輪未改善。原 score 0.5 下逐類 Precision/Recall 門檻均 50%；未過項目為 bicycle/recall, motorcycle/recall, traffic_light/recall。不降低門檻、不開 development/lockbox 模型評分，Phase3 尚未通過。

檔案核對已對 canonical 3600 GT/ignore 與完整／逐場景 AP/PR 重算 exact；checkpoint reload exact。這是資料與結果一致性通過，不能當模型品質通過。原批次的 GPU 輸出錯例圖仍待獨立 post-batch QA 與實際人工檢視。

| 類別 | GT instances | Precision @0.5 | Recall @0.5 | AP50 | AP50:95 |
|---|---:|---:|---:|---:|---:|
| car | 946 | 90.5634% | 67.9704% | 75.5080% | 54.0652% |
| pedestrian | 539 | 89.1266% | 92.7644% | 93.4161% | 63.4638% |
| bicycle | 729 | 95.8730% | 41.4266% | 49.2664% | 34.2689% |
| motorcycle | 845 | 89.9142% | 49.5858% | 59.1478% | 43.5302% |
| truck | 1042 | 73.1092% | 66.7946% | 66.3638% | 47.3364% |
| bus | 1018 | 96.0407% | 83.3988% | 84.3094% | 50.9687% |
| traffic_light | 2368 | 96.5759% | 46.4527% | 49.6827% | 26.8659% |
| temporary_obstacle | 419 | 77.7042% | 84.0095% | 85.9983% | 40.3399% |

本機 ignored best.pt SHA `ba6d2f4269a7f3f95e4b92d7daff1f330f4203a4619e1f5902c752aa371f7155`；last.pt SHA `64b4896113869f8c248b3d8843e281b577154c3e93fabb1c95f2d42d37bcfc98`。權重與完整 validation_frames.jsonl 沒有加入 Git；frames SHA `ed611e3cf06583986f70ad4c1031d365964145ea7c9a2c3fa05d28e311104b32`。

CUDA peak allocated 3411.06 MiB／reserved 4550.00 MiB；training wall 10177.625s。每一輪 9600 原 train views，1470 bicycle-positive，half-scale 實際次數見 epochs.jsonl。

檔案 watcher 首次呼叫 checker 漏帶必要 --root，未執行核對即以 argparse error 退出；保留原日誌後，以明確 data/sunny-cross-town-v3-007 路徑執行同一個未改動的 checker 並通過。沒有重訓／推論或修改資料。
