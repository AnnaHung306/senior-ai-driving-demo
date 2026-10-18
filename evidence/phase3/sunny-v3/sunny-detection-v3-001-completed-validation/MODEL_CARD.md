# Detector v3-001：訓練完成、驗證失敗

8epochs / 19200updates；選第4輪（zero-based3），AP50:95=.363684，exact checkpoint reload通過。

mAP50 60.21%；mAP50:95 36.37%。

| 類別 | GT 物件 | AP50 | AP50:95 | Precision@.5 | Recall@.5 |
|---|---:|---:|---:|---:|---:|
| car | 946 | 69.47% | 47.47% | 88.27% | 63.64% |
| pedestrian | 539 | 57.54% | 35.06% | 84.06% | 49.91% |
| bicycle | 729 | 37.77% | 25.98% | 83.58% | 30.73% |
| motorcycle | 845 | 57.83% | 40.62% | 96.28% | 52.07% |
| truck | 1042 | 65.72% | 36.87% | 75.27% | 66.89% |
| bus | 1018 | 83.27% | 48.58% | 98.24% | 82.02% |
| traffic_light | 2368 | 41.85% | 19.80% | 92.62% | 38.18% |
| temporary_obstacle | 419 | 68.23% | 36.57% | 90.17% | 63.48% |

行人、自行車、號誌recall未達.5，不開development/lockbox。使用原train Town01/Town03、val Town04與凍結visible bbox/ignore政策。best SHA `9cbed7cfd641261396e916128535297ecdc267e0cf1d5bbc2af531bd15db5549`。

GPU allocated峰值3094.87MiB、reserved5300MiB；training receipt16622.94s包含載入cache等程序範圍；歷史待命事件另有runtime-observation，不宣稱純GPU步速。原始RGB 640x360；八類COCO初始化MobileNet/FPN自行微調；理想ClearNoon、逐鏡頭2D，不代表駕駛/BEV/3D。

補救實驗以原生小框診斷為據，提高內部resize到720x1280，再微調已選v3 best；仍需重新通過品質和actual detector-box phase聯合門檻。
