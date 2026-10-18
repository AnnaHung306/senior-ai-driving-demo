# 物件尺寸與anchor形狀診斷

完整核對9600train/3600validation的可見物件GT，數量與獨立audit逐類相同。只讀標註，沒有模型推論、heldout或改動訓練。

| Split | 類別 | 物件數 | median寬×高px | 小框<32² | 短邊<8 | 現行shape IoU≥.7 | 加8px shape上界 |
|---|---|---:|---|---:|---:|---:|---:|
| train | car | 1621 | 37×19 | 60.02% | 1.73% | 42.50% | 45.22% |
| train | pedestrian | 2876 | 14×39 | 65.82% | 14.67% | 28.93% | 31.33% |
| train | bicycle | 1470 | 33×68 | 10.95% | 0.41% | 41.77% | 42.24% |
| train | motorcycle | 1283 | 17×26 | 66.10% | 7.64% | 45.75% | 51.99% |
| train | truck | 1933 | 49×31 | 37.56% | 4.50% | 51.68% | 53.34% |
| train | bus | 2534 | 68×42 | 20.96% | 0.59% | 51.14% | 51.18% |
| train | traffic_light | 3269 | 5×13 | 100.00% | 72.74% | 7.68% | 35.94% |
| train | temporary_obstacle | 1589 | 16×29 | 64.76% | 16.99% | 42.61% | 58.90% |
| validation | car | 946 | 57×22 | 43.76% | 0.53% | 55.60% | 55.60% |
| validation | pedestrian | 539 | 16×47 | 62.34% | 1.67% | 23.38% | 23.38% |
| validation | bicycle | 729 | 30×27 | 53.22% | 16.87% | 51.30% | 69.14% |
| validation | motorcycle | 845 | 33×44 | 34.56% | 0.24% | 41.18% | 41.18% |
| validation | truck | 1042 | 53×29 | 24.18% | 0.29% | 26.01% | 26.10% |
| validation | bus | 1018 | 50×27 | 36.44% | 12.67% | 53.44% | 53.44% |
| validation | traffic_light | 2368 | 5×12 | 100.00% | 81.42% | 7.09% | 44.97% |
| validation | temporary_obstacle | 419 | 23×46 | 43.20% | 11.46% | 53.46% | 65.16% |

使用安裝的TorchVision0.25官方AnchorGenerator圓整base anchors，現行size16/32/64/128/256、ratio.5/1/2。假設anchor中心可精確對齊GT，只量尺寸形狀的IoU上界；沒有feature grid偏移、proposal排名/預算、box regression或學習。RPN允許low-quality best match，所以低於.7不代表GT沒有正anchor。新增8px/ratio候選也不是已測得AP改善。

自行車val小框比例比train高；號誌大多只有數個像素寬。若目前8輪完成仍未達逐類gate，可分開驗證train小框採樣、額外小尺度anchor或aspect ratio。不能在同一個實驗同時修改資料/anchor/解析度後聲稱找出原因；不能藉尺寸分布降低門檻。

`train-candidate-views.jsonl.gz`只列現有train中小自行車或短邊<8的號誌視角，保留原RGB/GT/group。它是採樣研究索引，尚未接入sampler，沒有增加獨立場景或改變當前run。若新增多尺度anchors，需維持各pyramid level每位置anchor數一致、調整RPNhead與預訓練通道對應，並重測8GB容量/finite-gradient/ignore行為。
