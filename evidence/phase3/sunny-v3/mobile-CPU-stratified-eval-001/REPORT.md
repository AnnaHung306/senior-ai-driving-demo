# 同精度分割train/validation診斷

固定每scene每10個saved bundle×8方向，共960train/360val。Seen train成績不是泛化評估；此subset不取代官方完整GPU gates。每20views落盤並已實測停下/接續。

| 指標 | Train CPU FP32 | Validation CPU FP32 |
|---|---:|---:|
| wall IoU | 97.3956% | 6.5270% |
| fence IoU | 68.9410% | 0.7389% |
| sidewalk IoU | 98.1025% | 45.7214% |
| terrain_or_ground IoU | 89.9107% | 61.8962% |
| focus mIoU | 86.6309% | 51.2267% |
| visible margin F1 | 95.5622% | 40.1914% |

相同360張val的原GPU BF16 mIoU 0.51215469，CPU FP32 0.51226740，差0.01127個百分點。兩者target support逐張完全相同。明顯train/val差距不因這個小數值差異而消失；材質/城鎮分布差異仍只是待干預驗證的原因假說。

原驗收仍wall/fence/margin失敗；不調門檻、不開heldout。此probe不批准新資料或新模型。
