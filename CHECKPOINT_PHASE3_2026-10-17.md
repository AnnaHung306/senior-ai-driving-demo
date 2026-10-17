# Phase 3 段落存檔 — 2026-10-17 10:11 Asia/Taipei

使用者要求先存檔以節省剩餘 token，之後接續。**沒有要求停止訓練**；下列兩個程序仍在執行，不能重複啟動。Phase 3 尚未完成，不進 Phase 4。長期規格 `PROJECT_SPEC.md`，目前協議 `PHASE3_TRAINING_PROTOCOL.md`。

## 已確認的現況

- 工作目錄 `C:\code\senior_AI`；Python `.venv/Scripts/python.exe` 3.12.14、Torch2.10.0+cu128、TorchVision0.25.0+cu128、RTX4070 Laptop 8GB。固定 **ClearNoon 理想晴天**，雨夜不列當前驗收。沒有 subagents、automation、PR/push。
- **CARLA off**；GPU訓練與模擬器分開。CARLA0.9.16必須DX12/Epic、沿用PixelReader background patch，不能回到失敗DX11/Low。RPC127.0.0.1:2000在sandbox不可達，所有CARLA RPC命令需 `require_escalated`；不要因sandbox網路問題重新安裝/重啟server。
- 主資料 `data/sunny-cross-town-v3-007` 已完成18場、2700bundles、86,400PNG（21,600RGB）。`audit.json` passed；105個物件GT panels、18語意targets、16相位crops已實際檢視，`visual_review.json` 綁auditSHA+review圖SHA批准，限assistant抽查。
- train Town01/Town03（8groups/9600views）、validation Town04（3/3600）、development Town05（3/3600）、lockbox Town02（4/4800）。**任何新模型都還沒有對development/lockbox評分。** 同路線多frame/相機/補收phase不算獨立路線。
- Tiny實驗 `sunny-segmentation-v3-001` 已刻意abort，不是完成或runtime exception；保留三完整epochs/7200updates，best val mIoU **.2936935108**。實際train/val selected wall IoU .9735/.00165、fence .7954/.00753，判定過擬合後換模型。舊權重/指紋/診斷/evidence保留；原exec1528與queue80306已退出。

## 正在跑的兩個程序

1. **分割訓練 exec session `68540`**：
   - `scripts/train_sunny.py --task segmentation --config configs/sunny_training_v4_1.json --run artifacts/runs/sunny-segmentation-v4-001`
   - 日誌 `artifacts/reports/sunny-segmentation-v4-001.log`。
   - source snapshot來自 **046ec3b**，位於run/source；config原始bytes、data/source/runtime指紋已凍結。
   - MobileUNet，COCO MobileNetV3完整encoder tensor transfer/FrozenBN、多尺度decoder與RGB stem，**3,589,953params**；8epochs/batch8，encoderLR5e-5/decoder5e-4、cosine。BF16卷積，參數/optimizer/loss/輸出FP32。paired scale/crop/pad255/flip只用train；val不變。
   - 80步overfit passed，loss ratio .4867074；重新初始化後正式訓練。
   - **第一輪 val focusmIoU .4436310527、visible margin F1 .3654591039**，仍未達標。wall .02817、fence .05079最弱。最近progress為epoch1（第二輪）/1800updates、meanloss .30591；檔案會繼續更新。
   - 第1輪wall_s=33883.922，保留原值；牆鐘有長時間跳增，不能據此推定純GPU吞吐。效能另用同步benchmark量測，沒有證實具體暫停原因。
2. **排隊 exec session `57077`**：
   - 等上述run `state.status=completed`，遇failure.json立即停止。
   - 接著跑 `train_light_state.py --root data/sunny-cross-town-v3-007 --config configs/light_state_v1_1.json --run artifacts/runs/sunny-light-state-v1-001`，log同名放artifacts/reports。
   - phase命令成功後跑 `train_sunny.py --task detection --config configs/sunny_training_v3_9.json --run artifacts/runs/sunny-detection-v3-001`，log同名。
   - 存檔時phase/det兩run尚不存在，**只是排隊，不能說已訓練**。phase若品質未過但native exit0，det仍可獨立訓練；先讀state/validation再決定。

以上session只對目前app session有效。新session需用實際程序與新日誌時間確認；Windows sandbox Get-Process/CIM看不到host程序時，需escalated read-only CIM。最初新訓練PID17468/shim27348僅供查核，不能未核對命令列就殺PID。`write_stdin`可確認native session仍活著，不能只用舊log假設存活。

## 測試與已備妥工具

- 71 tests完整通過，`phase3-sunny-tests-008.log`；後續phase joint precision/false-positive與tiny-box gallery修正已有4項相關測試通過。存檔前完整suite **72 tests passed**，`artifacts/reports/phase3-sunny-tests-009.log`；相關收據已進evidence。
- Mobile CUDA synthetic嚴格deterministic fwd/back/exact reload passed；batch8 allocated4055.69MiB/reserved4624MiB，steady synthetic step約.31sec。只是operator/capacity，不是accuracy。收據在evidence/phase3/sunny-v3。
- `predict_sunny.py`/`predict_perception.py`支援新Mobile模型；CUDA推論與訓練保持TF32-off/strict flags。`benchmark_sunny.py`、`benchmark_perception.py`、`export_sunny_results.py`已實作，**尚未用最終三模型走完**。
- 獨立RGB-only demo輸入已備妥：`artifacts/runs/sunny-rgb-only-input-v1-001`，只有8張RGB與 `inputs/frames/frame.json`，沒有labels/training/calibration檔。預選validation Town04-00第75bundle，非獨立測試。SHA證據 `evidence/phase3/sunny-v3/rgb-only-demo-inputs.json`。輸出請用新目錄。

## 號誌補收仍未執行

主lockbox Yellow只有4crops/1group，不能降低原quota。已凍結 `configs/light_lockbox_supplement_v1.json`：17個原Town02-00/01視角 × Off/Red/Yellow/Green =68RGB/204三模態PNG，**heldout-only、非訓練、非新增動態路線**。phase config v1_1綁planSHA。

GPU訓練全結束後再：

```powershell
.\scripts\start_carla.ps1
# 用新的carla.Client確認/切到Town02；已在Town02時不可冗餘load。
.venv/Scripts/python.exe scripts/collect_light_supplement.py --plan configs/light_lockbox_supplement_v1.json --output data/light-lockbox-supplement-v1-001
.\scripts\stop_carla.ps1
.venv/Scripts/python.exe scripts/audit_light_supplement.py data/light-lockbox-supplement-v1-001
```

start/stop與RPC需escalated；PowerShell ps1不要用LASTEXITCODE判定成功（可能null）。原生load_world曾assert `_episode!=nullptr`，server仍可已載入新town；先用新client確認，不能盲目重裝/重載同map。最好先用獨立map準備程序再開正式capture，以免native assert留下失敗capture root；失敗保留、新版本重收，不覆寫。

補收audit檢查完整PNG/actualpose/optics/time/nativephase、lamp world OBB+軸向depth及原色域資格；已知相位需至少2相容色像素比同pose Off亮>30，防燈殼假色；quota各類≥10、≥2原groups。**必須實際看qa/*.png**再寫 `visual_review.json`：`status=passed`、`audit_sha256`、`reviewed_images=[{path,sha256}]`，記reviewer/limits；不得只產生receipt而沒看圖。

## 接續順序與品質底線

1. 先確認兩個live程序及最新模型狀態，不重複train。繼續分割→phase→det。若val仍有類別不足，針對train/val改善，不看lockbox、不降低門檻；wall/fence可能仍需增加train Town01/03內的材質/視角多樣性，不能把Town04加train破壞城鎮切分。
2. 完成phase/det後，**先跑validation joint，必須在兩者任何lockbox marker前**：
   `scripts/validate_light_joint.py --run artifacts/runs/sunny-light-state-v1-001 --detector-run artifacts/runs/sunny-detection-v3-001 --root data/sunny-cross-town-v3-007`。
   綁兩checkpoint、validation files及joint report SHA。實際detector boxes，score≥.5、IoU≥.5；漏偵測算錯、偽號誌進precision與錯例，matched-red→green≤.02，不能用漏判稀釋。
3. GPU訓練程序真的退出、CARLA off後，完成上述phase補收/audit/實際QA。
4. 每個main model完成且val過gate才 `evaluate_sunny.py --run ... --dataset-root ABS`；如果工作目錄source有變動，使用 **該run/source/scripts/evaluate_sunny.py** 凍結腳本。這是來源快照replay，不能略過source checks強行用不同程式。head同理使用其source snapshot的 `validate_light_joint.py`/`evaluate_light_state.py`，參數皆指向實際資料root。
5. detector main heldout通過後才 `evaluate_light_state.py --root data/sunny-cross-town-v3-007 --run ...light... --detector-run ...detection... --supplement-root data/light-lockbox-supplement-v1-001`。dynamic與controlled joint分別保存，pooled phase用原門檻。一次marker不得刪掉重試或換模型後重看。若據lockbox結果改模型，需另補fresh lockbox。
6. 正式三模型RGB-only八方向推論：`predict_perception.py --detector .../best.pt --segmentation .../best.pt --light-checkpoint .../best.pt --root artifacts/runs/sunny-rgb-only-input-v1-001 --manifest inputs/frames/frame.json --output artifacts/runs/sunny-perception-demo-v1-001`。實際看surround.png；不能把舊Tiny預覽當最終圖。
7. 個別與完整八鏡頭benchmark，保留FP/FN與phase錯例；`export_sunny_results.py`產生模型卡/evidence/summary/report，更新registry/README/CURRENT_TASK，實際通過後才宣告Phase3完成。權重/data ignored但本機保留SHA，不宣稱已上傳。

主gate：detection mAP50≥.5、8類各P/R@.5≥.5；seg16 focus mIoU≥.5、各IoU≥.2、marginF1≥.5。phase crop macro≥.8、各recall≥.5、unknown≥.8、red→green≤.02；joint macro≥.4且matched red→green≤.02；phase每類train20、val/dev/lock10及各2groups。

沒有BEV/跨鏡頭融合/3D occupancy/learned control；這些是Phase4以後。實際目前只完成資料基礎與訓練中的二維模型，不能因已花很多時間就把品質失敗改成完成。


權重存檔核對：`last.pt` SHA已與state.json一致；最近完整epoch/持久化updates與正在進行的partial updates分別記在 `evidence/phase3/sunny-v3/checkpoint-20261017/receipt.json`，不把尚未存進checkpoint的更新當成已存權重。
