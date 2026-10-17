# Phase 3 小段檢查存檔 — 2026-10-17 12:26 Asia/Taipei

使用者要求本輪只做容易中斷、立即接續的小工作。此檔更新當前進度；原完整流程、限制與未完成項目見 `CHECKPOINT_PHASE3_2026-10-17.md`。Phase 3 **尚未完成**。

## 已完成並落盤

1. 分割 `sunny-segmentation-v4-001` 已真正結束，原 exec68540 exit0；8 epochs、9600 updates，best epoch=6（第7輪）。重新載入輸出一致。最佳 SHA `a7031897c0ce35cc97f4ab035277c51e0a35558dccc225b69455a16cc6f9eb67`。
2. 新增純標準函式庫檢查 `scripts/check_sunny_segmentation_artifacts.py`；只讀已完成 run，不載入模型、不用 GPU、不開 heldout，輸出先寫暫存檔再 atomic replace。核對 best/last/config/validation/validation_frames 五個 SHA、27個凍結 source SHA與manifest digest、8輪完整紀錄和best選擇、3600個連續唯一 validation index、逐張混淆矩陣合計與每類IoU。**實際執行通過**。exit0只表示檔案一致，不表示模型品質通過。未重讀全部86400張raw，這一輪不能稱重新資料audit。
3. `evidence/phase3/sunny-v3/health-20261017/segmentation-v4-001.json`：val focus mIoU **.508456084924232** 達 .5；所有16focus支持存在。但 **wall IoU .0619479222、fence IoU .0072838274** 低於 .2，**visible margin F1 .4073333864** 低於 .5。品質仍失敗，不能進正式 heldout。
4. `input-preflight.json`：RGB-only fixture恰好9檔（8RGB+manifest），全部已存SHA相符，manifest只含RGB介面欄位且8view frame/time一致；phase config v1_1仍綁凍結補收plan SHA。沒有執行模型推論或採集。
5. `weak-class-diagnostic.json`：只從現有 validation 混淆矩陣分析。GT fence 約68.96%被判other、20.83%判wall，僅.888%正確；預測wall約36.89%來自terrain、35.42%來自other，真正wall只有6.96%。因此問題包含背景誤報與漏檢，不能只靠增加類別權重推定能解決。下一次先抽查 train/validation 對應材質與幾何錯例，再決定增加train城鎮內視角/材質多樣性或調整損失；以上是待驗證方向，沒有啟動新訓練或降低門檻。

## 仍在背景的程序

- 原 queue exec57077 仍活著，分割 completed 後已呼叫號誌 `train_light_state.py ...sunny-light-state-v1-001`。12:23建立的Python PID24688（shim27812，父queue27384）命令已經escalated CIM實際核對，勿重複啟動；PID只對此時有效，接續必須重新核對。
- 12:25時號誌 log仍0bytes、run目錄尚未建立，沒有任何epoch/checkpoint成績；只能說程序已啟動、仍在入口前置階段，不能說已開始更新模型或已完成。Python使用約588MB RAM、有CPU時間；啟動僅數分鐘，沒有充分依據判定卡死。後續讀新log/state/程序CPU差分再判斷。CARLA程序未見、GPU當次0MiB；後續會變動。
- detector v3-001仍排在號誌命令成功退出後。**不要在背景queue仍活著時再開相同run**。本輪不停止原工作。
- 12:27補存 `health-20261017/runtime-snapshot.json`：同一PID24688仍在，累積kernel+user CPU由前次約16.9秒增加至41.875秒，仍無run/epoch輸出；有進行CPU工作，但尚不能當成模型訓練進度。
- 中斷恢復時，先 `write_stdin` 查exec57077；跨app session則核對實際Python命令、檔案更新時間及failure.json。若程序已退出，先查失敗原因，再判斷從哪個階段接續，不能照舊紀錄盲目重啟。

## 可立即重跑的小檢查

```powershell
.venv/Scripts/python.exe scripts/check_sunny_segmentation_artifacts.py --run artifacts/runs/sunny-segmentation-v4-001 --output evidence/phase3/sunny-v3/health-20261017/segmentation-v4-001.json
```

只在分割run completed後使用，輸出必須位於run外。若任何檔案不一致會直接失敗，舊報告的時間不會更新；必須看本次exitcode及checked_at_utc，不得把舊報告視為新成功。本輪沒有修改模型、損失、訓練設定、凍結source或現有run。

## 接續優先順序

1. 用上述小檢查與live程序確認最新狀態。讓原號誌→detector隊列接續，不並行CARLA。
2. 分割品質尚差3項；先做validation/train錯例分析並保存，下一個訓練方案需另建版本。不使用Town04當train，不開lockbox找改善方向。
3. 號誌/偵測完成後另核對其品質與指紋；joint validation必須在phase/det任何lockbox marker前。黃燈補收、QA、demo、benchmark及最終導出仍未完成，詳原完整存檔。
