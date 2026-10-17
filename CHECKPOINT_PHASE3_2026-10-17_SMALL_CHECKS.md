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

## 後續小段：號誌JSON收尾修正

再次接續發現原queue57077已exit1：號誌v1-001其實完成16輪/832updates，但quality紅→綠比較回傳NumPy bool，寫state時TypeError。原run不修改、不提升為completed；保留failure、log、SHA、validation等於 `evidence/phase3/sunny-v3/light-state-v1-001-finalization-failure/`。修正僅在quality比較外加Python bool，不改訓練數學、資料或門檻。新增2%邊界pass/3%fail且可JSON序列化的回歸，相關5tests通過。後續用新v1-002重跑同config，接det v3-001；舊checkpoint保留作診斷，不視為正式完成模型。

新queue **exec56349** 已啟動：light-state-v1-002 → state completed檢查 → detection-v3-001。Log各自同名，沒有舊queue存活。CPU錯例panel程序 **exec74874** 正在跑 `scripts/review_sunny_segmentation.py`，輸出 `artifacts/runs/sunny-segmentation-v4-validation-review-001`。只選validation三scene每scene wall/fence最低IoU且GT≥512px，以及margin最低F1且GT≥50px。按舊GPU報告選例，CPU FP32只作視覺診斷，不取代GPU BF16驗收。校驗已綁inventory/index與所選raw SHA；逐panel atomic保存+catalog，重跑驗證identity後跳過已存panel。不能把產生圖片當成已看圖。

## 後續小段：分割錯例實際檢視

CPU錯例exec74874已exit0，產生8例/3pages；Town04-02沒有≥512px的fence候選，因此只有2例，不硬湊。已透過view_image實際看完3pages，`evidence/phase3/sunny-v3/segmentation-v4-validation-review-001/` 保存catalog、頁面、visual_review與prior-shift。標記是 **inspected_with_failures**，不是模型驗收通過。可見木/白色柵欄被漏掉，草地常判sidewalk並引出錯誤margin；wall支持有小/遠結構，需局部overlay再判材質。訓練/驗證有效pixel比例：wall8.719%/.247%、fence1.280%/.985%、sidewalk13.648%/3.854%、terrain4.362%/16.437%。這是train/validation分布差異證據，不是證明因果，也不准用validation城鎮回灌train。

15:40補正：先前Get-ChildItem空輸出不能證明run目錄未建立。新v1-002經Get-Item確認目錄15:35:59已建立，只是尚未寫coverage。PID33276讀取累計約320MB/81665次，仍有CPU/I/O活動；依程式順序應為完整verify或crop前置，沒有觀測堆疊，不能更精確斷言。CPU panel工具在原輸出目錄重跑exit0，已保存的8panels通過SHA後跳過，證實可續跑。

## 後續小段：固定樣本比對與號誌正式完成

- `probe_sunny_landscape.py`重用舊Tiny診斷的4個既定train/validation案例；CPU FP32重跑v4 best並逐例落盤。train wall .986128、train fence .906728、val wall .515145、val fence .007582；是刻意挑選樣本，不能當full train成績。已實際檢視contact sheet並保存 `evidence/phase3/sunny-v3/mobile-fixed-landscape-probe-001/`。train牆面是大片隧道，train fence是語意標為Fence的寬混凝土障壁；val木柵欄仍幾乎漏光，支持材質/幾何覆蓋差異假說。
- **light-state-v1-002 completed**，16輪/832updates，best epoch8（第9輪），訓練wall_s 6.171秒不含全資料verify/crop前置。SHA e4fdbc431dcf0db5d772ba0724ee74a00f4ea0fb525025c42db12493859ee60b。macro recall .94381170398，red→green 10/753=.01328021248，GT crop所有門檻通過，exact reload true。CPU逐張重算2368crops混淆與quality一致，source/plan/SHA核對，新舊best和last全部model tensors完全一致，沒有變更訓練數學。原v1-001 failure保留。
- 號誌11類型錯例/3pages已實際看圖，保存 `evidence/phase3/sunny-v3/light-state-v1-002-validation/`；標記inspected_with_failures，不把GT crop gate當detector joint合格。小crop細節不足，存在紅→綠案例；yellow val只有31crop/2groups，不能過度宣稱。
- queue **56349**已接續偵測v3-001；15:43:44已建立run/config/provenance/source，git_revision d3c2ef7，尚未epoch成績。模型程式尚未修改，不啟CARLA、不重開訓練。後續所有phase命令使用 **v1-002**，v1-001是失敗歷史。

15:48進度：detector v3-001真實訓練epoch0/100updates，meanloss .84613；overfit預檢已落盤。新增 `survey_sunny_materials.py` 純numpy/PIL、不載模型、不用GPU，對train/validation每scene bundle位置0/40/80/120×8方向共352views讀RGB+raw semantic，核對bound inventories/canonical indexes/selected raw SHA，逐view獨立atomic JSON可續跑。8個範例contact page已實看；全部352records、摘要與圖保存evidence/phase3/sunny-v3/material-survey-001。選到train混凝土/金屬網、val木板/白柵，不足以推斷全train完全缺類，下一步掃全train再決定是否補收。

全train survey程序 **exec19187**：`.venv/Scripts/python.exe scripts/survey_sunny_materials.py --root data/sunny-cross-town-v3-007 --training-all --output artifacts/runs/sunny-train-full-material-survey-001`，log同名放reports。全9600train +固定96val共9696views，15:50已存1137view。現工具加入--training-all及每group暖色/中性亮色代表圖選擇，新tool SHA綁新output，不覆寫原352view結果。原v1工具在commit ab4e748可追溯；原舊report若要續跑需用同一tool bytes，不能忽略identity變更。detector仍epoch0/300updates、meanloss .55927；兩程序一個GPU訓練、一個CPU讀圖，不啟CARLA。

## 後續小段：Windows survey寫入恢復

全量survey exec19187在1138張record後，atomic replace progress.json收到WinError5並exit1；確切鎖持有者未證實。資料records沒丟。只為survey.write加8次有界PermissionError retry，永久鎖仍raise且保留舊完整JSON；兩項專門回歸通過。初次測試的系統temp目錄受sandbox權限限制，改用workspace下artifacts/runs/test-temp後通過，不是產品算法錯誤。

為避免默默忽略source identity，先從git d197ca6讀原工具，驗證原toolSHA等於舊progress；以AST確認main/data計算/選例完全一致（只排除新增io_recovery元資料欄位），再備份before-io-recovery的原source/progress/failure，保存1138records SHA及新舊toolSHA receipt，最後只遷移progress identity。raw/records與模型run不改。evidence/phase3/sunny-v3/material-survey-io-recovery保留receipt/failure/regression。新續跑 **exec49830**，log sunny-train-full-material-survey-resume-001.log，15:54已超過2368views。

新增 `.\scripts\status_sunny_training.ps1` 唯讀各run，開檔使用FileShare.ReadWrite|Delete，不阻擋訓練atomic JSON replace；只讀newline完整epoch記錄。它只顯示已存紀錄，不推斷程序存活；可配合exec56349或escalated CIM查實際命令。15:54 detector epoch0/700updates，還沒有首輪last.pt/state，不要說checkpoint已存。

## 後續小段：全train掃描落盤、啟動與匯出檢查

- exec49830已exit0，全9696view完成；已實看17代表圖/3pages。evidence/phase3/sunny-v3/full-train-material-survey-001保存約2.4MB deterministic gzip全records、摘要、圖及review。wall192741757px、fence28297482px、terrain96425301px與凍結trainer統計逐一完全相同；Fence>=512px的train views分別Town01四groups526/455/839/767，Town03四groups均0。自動掃描全train，但只人工式實看17代表，不可宣稱已視覺檢查全部圖或證明所有材質缺失。
- start_carla.ps1新增assert_training_idle.ps1，先查本機感知GPU腳本實際命令及GPU使用量；訓練/評估或>2048MiB先拒啟，不能與train爭VRAM。escalated唯讀實測目前detector PID27224/9168被拒，沒有啟動/停止任何程序。
- export_sunny_results.py發現原本只驗證singular checkpoint欄位，整合benchmark三個SHA可被漏檢。現在逐模型綁定、同demo frame/8方向、正確測量數/有限正時間，要求det/seg/integrated三種收據齊全；validation JSON/frames SHA與checkpoint reload、final checkpoint和lockbox marker也核對。錯證據在建立output前拒絕。5項專門回歸通過，無GPU。尚未以最終三模型真的export，不能當Phase3完成。
- 偵測16:00仍epoch0/1600updates，meanloss .39615，首輪state/last還未產生；queue56349保留。模型/訓練src未改，phase後續用v1-002。

## 後續小段：RGB輸入收據與CPU全suite

- 新perception_inputs.py只讀嚴格RGB-only manifest，產生manifest與8張RGB各自SHA，另外綁helper source。predict_perception、benchmark_sunny與benchmark_perception都在推論/計時前記錄；hash計算不列timing。export除了模型tuple/frame也核對真實RGB bytes，避免相同frame編號下混入不同圖。實際預選RGB-only fixture8圖核對通過；2input tests與6export regression通過，含同frame改RGB/GT欄位拒絕/錯收據不先建output。未執行真實三模型demo或benchmark。
- CPU-only完整suite010：**Ran83, OK skipped3，即80通過/3CUDA跳過**，9.316秒。CUDA_VISIBLE_DEVICES=-1僅test子程序設定，沒有改既有GPU訓練；不把skipped說成GPU測試通過。log保存在evidence/phase3/sunny-v3/perception-input-binding/cpu-suite-010.log。
- 16:05 detector仍epoch0/2100updates，meanloss .37545，queue56349活著，還沒有首輪last/state。CURRENT_TASK已重新整理為最新現況，舊節點全部保留本checkpoint。

16:09小段：新增analyze_sunny_phase_confidence.py，純JSON/CPU，不開模型或heldout；先核對best與validation兩SHA、2368逐張混淆，再計10bins top-label ECE .007816與離線棄權探索。紅→綠10例confidence最大.7563；門檻.9時此val紅→綠為0，但保留率84.38%、含棄權macro recall74.75%（不能忽略被棄權GT），.999則全棄權/undefined accuracy，報表以—處理。沒有採用門檻、修改模型/gate或將棄權當成原unknown語意類。此分析只描述少數相關路線，不是安全機率/獨立泛化CI。JSON與REPORT落evidence/phase3/sunny-v3/light-state-v1-002-confidence。偵測首輪訓練更新已2400，meanloss .36573，尚未完成首輪validation/state。
