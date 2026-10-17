# 目前任務：持續做可中斷的小工作，完成晴天 Phase 3

使用者最新要求：可以一直做到被強制停止；每個小段落盤/commit。詳細順序、坑與歷史見 **CHECKPOINT_PHASE3_2026-10-17_SMALL_CHECKS.md**，原完整協議見 **CHECKPOINT_PHASE3_2026-10-17.md** / **PHASE3_TRAINING_PROTOCOL.md**。本狀態截至2026-10-17 18:50 Asia/Taipei。

- **唯一背景訓練queue exec56349**：號誌v1-002已成功退出，現跑 `train_sunny.py --task detection --config configs/sunny_training_v3_9.json --run artifacts/runs/sunny-detection-v3-001`。18:50 epoch2/5100updates（第3輪）；第2輪已存last/state，best仍第1輪 AP50:95=.30769、mAP50=.56836；4項逐類門檻不合格，尚未完成run；不要重複啟動。det source凍結git d3c2ef7，原queue57077已失敗退出。
- 分割v4-001 completed：8輪/9600updates，best第7輪，val mIoU50.85%；wall6.19%、fence.73%、marginF1 .40733仍失敗，不能進heldout。權重/27source/3600逐張混淆均核對通過。
- 號誌 **v1-002 completed**：16輪/832updates，best第9輪；GT crop macro recall94.38%、red→green10/753=1.33%，gate passed。actual detector-box joint待做。v1-001僅JSON NumPy bool收尾失敗，failure/權重保留；型別修正後新舊best/last tensor相同。後續命令用v1-002。
- 全train材質survey已完成並實看17代表圖：9600train+96固定val，全9696records壓縮存evidence/full-train-material-survey-001。三類train像素總數完全等於frozen統計；Fence>=512px只在Town01四groups，Town03四groups為0。代表train多混凝土/網、val木/白柵；不是全圖人工材質標註。先據此規畫train-only補景，不用val/heldout補train、不盲目加權重重訓。
- 所有CPU掃描/圖片程序已完成；exec19187失敗已由49830續跑完成。Windows短暫replace失敗的8次有界重試/兩項回歸、原1138records與AST一致恢復收據已保存。status_sunny_training.ps1以ReadWrite|Delete分享讀取狀態，不把檔案紀錄當process活著。
- 最新**83 tests discovered，80 passed / 3 CUDA tests skipped**，CPU-only suite010（CUDA_VISIBLE_DEVICES=-1只用於該test子程序，不影響現有GPU訓練）；之前GPU suite72通過仍保留。8RGB實際輸入指紋核對通過；demo/個別與整合benchmark會綁manifest與8RGB SHA，export拒絕錯模型/輸入/測量收據。尚未以最終三模型真的demo或benchmark。
- **CARLA off / ClearNoon**；start_carla.ps1新增training/GPU busy前置拒啟，實際busy branch已驗證。黃燈heldout補收204PNG仍未採集；全部新模型development/lockbox未評分，joint validation必須先於det/phase任何lockbox marker。
- 下一步：保存本段變更；持續看det首輪驗證逐類結果（勿在GPU訓練中再開GPU推論/測試/CARLA）；完善train-only景物補收計畫。完整3模型驗收/補收/QA/demo/benchmark/export仍未完成，Phase3不算完成，不進Phase4。

- 新增純JSON號誌置信度分析，2368crops驗證輸入SHA/混淆一致；門檻.9保留率84.38%、含棄權macro recall74.75%，沒有更改部署/gate或把棄權混同unknown。報告evidence/light-state-v1-002-confidence。

- train-only景物scout工具與PHASE3_SEGMENTATION_RECOVERY.md已準備；2項offline邊界/原生幾何序列化測試通過，未連server/未採圖。等GPU程序退出才查Town01/03，inventory不是可訓練資料，不能假裝補景完成。

- CPU系統性分割抽查：exec99969已exit0（1320全完成），evaluate_sunny_segmentation_cpu.py --bundle-stride10，選每scene每10bundle×8views，960train+360val=1320。每20view atomic chunk；先max-new-chunks1成功存20張退出，再接續第一新log=40張，18:41已380張。輸出artifacts/runs/sunny-mobile-CPU-stratified-eval-001；不能與官方GPU gate混用。原smoke26912已exit0，勿再開同output並行。
- 偵測首輪權重與27source已另存本機artifacts/runs/sunny-detection-first-epoch-snapshot-001，best/last為hardlinks，原trainer後續atomic replace不覆寫snapshot；evidence/detection-first-epoch-001有SHA與完整數值。初次sandbox硬連結WinError5後native重試成功；沒有改權重/停訓。16:12→18:38牆鐘跳增保留，不推定原因/GPU速度。

- 最新CPU診斷完整落盤：960train/360val，wall IoU97.40%/6.53%、fence68.94%/.74%、marginF1 95.56%/40.19%。同360val CPU/GPU mIoU差.01127百分點；evidence/mobile-CPU-stratified-eval-001含66chunks gzip與報告，不是新gate。
- 系統event506確認16:13 Idle Timeout進入Modern Standby，event507於18:34鍵盤喚醒，已保留runtime-observation。新增防閒置待命helper **exec8890**，native PID14152，綁定訓練PID9168及精確creation FILETIME；18:48啟動，目標退出或最多6h自動清除。2秒短子程序自動解除實測已通過。powercfg /requests需OS管理員故不可讀；只確認API接受與receipt，未改電源方案/螢幕。勿重開同target helper。
