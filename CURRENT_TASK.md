# 最新實際狀態

007完整audit已通過86400張、105物件GT抽查/18語意圖/16 phase crops已由assistant實際檢視，visual_review.json已綁定批准；不是人工認證。CARLA off。準備正式segmentation/detection training configs/sunny_training_v3_9.json。phase lockbox黃燈僅4crops/1group，必須額外正式補收才能phase驗收，不能放寬門檻。

# 目前任務（2026-10-16 晴天 Phase3）

使用者要求這次盡力完成整個Phase3才結束；明確允許全程理想晴朗白天，雨夜需求移除。Phase3尚未完成，不進Phase4。請持續做到實際訓練/評估，不能再只交小probe。

## 現在真的在執行

- canonical資料 data/sunny-cross-town-v3-007，凍結計畫 configs/dataset_sunny_v3_8.json。18個30秒/150bundle×8方向×4模態，預計86400PNG；train Town01/Town03共8groups、val Town04 3、development Town05 3、lockbox Town02 4。
- **18場全部採集通過：2700 bundles /86400 PNG；007沿用006的17個已過場景，最後一場重新採集後自車76.716m、collision0、全部移動/清理門檻通過。** 最後失敗原因全時間線證實134/150樣本Red/brake，不能只看首尾兩張Green。首次綠燈介入被TM set_random_device_seed 的 ResetAllTrafficLights覆蓋；修到TM seed後才freeze→tick→setGreen→tick，開拍確認24燈Green/frozen，逐幀檢查不變。原失敗005/006不轉pass。
- **CARLA已由stop_carla.ps1停止project-owned launcher29748/shipping16004，GPU已0MiB；不要再做RPC。** 目前CPU完整audit在跑，exec session26202，日誌artifacts/reports/sunny-cross-town-v3-007-audit.log；只靠write_stdin確認存活/完成，不能用舊檔假設程序活著。尚未批准training。不要在audit進行中修改auditor/geometry/light-label sources，避免hash和實際runtime不符。
- CUDA synthetic三模型已過strict deterministic finite forward/backward；artifacts/reports/sunny-models-cuda-001.json，batch2 peak allocated detector1566MiB、seg853MiB、phase69MiB，尚非真實訓練/accuracy。65 unit tests已過（phase3-sunny-tests-006.log）。
- 005沿用004六個完全相同scene的通過資料，raw PNG hardlinks、metadata copies；report.reuse_lineage存hash。其餘實際新增採集，沒有把failed scenes改成pass。完整18成功後仍須audit及GT抽查，尚未批准training。

## 本輪已做但仍需真實GPU/資料驗證

- 五town32-sensor卡車raw15、公車raw16、trafficcone01 raw21 taxonomy工程試驗通過。視覺已確認trafficcone01實際是橙白交通桶，模型類temporary_obstacle，不宣稱涵蓋全部施工錐。evidence/phase3/sunny-v3/assets-contact.png 10 panels已看；fixtures永遠training=false。
- configs/dataset_sunny_v3_1..v3_6保留幾何及controller修正歷史：v3.1 Town03 collision29；v3.3 TM desired_speed=0導致NaN（上游MotionPlanStage:159除以targetvelocity）；0.1floor與finite actuator guard後20bundle短回歸通過。003低卡車位移失敗不可斷言zero-limit原因，004profile無缺值但通過。Town03-02卡車彎道spawn失敗已移直路，餘下12場原生預檢全過。
- 新detector: TorchVision0.25.0+cu128、official COCO權重SHA907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5。自訂MobileNet FPN strides4/8/16/32/64、小anchor16..256；backbone與高層FPN/RPN transfer，新低層FPN+classhead再訓練。真實RPN/ROI ignore、resize、positive precedence、all-ignored differentiable zero已測。Windows無Triton且is_compile_supported(cuda)=False，使用官方eager deterministic ROIAlign，ROI128/sample ratio2限制記憶體。CPU三模型forward/backward finite；新17class/Dice/empty-loss增補後還需總測及CUDA probe。
- 新分割17類（RAW_TO_SEG見sunny_data.py）：road1/sidewalk2/ped3/rider4/car5/truck6/bus7/moto8/bike9/line10/light11/wall12/fence13/pole14/verified-temp15/terrain16，other0、ignore255。**raw4 Walls、5 Fences、6 Poles、11 Sky、10 Terrain、25 Ground**已依官方enum鎖測；草稿誤碼在任何訓練前修正，沒有錯版權重。temp只依經OBB-depth/semantic驗證的allowlisted actor IDs，不可把全部raw21當障礙物。
- road_boundaries.py輸出predicted road/line↔sidewalk/terrain可見界面與2px boundaryF1，不推斷遮擋後路界；sunny_losses.py weightedCE+0.5 foreground Dice（ignore255、absent classes處理）。
- 號誌head LightStateNet RGB crop32×64，phase_target以world phase為GT、至少2個semantic-lamp pixels有互斥發光色才標red/yellow/green，其餘unknown。configs/light_state_v1.json凍結min counts/group、macro/perclass recall與red→green gate。**尚未知main資料是否各phase覆蓋夠；若不夠要正式補收相位介入資料，不能把舊controlled fixtures轉train。** 現有train_light_state.py只支援main crops，尚未實作aux collection。
- audit_sunny_dataset.py全PNG CRC/SHA+sensor attrs/pose/timestamps、可見候選/GT燈具world OBB-depth、per split每object10views/2groups及semantic15focus支持、verified-temp pixels；含phase crop coverage統計但不把缺phase偽造通過。audit自動pass仍training=false，還需visual_review.json statuspassed/training_approvedtrue/auditSHA+reviewed_images hash，sunny_data.verify會重查所有raw+metadata+review images。
- train_sunny.py（det/seg）、predict_sunny.py（RGB-only、optional phase head）、evaluate_sunny.py（val gate後一次heldout）、benchmark_sunny.py、verify_sunny_models.py；完整source/config/data hashes、epoch checkpoint/resume、RGB-only reload、FP/FN gallery與小框/truncation recall。**訓練最終草案 configs/sunny_training_v3_9.json 指向007**。16→17類新增及Dice/road margin後還沒重跑完整suite；前輪63tests過，個別empty-ignore/taxonomy也過。code未commit。
- 品質門檻在任何新model score前：mAP50>=.5，score.5每類P/R>=.5；seg focusmIoU>=.5、每focusIoU>=.2、visibleMarginF1>=.5。val選模；不達標不開lockbox、不降低門檻。state head另有預凍結gate與joint實際detector-box召回。不可把舊26.22%當新成績。

## 接下來必須做完

1. 007已完整採集、CARLA off、CUDA probe passed；繼續監看CPU audit。若有新scene失敗保留並根因修正，盡量用已通過同geometry/criteria資料lineage沿用，勿重跑一切。
2. 跑完整audit（python scripts/audit_sunny_dataset.py data/sunny-cross-town-v3-007）；若coverage/geometry問題另版修正，不偽裝pass。看QA/逐class snippets後寫綁定receipt。phase coverage不足要另立正式補收計畫。
3. 真正det/17-class seg/phase訓練與val選模。先verify_sunny_models.py --device cuda（8GBVRAM，source hash/strict determinism），實測容量再選batch；OOM保留失敗、改新config/run，勿覆寫。正式train_sunny參數與readme草案。本輪將先commit訓練基礎，再train以留source revision。不要訓練讀lockbox。
4. 各模型val通過後一次heldout、逐類mAP/P/R/IoU、boundaryF1、state crop+detector-box joint、實際RGB-only推論與efficiency、FP/FN案例/模型卡；據結果真正完成Phase3。

---

以下保留前輪狀態作歷史（其中下一步/程序停止等描述已被上方更新取代）：

# 目前任務

更新：2026-10-16（Asia/Taipei）；長期需求見 PROJECT_SPEC.md。

**Phase 3 進行中：首輪 segmentation、可見 instance 政策、靜態／動態32 sensors與配對天候診斷已完成；整個Phase3尚未通過，不進Phase4。** Phase0、Phase1初始規則Baseline、Phase2 v2基礎資料驗收已通過。

## 最新一輪成果

- 四次工程採集：controlled五案、matched noon、matched rainy night、短動態，各5 bundles /160 PNG，共640 unique raw PNG（160 RGB）。全部 training_approved=false，不是已收完的v3。
- Static32 sensors=八方向RGB/semantic/depth/instance，timestamp / modality pose error 0；涵蓋occluded/revealed/partial/truncated/near-plane。Phase0回歸008與009通過，patched CARLA0.9.16 / DX12 / Epic沿用。
- 已知instance解碼R=semantic、ID=G+256B；拒絕目標ID高位截斷或重複。Rider13與兩輪車body19/18共用ID實測；visible box合併同一verified instance的rider+body。原模型把Rider歸other，原bicycle IoU不是完整instance AP。
- 深度軸向支持率99.657%，range假設76.812%；不可把depth直接當歐氏距離。移開遮擋車後front行人14→212pixels，partial48。bbox fill不是occlusion fraction。
- visibility_policy_v1_1：actual tight mask xyxy；min32 pixels/min4px side，兩輪車body>=8px；near-plane/rider-only/未解析前景明確ignore。v1誤排兩張真實邊緣車輛，v1.1投影與depth一致使用0.15m OBB support envelope；raw visible/amodal boxes不改，v1及audit-v2保留，新增audit-v3。
- 原static160 actor-view pairs=34 positive/4 ignore/122 not visible，非獨立物件數。proposal background ignore / positive precedence契約已測，尚未接真正detector RPN/ROI；不可批准fixture訓練。
- 短動態probe：BasicAgent自車0.4196m、span0.8s、ego collision0。160 PNG全CRC/SHA/同步通過，capture vs snapshot×rig平移差最大1.839e-5m、rotation matrix差4.22e-7，各模態pose差0。這是模擬內部一致性，不是實體校正精度；部分NPC短暫停住，非長episode motion coverage。
- Collector支援第4模態、owned capture-time pose與32 sensor attributes，GT world transforms只進labels。原snapshot/一tick readback、暖機後導航及TM cleanup次序保留。
- 44 tests通過，全部640 PNG封存時再驗SHA。Assistant檢視static含角色17圖/38 visible object views、動態首尾及配對模型圖；不是人類認證。最大case-end GPU樣本3938MiB，非連續peak，不能稱20Hz即時。
- 原凍結模型paired diagnostic：ClearNoon focus mIoU47.56%、HardRainNight2.37%；0 optimizer updates、pose差0、semantic targets差0.0982%。同一已見train road、少量相關cases，非新的泛化成績，也不是雨單因素因果結論。

主報告：evidence/phase3/visibility-v1/REPORT.md；執行規格：PHASE3_VISIBILITY.md。原始資料：data/instance-probe-v1-001、data/instance-weather-noon-v1-001、data/instance-weather-night-v1-001、data/motion-instance-probe-v1-001。配對模型圖：artifacts/runs/illumination-diagnostic-v1-001。原始圖ignored，收據/指紋/QA圖進evidence。

## 上輪模型保留

segmentation-v1-002，source a1eff7b，config segmentation_v1_1.json，Tiny U-Net 488120 parameters，16 epochs/4800 updates，best epoch12；validation27.50%、原test26.22%。原test夜雨11.83%、自行車recall20.21%/precision8.10%，不可接控制。checkpoint SHA f06e4b110ea767d2c4b910d2cbfa8f9ea3ef96fa807c950bd2afd5f54947a1d4未變。本輪沒有新detector、重訓或重跑原test；要重現原訓練須用原source revision，不可目前source強行resume。

## 下一個單一任務

準備並凍結v3跨城鎮幾何/split/長episode收集計畫：先驗證Town03/Town04/Town05/Town02載入、導航/人行道與32-sensor資源，現僅assets已知，未RPC/render新城鎮。擬train=Town01/Town03、validation=Town04、development=Town05、sealed lockbox=Town02；仍待feasibility probes及正式route/quotas凍結，Opt與非Opt同一地理群組。

補truck/bus、號誌物件/相位、temporary-obstacle的asset taxonomy/visibility QA；明確保留Rider或另立版本映射，不能靜默改舊指標。以新groups、30秒episodes、多種天候、空目標負例補資料，實作真正RPN+ROI ignore integration後才批准detector training。Proposed18groups/86400PNG尚未收集；原test與本輪diagnostic都已揭露，只能development，正式lockbox與品質門檻另凍結。詳configs/dataset_expansion_v3_plan.json。

## 不要重犯

1. Town01 已載入時沿用地圖，不冗餘 reload。新城鎮的載入與幾何須另做 probe，不能沿用 Town01 座標。
2. PixelReader background patch SHA256=b0370d6dcd57b95299d12005cebad1dc6187804e83aeb51a184ccbc4b581fea8；保留 stock 備份與 exact-build guard。使用 DX12 / Epic；不要回到已失敗的 DX11 / Low。
3. sensor readback 按原 snapshot frame 配對，不拿下一 world tick 冒充當前 sensor frame。
4. NPC cleanup 先取消 autopilot，world restore 前保持 TM sync，最後才關 TM sync。
5. 新 actor 首次 tick 前 get_location 曾為 (0,0,0)，導航必須在暖機 snapshot 後初始化。
6. GT 只在訓練／評估，推論只用 RGB；BasicAgent 是規則。semantic road 不是可導航區域，other 不是空白安全空間。
7. 6 個短場景不是 2400 個獨立樣本；validation 1 group / test 2 groups 無可靠泛化信賴區間。沒有 AP、偵測器、號誌狀態、3D / BEV、導航介面或 AI 駕駛閉環。

目前模型訓練、CARLA 與資料程序均已結束，沒有監控排程或其他背景工作。
