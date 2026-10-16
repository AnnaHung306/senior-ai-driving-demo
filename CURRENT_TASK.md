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
