# Phase 3 可見標註／動態擷取協議

此迭代是偵測訓練的前置工程，實測結果與圖片見 [REPORT](evidence/phase3/visibility-v1/REPORT.md)。不是 detector training、自然資料集驗收或 Phase 3 完成。

## 原始資料與來源邊界

- 沿用已核對的 CARLA 0.9.16 background patch、DX12 / Epic、8×640×360 / FOV90。先 Phase0 doctor/smoke，Town01 已載入時不 reload；單一 tick client。
- 四個 raw modalities 共32 sensors。Callback 立刻複製像素與 image.transform；owned immutable payload 不保存 native image。深度及 instance 必須保存原通道，不套 palette／灰階／JPEG。
- 模擬 snapshot F 的 actor GT 在下一 tick 前複製，等 F+1 readback，仍按 image.frame=F 配回 snapshot F。後續模型只讀 RGB manifest；actor／world pose、depth／semantic／instance 均在 labels。
- 動態 collector 同時保存實際32個 sensor.attributes、每模態 capture world matrix、RGB-only feature manifest、rig K/T；audit 比較各模態姿態與 snapshot×rig。一致性成立才擴充長 episode，不能只拿靜態測試推論動態配對正確。
- 受控 fixture 的位置、角色與五種干預先在 configs 凍結；受控重疊車不是安全駕駛測試。短動態 probe 僅5 bundles /0.8s，正式 v3 的 scene/group／長片／coverage 門檻沒有因此通過。

## 可見標註政策

目前 active candidate：configs/visibility_policy_v1_1.json；v1 保留。ID低位解碼、near-plane box clipping、exclusive-max mask box、axial/range區分、Rider聯合、positive precedence／ignore 邏輯有獨立單元測試。

原始 mask 決定 visible box；3D amodal projection 另存。CARLA bbox 可能不完全包含邊緣鏡面／外觀，所以用和深度一致的0.15m expanded OBB支持區作一致性檢查；不把觀測可見框改成較漂亮的投影框。Projection margin仍2px；depth支持至少95%，semantic disagreement最多2%；所有值在policy與geometry source中追溯。

32 pixels／4px side／兩輪車8個body pixels門檻只批准候選標註；rider-only、near-plane、未知／不支持前景要保留 ignore mask/regions。不可見 actor 不輸出 positive，也不能將其 amodal box整塊ignore，否則會遮掉前方真實物件的訓練訊號。

ignore_background_proposals 是 framework-neutral 契約，僅 background labels可變 -1，positive match優先。**後續真實detector必須在RPN及ROI兩處接入、驗證未標前景不成負例**；目前尚未完成整合。未解析交通號誌／標誌等不能默默當背景。

所有 fixture export `training_approved=false`。只有新自然資料完成 acquisition／full disk audit、獨立split、類別／episode配額、stratified視覺QA及ignore-aware trainer整合後，才能另立批准收據；不得只改旗標強行匯入。Rider分類／truck／bus／light state／temporary props需另驗資產taxonomy並版本化，原raw GT保留。

## 已執行命令記錄

以下目錄／revision已存在；重做需先取全新data/run ID與新的diagnostic config版本，不能覆寫。原始capture source revision與後續auditor／policy revision不同，各收據保留自身hash。Sandbox連不到host RPC時使用允許本機loopback的執行環境；不要因此重裝CARLA或改圖形API。

```powershell
.\scripts\start_carla.ps1
.venv/Scripts/python.exe scripts/verify_phase0.py --run-id phase0-before-instance-008
.venv/Scripts/python.exe scripts/probe_instances.py --output data/instance-probe-v1-001 --baseline artifacts/runs/phase0-before-instance-008/report.json
.venv/Scripts/python.exe scripts/audit_instance_probe.py data/instance-probe-v1-001 --revision v3
.venv/Scripts/python.exe scripts/export_visible_instances.py data/instance-probe-v1-001 --audit audit-v3.json --policy configs/visibility_policy_v1_1.json --output visible_instances_v1_1

# Matched pair: identical static geometry and 80-tick sensor warmup, different weather presets.
.venv/Scripts/python.exe scripts/probe_instances.py --output data/instance-weather-noon-v1-001 --config configs/instance_weather_noon_v1.json --baseline artifacts/runs/phase0-before-instance-008/report.json
.venv/Scripts/python.exe scripts/probe_instances.py --output data/instance-weather-night-v1-001 --config configs/instance_weather_night_v1.json --baseline artifacts/runs/phase0-before-instance-008/report.json
# Both roots audited independently before evaluation; then stop simulator for GPU model work.
.\scripts\stop_carla.ps1
.venv/Scripts/python.exe scripts/diagnose_illumination.py --output artifacts/runs/illumination-diagnostic-v1-001

# Separate fresh server, Phase0 regression, five-bundle moving 32-sensor probe.
.\scripts\start_carla.ps1
.venv/Scripts/python.exe scripts/verify_phase0.py --run-id phase0-before-moving-instance-009
.venv/Scripts/python.exe scripts/collect_dataset.py probe --criteria configs/motion_instance_probe_v1.json --plan configs/motion_instance_probe_scenes_v1.json --output data/motion-instance-probe-v1-001 --probe-scene 0 --probe-bundles 5
.venv/Scripts/python.exe scripts/audit_motion_instances.py data/motion-instance-probe-v1-001
.\scripts\stop_carla.ps1
```

每個native command都必須確認exit code後再進下一步；PowerShell cmdlet / ps1的錯誤則看ErrorAction/exception，不把未設定的LASTEXITCODE當成功或失敗。原dataset-collect完整auditor仍拒絕probe，不能用本次probe audit取代完整dataset驗收。

## 凍結模型的診斷限制

illumination_diagnostic_v1.json在模型評分前凍結，checkpoint hash固定，0 updates /0 selection。各40 views為同一已見道路的5干預×8鏡頭，不是40獨立scene。Actor/camera幾何差0；raw semantic target仍差0.0982%，完整weather preset混合低光、雨、霧等因素，不能聲稱雨的單因果效果或新泛化成績。兩次靜態資料只固定預設camera source/build與80ticks warmup，未另存完整exposure attributes；後續moving collector已補actual sensor attributes，未來long captures另存完整weather attributes。

原模型既有26.22% test保持歷史結果；新診斷47.56%/2.37%不能拿來宣稱模型改善或測試退步。其用途是決定新資料／實驗優先序，後續新lockbox要保持模型評分未揭露。
