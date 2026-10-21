# Phase 10 問題登錄（2026-10-21）

本輪稽核完成，正式品質仍未通過。P1阻礙可靠駕駛／品質結論，P2影響穩定性或可觀測性，P3影響呈現。已修復表示所列觸發與回歸通過，不能推論所有類似情境都已解決。證據根目錄為 `evidence/phase10/system-audit-v1-001`；完整分母見 [PHASE10.md](PHASE10.md)。

| ID／級別 | 觸發、實際影響 | 處置與驗證／尚待處理 | 證據 |
|---|---|---|---|
| T10-01 P1 未解 | 六條依地圖預先選定路線只有2/6抵達；左右轉共有三次保守停止及一次90秒逾時 | 保留原失敗，不以受控成功案例改寫完成率；任意起終點可靠性不成立 | 主評估、六案report、catalog、全部路線圖 |
| T10-02 P1 未解 | 號誌ROI與所行車道的關聯／紅燈停止記憶可能造成誤停；101個ROI可解析、72個不可解析；100個SIGNAL_STOP樣本附近找不到本車地圖停止點候選 | 不能把此代理直接判為誤闖或誤停。四次停止點穿越代理沒有穩定紅燈，但不足以認證遵規；需新路口方向／可見性／滯留矩陣 | `phase10-signal-association-final.json`、signal map002、原始phase observer |
| T10-03 P1 未解 | right2道路gate低於.6；RGB soft、hard segmentation及同幾何GT oracle都只有.4667/.4667/.5333 | 已排除「只要改善分割即可解決」；該幀末切線前向比.6297未達.7，實際沒有啟動終點延伸，因此延伸假說也不成立。平地投影、視野盲區、地圖對齊與曲路取樣仍待分開驗證；未放寬gate | road-gate-oracle、road-context-hypothesis、F30328record |
| T10-04 P1 未解 | 汽車常被高信心誤判卡車；77個可見car GT，class-aware R@.5僅5.19%；不分類最高分配對44/77為truck | 保存錯例、分類混淆與分數可靠度；先補同資產不同距離／角度／遮擋的平衡診斷及負例，再決定資料修正，沒有盲訓或把錯類隱藏在UI | detection-confusion、per_image、補充contact0；兩配對協議不同不能直接互換TP |
| T10-05 P1 未解 | traffic-light偵測P91.2%但R17.30%；joint號誌macro Recall42.37%，GT crop85.67%；unknown→green 173個 | CPU凍結head診斷不是CUDA bit-exact；需審核可觀測性與燈頭方向。unknown→green不能直接稱真實闖紅燈 | light evaluation、exact head_inputs.npz與samples |
| T10-06 P1 未解 | 3D嚴格中心TP26/51，IoU.25僅2/51；匹配yaw平均誤差51.27° | 缺陷涉及類別、召回、尺寸與朝向；條件中心MAE.531m無法代表完整方框品質。不能用放大框或GT替換主展示 | matched_3D、world3D counts、錯例圖 |
| T10-07 P1 未解 | 相同33個class-correct 2D TP crop，learned 3D IoU .10445低於尺寸先驗.24170，中心也無改善 | 不依同一audit直接選新模型／切換prior；先設獨立姿態與距離分層驗收，再考慮3D資料及loss | crop_ablation、主評估 |
| T10-08 P1 未解 | 2秒CV預測僅覆蓋14/41可評GT；條件FDE均值1.454m，但max4.830m高於靜止基準max2.356m | 同配對基準、全GT覆蓋、消失與右設限分開報告。Kalman／CV不能保證轉彎或急停；稀疏抽樣ID switch1、間隙重識別改ID3不是完整HOTA/MOTA | forecast_samples、velocity_samples、forecast圖 |
| T10-09 P1 未解 | 無GT循環257次，p50 1193.94／p95 1303.59ms，超過5Hz即時預算200ms | 靜態投影cache與PNG優化已採用，但端到端仍不即時；需分別量測八鏡頭detector、保留號誌head、時序融合及I/O，不能以60FPS或5Hz回放冒充推論頻率 | timing、performance-comparison、兩paired benchmarks |
| T10-10 P2 部分改善 | 靜態projection重複計算；PNG level6耗時 | 真觀測3次同輸入所有fusion陣列exact，p50 516.02→78.63ms（僅此段6.56倍）；12圖×3 paired PNG 44.54→24.43ms，bytes×1.555且解碼exact。整路線GT／負載不同，不能把總差異全部歸因優化 | fusion/png benchmark、optimization config |
| T10-11 P1 已修復 | NaN/Inf road evidence、時間、路線或confidence，舊版12個案例有8個被接受 | 明確finite／時間／shape防護，12/12拒絕或煞停；保留before／after與測試 | numerics-before/after、safety tests |
| T10-12 P1 已修復 | 對非單調x彎路使用np.interp，可能漏掉曲線上的障礙 | 改線段投影與弧長，障礙相對本車未來曲路驗證，含圓形保守外接；數值回歸通過，不宣稱已證明所有彎道避碰 | navigation_safety、corrected-planner tests、source snapshots |
| T10-13 P2 已修復 | 正當紅燈停超過12秒被誤判stall | ProgressGuard對SIGNAL_STOP暫停累積，恢復後重置寬限；實際15秒紅→綠轉彎抵達236幀 | long-red director/run、safety tests |
| T10-14 P2 已修復 | Windows短暫禁止刪除分享鎖使progress原子replace報WinError5，blocker001失敗 | 有界重試≤.8秒，舊完整pointer保留、唯一temp；實際Win32鎖.15秒，8次嘗試.156秒成功；blocker002抵達。永久鎖仍明確失敗 | Windows-lock、atomic tests、兩版blocker |
| T10-15 P2 已修復 | 自行車actor銷毀後讀bounding_box回傳退化值，GT儀器出NaN，使bicycle001失敗 | cache有效靜態bbox，使用capture snapshot pose／velocity並標示無效幾何；bicycle002抵達。先保存已套用control紀錄再跑可選GT診斷 | observer source、兩版bicycle、integrity |
| T10-16 P1 已修復 | GT sensors建立一半拋錯，所有權尚未回傳可能漏清理 | finally回收部分setup所有權；真建立8個GT sensor後注入例外，零records、world actors0／sync還原 | setup-fault、world-cleanup |
| T10-17 P1 已修復／有延遲 | 關頁未必送到fetch取消；原12秒viewer lease實測13.51秒才出report | 加beacon＋fetch fallback，lease降3秒；真移動後無心跳4.391秒、真關頁4.402秒完成停止與清理。未單獨證明beacon即送；仍受當次推論／收尾延遲，且3秒背景節流可能保守停車 | 3s watchdog、real-browser-close、service tests |
| T10-18 P1 已強化 | 真相機停止、sensor validity故障、WebGL loss及實線越線 | 實際sensor.stop→15秒timeout停止；validity故障、WebGL停止均有獨立紀錄；實線event立即stop（未刻意實車越線刺激）。sensor loss時同步世界等待，非任意非同步世界安全保證 | camera-timeout、sensor-fault、browser-stops、regression |
| T10-19 P2 已修復 | seek／切run時RGB URL不固定或舊image仍顯示；晚到preview覆蓋新起終點 | RGB固定run＋frame，載入前隱藏舊圖；preview fingerprint拒絕舊結果，action錯誤持續可見、新job清空摘要／RGB標籤；最終67幀3D／RGB逐幀一致 | browser verifier、JS52 tests、HTTP27 checks |
| T10-20 P2 已修復 | 契約未嚴格擋非法road／yaw／forecast；一度過度嚴格拒絕合法held無yaw及forecast t0 | 修正有效例外，舊1123幀及新2194幀真Three全部通過；初次失敗log保留 | old-Three001–003、新Three-final、contract tests |
| T10-21 P2 未解 | 真離線播放仍有RAF長幀，512個active interval p95 41.735ms；同機CARLA時早期p95 39.29ms | CPU paint p95 1.5ms不能證明GPU／合成流暢；不能由這兩輪推論GPU競爭因果。需GPU／compositor trace與實際200% zoom；尚未做長時間洩漏soak | browser receipts、visual review、host samples |
| T10-22 P3 部分改善 | 原介面無旋轉操作、錯類只靠顏色難理解、直路圖等比軸壓成細條 | OrbitControls＋reset已實際drag驗證；顯示物件類別／暫失數。路線圖另出v2固定米制視窗，舊圖保留；手機375px無橫向溢出，但9–11px小字、即時標題重複與真200% zoom仍待改善 | delivery desktop/mobile、route-visual-v2、screenshots |
| T10-23 P2 已強化／限制保留 | 原source只存主要adapter、GT靜態物件未綁actor容易誤算FP、GT采樣稀疏 | 每run保存載入Python依賴閉包及root configs／runtime／weights；嚴格與ignore-aware雙報，16個僅ignored前景3D預測不作確定FP；資料manifest明示事後建立。未支援類別、斜車fullIoU、完整歷史／資產獨立性不可偽稱已評 | contracts、integrity、input manifest、validation |
| T10-24 P2 未驗收 | 同城少量受控交通、最高約3m/s、沒有密集多車／長距離／多輪重複、獨立新資產和長期記憶體測試 | 這些是明示覆蓋缺口，不是已觀測故障；未新增雨夜／Wall需求。先解T10-01至09，凍結新的獨立驗收矩陣再做後續訓練與壓測 | audit config、catalog重疊距離、完整測試矩陣 |

## 後續順序

1. 固定四個新路口，分離「正確車道燈頭關聯／phase latch」與「曲路道路證據」；先用可見性與同幾何oracle定位，不重複調整牆面或盲加訓練輪數。
2. 用新的距離×yaw×車型×遮擋樣本審查car/truck錯類及3D target定義。訓練前固定未參與調參的holdout；本輪393組GT已成為診斷資料，不能再當正式獨立驗收。
3. 用相同保存RGB、模型與輸出容差做運算／傳輸優化，每項報準確率及延遲／記憶體代價，再補完整端到端同條件回歸；禁止只報快的子段。
4. 補號誌方向、急停／橫穿的尾端預測與漏檢覆蓋，以及長時間browser／service資源測試。展示前優先使用明標錄製回放或已知具名路線。
