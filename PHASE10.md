# Phase 10：固定回歸、準確率、效能與展示稽核

2026-10-21依使用者要求執行。目標是系統性找出可重現問題，不以有限測試宣稱已找出所有可能錯誤。沿用晴天、無近牆範圍；不補Wall、不採depth、不盲訓、不開既有development／lockbox。既有Phase3–9證據不覆寫。

## 執行前固定的範圍

1. 凍結當前四權重、地圖、程式與門檻，建立新版本輸出。先用地圖、固定選擇順序選六條不同起點的新路線（兩直路、兩左轉、兩右轉）；不依模型成功挑選。與Phase89測試路線及007 train自車軌跡距離逐案公開，同城／遠景／歷史資料重疊不稱heldout。
2. 六案實際八RGB閉環；沿用Phase89距終點≤2.5m／末速≤.3m/s／碰撞0／實線越線0／橫向誤差≤1.5m／90模擬秒。完整分母與未成功原因必須列出，不為通過改規劃門檻。長路線／轉彎、道路支持及停走是重點。
3. 獨立評估感測器只寫診斷目錄，不回傳道路／物件／號誌真值給planner。八路instance raw同幀提供可見語意與actorID；可見框、3D框與速度真值和RGB模型比較。每五個推論影格保存一次GT（含首幀），時間戳嚴格匹配。沒有支持的類別標無法評估；不把未標註靜態交通號誌算真負例。
4. 報告語意逐類IoU／support／混淆；偵測可見框AP50／AP50:95／P/R／尺寸分層；未知GT的偵測分開列，嚴格／忽略政策都公開；3D中心／尺寸／yaw／IoU與漏檢、追蹤ID／速度／CV預測和可評未來覆蓋。相同新影格比較learned3D與幾何尺寸先驗，不能把條件框誤差當端到端Recall。
5. 額外固定障礙車、橫穿行人與紅→綠的受控交通診斷，檢查近距離漏檢／誤判及停止距離；director與GT不參與主控制。既有有效性／真相機timeout／人工stop／WebGL故障回歸。
6. 效能分離啟動、tick/RGB等待、PNG寫入＋hash、GPU感知、規劃控制、GT評估額外成本、JSON寫入／publish與完整循環；p50/p95/max、冷啟動、GPU記憶體、實際牆鐘／模擬倍率。同步5Hz的即時預算為200ms，超過如實fail，不用render FPS代替。GT感測器使實驗變慢，另有不掛GT的基準回合供比較。
7. 網頁真browser核對桌機／390px、鍵盤／200%字體縮放、路線預覽／失敗、回放seek／完整播放、八鏡頭、3D來源／控制量、場景資訊與錯誤狀態；量測非idle更新、記憶體資源及服務傳輸。檢查舊回覆覆蓋、載入失敗與斷線行為。畫面圖表必須實際檢視。
8. 工程補充：NaN／不合法shape／舊時間戳、並行tick、取消／關頁、來源篡改、路徑／Host／Origin、完整性與依賴／可攜性、檔案／記憶體成長、停車後actors與world恢復。回歸測試與保存來源核對獨立於品質。

不預設新樣本的模型品質會通過；保留原品質門檻與未達項目。新增問題使用ID、嚴重度、證據、觸發、影響、處理／未解原因及回歸對照。可直接修正可重現工程錯誤；涉及模型再訓練只在有明確對照設計時另列，不用重複調整同測試集追高分。

## 交付

完整測試矩陣、所有新路線結果／失敗、逐類準確率與分層、效能分解／瓶頸、視覺圖文與真browser收據、同條件消融、問題清單、部署限制與下輪建議；封存source／權重SHA／資料manifest／原log，最後停止本輪服務並保存本機Git版本。此報告下方會在實測完成後追加權威結果，不覆寫先驗計畫。

## 最終結果（2026-10-21）

**Phase10系統稽核與工程修復完成；正式駕駛／模型品質未通過。** 六條預先固定的新路線只有2/6抵達。所有24次attempt（含故障刺激、取消與儀器失敗）保留，23次有影格，共2194幀／17552張RGB。獨立GT每五幀取樣共393組八視角、3144張影像；另保存6288張GT與預測語意PNG。四權重未變、沒有新訓練、套件安裝或舊development／lockbox評分。

權威評估是 `artifacts/runs/phase10-evaluation-final-v1-002`；早期001／六案／preliminary均保留。封存入口：[證據README](evidence/phase10/system-audit-v1-001/README.md)，具體缺陷與處置見[24項問題登錄](PHASE10_ISSUES.md)。

### 固定六案與受控回歸

| 原始路線 | 長度m | 幀數 | 結果 | 末端距離m | 最大橫向誤差m |
|---|---:|---:|---|---:|---:|
| left-1 | 50.56 | 246 | blocked_safe_stop | 5.582 | 0.797 |
| left-2 | 88.86 | 289 | blocked_safe_stop | 32.496 | 0.686 |
| right-1 | 66.12 | 450 | simulation_timeout | 24.995 | 0.276 |
| right-2 | 39.61 | 176 | blocked_safe_stop | 7.320 | 0.916 |
| straight-1 | 54.00 | 94 | arrived | 1.896 | 0.303 |
| straight-2 | 54.00 | 94 | arrived | 1.654 | 0.249 |

六案碰撞／實線事件皆0，但只代表低速且有限場景觀測；保守停止不能算抵達。來源、controller重算與清理均驗證，不能取代完成率。路線catalog只依地圖固定選取，同城／視野／資產及訓練歷史不獨立，逐案重疊距離公開，沒有heldout或泛化通過宣稱。

工程修復後另外五個具名受控案例抵達：障礙車移除恢復76幀、行人橫穿67、自行車65、前車煞停／恢復111、15秒紅燈→綠燈轉彎236。它們不是五個新的獨立成功率樣本。blocker001因Windows鎖失敗、自行車001因GT bbox生命週期失敗均保留；setup-fault零影格也是一次attempt。七次web job含harness取消、WebGL、關頁與watchdog刺激，不混為正常導航完成率。

right2 F30328的道路支持RGB soft／hard與GT同幾何oracle一致；末切線前向比.6297未達.7，沒有套用終點延伸，因此「延伸超過終點造成失敗」在此案例被排除。尚待檢查曲路投影、視野與地圖／取樣對齊，沒有修改.6證據門檻或重跑改寫2/6。號誌關聯存在大量未能解析樣本與缺少對應停止點的停等；四次停止點穿越代理無穩定紅燈，不足以認證遵規。

### 語意、偵測、號誌與三維準確率

GT由獨立instance cameras及capture snapshot產生，只進audit，不輸入主planner；主路面／物件仍來自八RGB模型。raw IDs、忽略未知／動態策略、可見tight框≥16px且邊長≥4px、score .5與IoU協議均保存。這套可見框協議不等於舊完整validation，不直接宣稱模型退步或進步；temporal重複影格高度相關，沒有獨立樣本置信區間。

| 語意類別 | GT像素 | IoU |
|---|---:|---:|
| other_annotated | 365340983 | 95.51% |
| road | 148026130 | 98.06% |
| sidewalk | 118110495 | 94.78% |
| pedestrian | 47137 | 49.70% |
| rider | 10697 | 31.09% |
| car | 204453 | 72.85% |
| truck | 7306 | 8.58% |
| bus | 0 | 0.00% |
| motorcycle | 0 | 0.00% |
| bicycle | 1699 | 34.75% |
| road_line | 608978 | 86.63% |
| traffic_light | 933671 | 11.91% |
| fence | 13492719 | 57.28% |
| pole | 5438773 | 66.30% |
| temporary_obstacle | 0 | 0.00% |
| terrain_or_ground | 49382388 | 87.62% |

Wall依使用者範圍不作改善目標；無GT支持不能算完美。道路98.06%是2D像素統計，不能當作BEV、深度或閉環品質。

| 可見框類別 | GT框 | AP50 | AP50:95 | P@.5 | R@.5 |
|---|---:|---:|---:|---:|---:|
| car | 77 | 17.13% | 11.94% | 20.00% | 5.19% |
| pedestrian | 23 | 85.61% | 62.38% | 15.44% | 91.30% |
| bicycle | 10 | 44.53% | 29.28% | 22.22% | 100.00% |
| motorcycle | 0 | N/A | N/A | 0.00% | N/A |
| truck | 0 | N/A | N/A | 0.00% | N/A |
| bus | 0 | N/A | N/A | 0.00% | N/A |
| traffic_light | 1318 | 18.45% | 10.62% | 91.20% | 17.30% |
| temporary_obstacle | 0 | N/A | N/A | 0.00% | N/A |

同時保存strict-no-ignore與ignore-aware分數；可見但不在actor snapshot的靜態車輛不可直接判為確定假陽性。car在最高分不分類IoU50配對中44/77被叫truck，與class-aware AP全域greedy的4個TP是不同診斷協議。car小框Recall0/38，中框4/39；可見性、截斷與各類分數可靠度詳JSON。行人Recall高但Precision15.44%，表示誤報也很多。

號誌head另做1084個phase前後夾定且相位不變的可見crop CPU重播，234個無法夾定者排除；保存exact head_inputs.npz，不稱CUDA bit-exact。GT crop macro Recall85.67%，detector＋state joint macro42.37%；joint red91.45%、yellow37.50%、green40.40%、unknown .12%。GT unknown被head判green173個，需釐清燈頭可觀測性與方向，不能直接算闖紅燈。

| 三維／時序指標 | 實測 | 解讀 |
|---|---|---|
| 可見動態GT／預測 | 51／239 | 視角合併後actor級；靜態未綁定另列 |
| strict中心TP／FP／FN | 26／213／25 | P10.88%、R50.98%；不分類中心TP50 |
| ignore-aware中心TP／FP／FN | 26／197／25 | 16個僅ignored前景來源保持未評，P11.66% |
| full3D IoU≥.25 | 2/51，Recall3.92% | 不可用中心接近取代尺寸／朝向品質 |
| 條件中心MAE／3D IoU均值／yaw MAE | .531m／.1499／51.27° | 僅成功匹配26個，不含漏檢 |
| 同33個2D TP：learned／prior中心MAE | .6406／.6361m | 無明顯改善 |
| 同33個2D TP：learned／prior IoU | .10445／.24170 | learned尺寸與朝向表現較弱；未依此自動換模型 |
| 速度向量MAE | .4835m/s（21匹配） | 動態12個.6562；靜態9個.2532 |
| 抽樣連續ID switch／間隙重識別變ID | 1／3 | 不是完整MOT／HOTA統計 |
| 1秒ADE／FDE／GT覆蓋 | .6793／.8995m／18/46 | 相同匹配靜止基準FDE .8981m |
| 2秒ADE／FDE／GT覆蓋 | .9139／1.4536m／14/41=34.15% | 靜止基準FDE1.5066m，均值差小 |
| 2秒FDE最大值 | CV4.8298m／靜止2.3563m | 平均略好仍有更差尾端 |

未來消失、右設限、非upright與無支持類別分開標示。沒有獨立metric BEV深度／全occupancy GT；不能以語意oracle同投影推導幾何準確率。

### 效能、畫面與資源

| 不掛GT的257個循環 | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|
| ticks_ms | 6.481 | 10.125 | 20.630 |
| RGB_receive_decode_ms | 16.204 | 20.718 | 86.218 |
| RGB_PNG_and_SHA_ms | 199.640 | 232.069 | 304.607 |
| inference_ms | 937.079 | 1024.653 | 1628.103 |
| planning_control_ms | 0.774 | 1.775 | 2.960 |
| cycle_to_control_ms | 1159.919 | 1268.664 | 1902.852 |
| evaluation_observer_ms | 0.000 | 0.001 | 0.001 |
| JSON_and_publish_ms | 30.726 | 43.403 | 158.740 |
| total_cycle_ms | 1193.935 | 1303.592 | 1932.190 |

總計2193個timing，而保存record2194個：舊儀器失敗時有一個已記錄control缺timing，修復後finally記錄；分母不補造。無GT仍約1.2秒／輪，200ms即時門檻未通過。原straight1分段中位數：input/seg137.09、八路det/light403.49、BEV/temporal/retainedlight641.26、cuboid/motion1.42ms；不能把這些中位數加成總p50。startup逐run保留，冷啟動／fault樣本不隱藏。

兩項有界同輸入優化：靜態投影cache在3組真八RGB觀測中所有陣列與maps exact，fusion p50 516.02→78.63ms（6.56倍僅該段）；12張實圖×3 paired PNG level6→1，encode/write/hash p50 44.54→24.43ms，減45.15%，bytes×1.555、解碼exact，通過事前≥20%時間／≤2倍大小篩選。採用的optimization config綁benchmark hash。相同54m route原GT版total p50 1623.93ms、優化noGT版1187.70ms；GT／交通負載不同，不能將全部差異當cache因果。

最終真瀏覽器完整67幀，83物件、64條實際Float32預測buffer、道路／路線／控制／八RGB逐幀來源相符；後端全部2194幀／1338個cuboid使用實際Three驗證。CPU paint 135次p50 .7／p95 1.5ms；512個active RAF interval p50約16.6／p95 41.735ms，仍有長幀。早期CARLA同開p95 39.29ms，無法從此證明GPU競爭因果。render約60FPS、5Hz模擬時間回放與秒級模型更新必須分開。

實際桌機與375px手機頁已檢視，八鏡頭、seek／完整播放、3D拖曳旋轉及reset、路線／錯誤流程有收據。手機無橫向溢出但字偏小；真正200% browser zoom未成功驗證，640px窄窗不能代替。即時標頭重複文案尚留。首版路線圖因等比窄軸壓成細條，另出v2統一米制範圍並實看六頁contact；原圖不覆寫。

本輪RGB實際儲存6,000,625,990bytes、GT＋pred語意PNG 58,992,917bytes。這是保留證據成本，未做無限run輪替／長時間磁碟壓測。30次粗粒度共用GPU抽樣不能當每模型記憶體；67幀renderer geometry min/max/final=1/12/1，texture max=0，短案無單調增長不代表已通過長期洩漏測試。

真loopback warm傳輸：state63,971bytes p50 5.047／p95 28.869ms；frontPNG416,021bytes p50 30.911／p95 34.106ms。27項真HTTP包含來源、錯誤／路徑／Host／Origin與純回放限制，通過不代表公開網路安全或延遲。

### 修復、故障與回歸

修復範圍詳問題T10-11至23：finite/time gate、曲路障礙幾何、紅燈stall、Windows有界atomic重試、GT actor bbox／部分setup清理、control-before-GT紀錄與timing finally、實線立即停止、RGB run/frame一致性、preview競態／錯誤可見性、Three契約、OrbitControls及來源依賴閉包。永久檔案鎖仍要明確失敗，不無限重試。舊core／bev_fusion／權重與Phase0–9證據未改。早期六案及setup-fault的contract只列三權重，cuboid SHA由當時凍結的live_perception字面常數綁定；後續17案contract明列四權重及載入依賴閉包，早期來源範圍較窄的限制保留，沒有事後補造當時快照。

真相機sensor.stop後15秒timeout、有效性故障、WebGL失效、setup建立8GT sensor後例外都分別驗證停止／清理。實線stop已加防護與回歸，未故意實車越線。關頁fetch不可靠，原12秒lease實測13.51秒才出report；加入beacon fallback並改3秒lease後，真移動watchdog4.391秒、真關頁4.402秒完成停止且cleanup速度0。未單獨觀察到beacon即送；仍是推論週期加收尾的有界延遲，背景節流也可能觸發保守停止。

完整Python322項、0fail／0skip（27.726s），最終3秒lease修改後重跑service與Phase10 service小組全過；JS52項、0fail；27真HTTP；原1123幀Three回歸及新2194幀全部過。Windows沙箱loopback10013與正常host測試分開，初次harness／過嚴契約／錯誤測試期待的失敗log保留，沒有把工具問題當模型成敗。每run以自身凍結source重算planner/controller exact，所有RGB／GT hash核對；沒有借用最新程式重算假稱舊run一致。

### 展示、封存與恢復

本機閉環仍用 `./Start-NavigationDemo.ps1 -StartSimulator`，`http://127.0.0.1:8768/`自訂Town01道路起終點／預覽／啟動；任意有效路線不保證抵達。`-ReplayOnly`不載GPU或CARLA。詳[DEPLOYMENT.md](DEPLOYMENT.md)。

新增16:9實際瀏覽器錄製回放 `phase10-focus-replay.webm`，1280×720、67幀、5FPS、13.4秒、2,483,330bytes，實際browser解碼播放，明標recorded/not live與地圖先驗。影片有損編碼不作準確率輸入；保存原PNG／JSON與SHA。可攜ZIP `phase10-portable-replay-v1-001.zip` 139,595,970bytes，609項manifest＋自身manifest共610檔；含Phase89真行人64幀／512RGB，8770新資料夾實測、409拒絕起車且未import Torch/CARLA。新版加入已實測Orbit旋轉；舊Phase89 ZIP不可旋轉的原文宣稱在部署文件更正，舊ZIP不改。

source-snapshot、所有新run與失敗／GT labels／pred masks／exact light inputs／browser收據／原log／錯例圖／影片及post-run input manifest封存於evidence。完整17552RGB、四權重與133.1MiB可攜ZIP保留本機ignored artifacts，不上傳／不公開。每ZIP CRC及逐項SHA核對；manifest證明bytes，不證明模型品質或完整真值。

交付時CARLA world為async，vehicles／walkers／sensors均0，之後停止server；本輪8768／8769／8770／8771與2000無listener、沒有Phase10 worker，瀏覽器測試tabs關閉。下一輪從問題清單的新獨立矩陣開始，不自動重開Phase3、Wall或depth。

### 完整性與未測項目

| 領域 | 已做 | 明示缺口 |
|---|---|---|
| fixed_new_routes | 2/6 arrived; all six preserved | same-town assets, no heldout claim |
| 2D_accuracy | 3144 visible GT/pred mask images; AP50:95/class/size/confusion/score analysis | no GT for bus/motorcycle/temporary obstacle; visible-mask protocol differs old validation |
| 3D_and_forecast | 51 dynamic GT; strict and ignore-aware; learned/prior paired33; CV/persistence paired14 at2s | sparse tracking, low future coverage, no full MOT/occupancy-depth GT |
| signal_state | 1084 phase-bracketed samples, exact CPU head inputs; ROI association diagnostic | 234 excluded; association and actual violation certification unresolved |
| performance | 2193 timed cycles; 257 noGT; paired cache and PNG; loopback timing | one old applied-control record missing timing before instrumentation repair; no public-network or per-kernel GPU trace |
| UI | 67 final frames,83 objects,64 Float32 forecasts; eight RGB; desktop/mobile; actual drag/play/seek | real200% zoom not verified; small text and RAF long frames remain |
| faults_lifecycle | real sensor.stop, validity, setup exception, WebGL loss, close/watchdog, lock contention | no intentional live solid-line crossing; finite tested cases not all failures |
| resources_storage | 30 shared-device samples; per-frame renderer resources and actual artifact sizes | no sustained soak or per-model memory isolation |
| delivery | CPU replay ZIP609 manifestfiles/512RGB; focused real13.4s WebM | no external/public deployment; raw RGB and weights local ignored |
| regression_provenance | 322 Python,52 JS,27 HTTP,2194 actual Three frames; frozen source/weights checks | engineering correctness is not formal model quality |

不能用有限測試保證找出所有問題。後續優先順序是號誌關聯／曲路道路證據、car/truck與3D target、端到端效能、尾端預測／覆蓋及長期穩定性；任何新訓練先固定獨立資料和門檻。
