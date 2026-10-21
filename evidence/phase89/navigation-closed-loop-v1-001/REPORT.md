# Phase8–9：本機導航、RGB閉環與網頁操作（2026-10-21）

本輪依使用者要求一次實作並驗證Phase8–9，提供自定義起終點、global/local/behavior planning、轉向與速度控制、即時本機網頁及部署說明。晴天、無近牆理想環境，沿用凍結權重，不重啟Wall/depth或開發／lockbox資料。規則規劃是原規格允許的第一版，不假稱新訓練端到端駕駛模型。

## 實測前固定的邊界與門檻

- global routing使用CARLA OpenDRIVE地圖及自車pose／speed（模擬定位／里程計）；這是明示導航先驗，不是RGB估計。其他actor的真值、號誌狀態不得進RGB感知或規劃器；碰撞／車道事件與新路線評分在獨立評估欄。
- 同步單一tick owner、固定dt0.1s、每輪8張640x360RGB→凍結seg/det/light/cuboid→BEV／追蹤／Phase7→local/behavior→自寫pure-pursuit／速度控制→CARLA新影像。GPU readback需一tick，明列影像年齡，不把牆鐘慢於模擬當即時。
- 起終點由地圖點選或spawn index指定，須先預覽可連通路線；不接受NaN／越界／空路線／過短過長／不連續點。自車不用BasicAgent或autopilot，後端不使用其他actor位置偷渡控制。
- Local以地圖走廊為參考、RGB路面支持及預測物件作可行性檢查；有限同車道橫向候選，未知／無可行路徑停止。無已知安全鄰道不強行換道。紅／黃／未知可疑正前號誌保守停，穩定綠後恢復；只有模型辨識到的號誌能影響決策，漏檢風險不得隱藏。
- 規劃／推論異常、影像不同步／超齡、越路線限制、操作停止時全煞車；worker取消後清理本次actors並恢復world settings。瀏覽器關閉不自行啟動任務。loopback API、Origin及JSON驗證、最多一個駕駛worker，重複啟動拒絕。
- 固定新路線catalog先於駕駛結果保存SHA，至少兩條直路及一條有轉向的路線；先做空交通低速15km/h以下工程驗證。成功必須距終點≤2.5m、speed≤0.3m/s、碰撞0、非允許越線0、最大路線橫向誤差≤1.5m、模擬時間≤90s。失敗原樣保留；不將舊訓練道路／挑選成功當未見泛化。
- 純數值診斷固定紅→綠、前車減速、橫穿行人、阻塞、可用側移／無側移、急彎、失聯／NaN／倒序、終點煞停；這些不能替代CARLA真RGB交通驗收。實機新路線須另核對與原訓練軌跡距離後才稱未見路線。
- 網頁必須能選點／預覽／啟動／停止，顯示即時車速、決策、進度、RGB及AI物件／路面來源；可載入已完成run回放，不將回放標live。真browser驗證提交、重複啟動、停止、完成與錯誤。最後統一Python／JS／HTTP／來源及視覺驗證。

## 部署設計

CARLA Windows DX12 Epic與PyTorch CUDA推論在本機RTX4070；Python控制服務只綁127.0.0.1，提供JSON／RGB／run；瀏覽器HTML/Three.js只畫圖及提交導航命令，不執行Torch、不直接控制CARLA。由此可在同一台筆電投影展示。跨裝置／公開網路需要另建認證、反向代理／TLS與GPU主機，這輪不公開控制端口。

技術依據：CARLA本機0.9.16 `GlobalRoutePlanner`地圖路由與官方[synchronous mode](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)；自車規劃／控制為本專案實作，不把TrafficManager規則當學習AI。

## 交付結果

**Phase8–9本機工程展示版完成；通用自駕、未見資產泛化及正式模型品質仍未通過。** 本輪已讓新八鏡頭影像實際進入凍結模型，再由規劃／控制輸出讓CARLA車輛移動，取得下一組影像，形成同步閉環。不是用BasicAgent／autopilot代駕，也不是把錄製畫面當即時駕駛。沒有新訓練、套件安裝或開development／lockbox。

| 工作項 | 交付與實測 |
|---|---|
| Global routing | 本機OpenDRIVE map／自車定位先驗；Town01地圖點選吸附到255個道路spawn節點，下拉亦可自訂起終點；預覽8–600m連通路線。明確拒絕未驗證換道 |
| Local planning | 導航走廊、RGB路面支持、當前與≤0.65s暫失物件、0/±.45m同車道候選、時間間距與CV接近檢查；無可行或未知則停車。非任意超車／變換車道 |
| Behavior | 循線、跟車／減速、同車道避讓、號誌停等／穩定綠燈起步、障礙停車／恢復、終點煞停、失效停止；停止原因在網頁可見 |
| Control | 自寫pure-pursuit轉向＋PI速度控制／全煞車，單一OS tick lease、同步dt.1s、八鏡頭一tick readback；每次控制循環2 ticks。模型推論較慢，非牆鐘即時 |
| 真交通探測 | 獨立director擺放障礙車／橫穿行人、受控紅→綠；環境真值沒有輸入自車planner。停讓、恢復與抵達均有真RGB／控制／碰撞及越線事件 |
| 網頁控制 | 已從網頁自訂179→62，11.0m實際導航，4.6模擬秒／42.375牆鐘秒，23影格，抵達距離1.595m、速度.000446m/s；不是預設route按鈕代替自訂 |
| 停止與故障 | 人工stop及真WEBGL_lose_context均取消job、清理前實測速度0；有效性故障注入及真正停止front相機的15s timeout均煞停；真timeout收尾速度0，原TimeoutError保留 |
| 純回放 | `-ReplayOnly`在CARLA關閉時可播放與seek，禁止起車；服務匯入不載入Torch／CARLA。資料仍需帶完整record／PNG，不把Git當完整資料集 |
| 驗證 | 295 Python、48 JS、23項真HTTP；15個有逐幀結果的run，共1123影格的來源／控制重算；實際瀏覽器411影格／333物件／169組Float32軌跡核對；八鏡頭、390px、完整播放、source／path拒絕與故障恢復 |

### 最終具名版本（5案／3條地理路線）

| 案例 | 長度 | 模擬／牆鐘秒 | 最大橫向路線誤差 | 結果 |
|---|---|---|---|---|
| v002分離軌跡直路1 | 28.0m | 10.8 / 92.89 | .208m | 抵達停穩 |
| v002分離軌跡直路2 | 28.0m | 10.2 / 88.02 | .050m | 抵達停穩 |
| 受控紅→綠＋轉彎 | 50.788m | 25.2 / 212.39 | .706m | 先停等、RGB確認綠燈後起步並抵達 |
| 障礙車v002 | 28.0m | 15.0 / 129.05 | .080m | 停車，移除障礙後恢復並抵達 |
| 橫穿行人 | 28.0m | 12.8 / 112.00 | .130m | 減速／同車道避讓後抵達 |

五個**最終具名版本**皆符合原距離≤2.5m、速度≤.3m/s、碰撞0、越線0、橫向誤差≤1.5m及90模擬秒門檻；最高速度3.2205m/s（11.59km/h），低於15km/h。行人案最後記錄速度.185m/s，仍在門檻內；收尾再全煞車。這不是全部開發嘗試100%成功的統計宣稱：**18個嘗試含預檢失敗、中斷及舊版失敗都保留**，見`phase89-validation-final.json`。同案例反覆開發不能當獨立測試，正式品質false。

原catalog001與007訓練軌跡重疊，因此另在**駕駛前**建立catalog002，要求全路線頂點距007 Town01 train自車軌跡至少15m；三案實際最近距離139.03／86.49／133.76m。只證明此幾何軌跡分離，不排除歷史資料、相機遠景或同城鎮資產重疊；不稱heldout泛化。原catalog001／選例失敗與source恢復收據保留，原結果不覆寫。

### 必須保留的失敗與修正

1. 第一輪把終點以外的紅燈當停止條件；加入靜態地圖「前方18m路口、終點以前」相關性，顏色仍讀RGB。
2. 短路線接近終點只剩hood盲區的採樣；用末段切線延續做**道路背景支持檢查**，沒有延長控制目的地，也沒降低道路證據門檻。
3. 原轉彎001／002曾因綠燈偵測框消失持續停等；後加入由原偵測ROI起始的RGB光流追蹤，只在RGB SE2確認靜止時，以**當前crop**重新判別號誌。30秒期限、.70狀態分數及3幀綠燈限制明示，沒有真值解鎖。
4. 障礙車001無碰撞停下但未恢復，不能算完整pass。因阻塞使道路不可見而丟失正確停止原因，v002同樣全煞車但保留前方預測物件的阻塞分類，並增加12秒無進度終止；原失敗不刪。
5. 最初強制中斷留下同步設定／actors快取問題，預檢拒絕重啟；確認舊worker退出後精確清理本次role並恢復。後續使用取消旗標及OS lease，實際確認world非同步、vehicles/walkers/sensors皆0才停止CARLA。
6. 最後QA曾因焦點在range上，Ctrl+Home將seek移回0；改用解除焦點後截圖。另加伺服器revision／前端POST排序，防範潛在較舊回覆覆蓋新seek；不把QA快捷鍵誤用當模型錯誤。

### 效能與限制

五案370影格的感知／融合／運動計算p50 **1216.02ms**、p95 **1338.98ms**，不含PNG存檔、RPC、所有控制與畫面顯示。同步時間戳5Hz不等於實際5Hz推論；牆鐘約慢8–9倍。browser抽樣render約60FPS（含idle），也不等於感知頻率。

模型仍會錯類、尺寸錯估及誤停：障礙Audi曾被辨識成卡車／機車，路邊背景也有假陽性；速度曲線可見短促停走。跟車舒適性、長時穩定、密集多方向交通、任意節點組合、未見城鎮及無近牆完整路線的廣泛驗收未通過。沒有學習式意圖／planner、規劃安全保證或任意超車；不把本輪路線誤差當RGB定位準確度。原Phase3／5／7品質缺口均保留。

## 啟動與可重現命令

操作及架構見[DEPLOYMENT.md](DEPLOYMENT.md)。

```powershell
# 真閉環：啟動／重用專案CARLA、檢查Town01，再開本機控制頁。
.\Start-NavigationDemo.ps1 -StartSimulator
# 純展示：不啟動CARLA／CUDA模型；錄製資料必須在本機。
.\Start-NavigationDemo.ps1 -ReplayOnly
# 兩種模式網址皆為 http://127.0.0.1:8768/
```

```powershell
# 重播請用新輸出目錄，不覆寫已封存run。
.\.venv\Scripts\python.exe -X utf8 scripts/run_phase89_drive.py --route straight-1 --output artifacts/runs/my-new-drive
.\.venv\Scripts\python.exe -X utf8 scripts/probe_phase89_traffic.py --scenario signal --output artifacts/runs/my-new-signal
# 獨立核對既有已保存來源、控制與實際Three幾何。
.\.venv\Scripts\python.exe -X utf8 scripts/verify_phase89_runs.py
node scripts/check_phase89_client.mjs
.\.venv\Scripts\python.exe -X utf8 scripts/check_phase89_browser_receipt.py
```

瀏覽器數字傳輸有IEEE尾差，核對容許絕對1e-12（實測最大7.11e-15）；字串、IDs、hash、布林與結構仍嚴格一致，實際Float32軌跡允許1e-5m。沒有改感知輸出／品質門檻。GT評估圖7張及兩contact頁實看，曲線使用**地圖與自車定位先驗**，非RGB定位GT分數。所有早期失敗、source版本、原始log及相關限制封存於[evidence/phase89/navigation-closed-loop-v1-001](evidence/phase89/navigation-closed-loop-v1-001/README.md)。

另已產出`artifacts/runs/phase89-portable-replay-v1-001.zip`（約133.1MiB），含真行人案64影格／512RGB，已在獨立資料夾實測回放、八RGB雜湊、607檔manifest及禁止起車；匯入不載入Torch或CARLA。用於沒有GPU的離線展示，操作見DEPLOYMENT.md。ZIP與完整RGB留本機ignored；Git保留其manifest、SHA及驗證收據。收尾已確認CARLA與導航服務退出、相關連接埠無listener，沒有後台駕駛／訓練工作。
