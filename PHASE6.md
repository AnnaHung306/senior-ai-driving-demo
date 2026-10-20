# Phase 5／6 聯合交付：WorldFrame 正確性與互動 3D

2026-10-20 最新使用者：「繼續驗證並確保phase5都有正確建立，你之前把phase5、6放一起，乾脆這輪一起做完吧」。本輪授權完成兩階段功能串接與驗證，取代先前「只停在Phase5首版」的工作界線；不把未過品質改寫為pass。既有晴天、無近牆、停止Wall／depth迭代與dev／lockbox不開的範圍維持。

## 本輪交付與事前驗收

1. **Phase5再核對**：原始來源／模型／資料hash，實際GT target變換，真CUDA validation reload與RGB／geometry干預（只診斷，不加訓），全部32份3D／occupancy重算、stream identity／timestamp／mask／單位與未知域拒絕。原baseline報告保留，模型品質沒有達標便明記；不調低IoU或忽略難例。
2. **嚴格顯示資料出口**：只從通過來源核對的WorldFrame與RGB-only Phase4道路線轉成顯示bundle。移除prior baseline和訓練／評估內容；保留物件位置／尺寸／yaw／ID、來源hash、當幀道路／線／界面、局部候選軌跡、occupancy、八RGB對照。道路／人行道界面不冒稱量測curb高度，候選軌跡不冒稱目的地導航；缺少的能力明示。
3. **本機互動3D介面**：Three.js固定版本vendor，只讀專用bundle與靜態資產；跟車／俯視／自由視角、縮放／旋轉／重設，真正的三維低面數示意mesh依預測LWH/yaw擺放，框／ID／物件檢視、道路線、路緣語意界面、路徑與可切換voxel。預製示意mesh只代表外觀，非重建形狀。
4. **序列與更新**：按原timestamp在scene內播放、暫停／逐幀／seek／速度／scene切換，保留漏幀／不可信提示；不在frame間偽造新感知。另接嚴格驗證的增量WorldFrame feed與斷線／stale／空輸出處理，驗證新結果到畫面同步。錄製回放與live feed來源明標，不宣稱CARLA閉環。
5. **誠實的介面**：unknown維持空缺，free／occupied不同；未校準分數／朝向、研究模型不足、資料模式、source age與資料頻率可見。GT只在獨立評估／檢查，不進主場景繪圖。原8RGB可對照，沒有GT世界模型／模型真座標渲染。
6. **完整測試與實看**：Python回歸、JS座標／yaw/corners與playback契約、HTTP資料隔離／hash／路徑拒絕、真瀏覽器操作與截圖、三視角／layers／scene／play／seek／selection／resize／錯誤重試／WebGL失效狀態。測更新率、render FPS／frame apply p50/p95與上游計算分開，不能把60FPS渲染稱60Hz感知。來源遭改／標籤混入／NaN／錯誤時間流不得畫成有效即時結果。

本機viewer功能驗收：所有預先固定UI流程通過、32份來源與空間映射一致、當前frame／ID／物件數與輸出一致、斷流明示、unknown不填free、全部回歸通過；渲染frame apply目標p95<200ms，桌面閒置視角render median>=30FPS（實機測不到或不達則不宣稱效能通過）。模型準確率沿用原32組及232 validation指標與失敗，不事後自訂新門檻讓模型pass。

## 架構與座標

本機read-only Python HTTP服務綁127.0.0.1，僅公開viewer資產與導出的資料根，不能遍覽workspace／labels／weights。無雲端註冊或發佈。可從CLI重新啟動，交付前停掉測試服務以免留下不明背景程序。

CARLA ego (x前,y右,z上) → Three.js (X=y,Y=z,Z=-x)，物件模型沿-Z前進、rotationY=-yaw。長寬高映射為(width,height,length)；八角點要與Python解析映射比較，不能鏡像左右。Orbit只是觀察者，不是車輛控制。Road／lane／occupancy顯示來自預測，預製ego車與觀察grid只作參考，未見處不畫成可行駛。

固定Three.js0.180.0；來源、npm integrity及實際vendor SHA／MIT license存檔。參照[官方OrbitControls](https://threejs.org/docs/pages/OrbitControls.html)、[WebGLRenderer](https://threejs.org/docs/pages/WebGLRenderer)及[InstancedMesh](https://threejs.org/docs/pages/InstancedMesh.html)。介面以按需／動畫loop管理資源與resize，切幀釋放舊mesh，避免播放記憶體累積。

## 本輪完成結果（2026-10-20）

工程研究版已交付；原規格的正式品質與模擬器即時驗收仍未通過。完整證據見 [聯合驗證報告](evidence/phase6/world-viewer-v1-001/README.md)。

| 項目 | 結果與限制 |
|---|---|
| Phase5 資料／訓練結果 | 891原RGB裁切逐byte一致、891組GT target獨立變換、232筆真CUDA reload exact；舊32份WorldFrame與3572來源核對沿用凍結證據；未更新權重 |
| RGB分支診斷 | zero RGB中心誤差.931→2.246m；同類RGB打亂中心仍.927m、IoU .231→.209。證明輸入依賴，未證明可靠視覺定位或泛化 |
| 主場景來源與座標 | 320檔hash、32影格／147物件／256RGB；Python原corners與真正Three.js BoxGeometry <1e-6；browser實際rendered transforms一致；GT geometry不進主場景 |
| 互動3D與序列 | 三視角、實際拖曳／滾輪／重設、框／ID／voxel／free／RGB、canvas射線與列表選取、四場景逐幀與.5×播放、1×／2×播放、seek／八RGB正常 |
| 更新與失效 | 已計算輸出pointer監看、兩次新key更新、GT欄位拒絕／保留最後畫面、恢復／stale、真WEBGL_lose_context及reload通過；80次正常主機原子發布成功 |
| 效能（本機IAB） | 70次CPU建立＋render提交p50 32.2／p95 41.1ms（不含GPU present）；161段約1秒render採樣median60FPS；geometry峰81、末42。通過事前200ms／30FPS顯示目標；非多小時壓力測試 |
| 測試 | Python235項0fail／0skip（15.329s），JS20項，真HTTP22checks；30筆瀏覽器操作收據、32份DOM診斷、桌面三視角／390px手機／WebGL故障截圖 |
| 仍未通過／未提供 | Phase5朝向、行人／free-space與長程ID品質；新無近牆完整路線；CARLA連續即時推論；新影格持續產生／跨run動態加入；路口拓樸、速度／運動預測、目的地導航及閉環 |

監看器只接受既有bundle內已驗證的推論key，不接受新key、其他run、任意GT或假時間戳。這是增量顯示接線測試，**不是已接上模擬器的live inference producer**。車道線與粗道路語意可顯示路面形狀，但沒有獨立路口拓樸辨識；路緣為語意界面，未重建高度。箭頭只代表局部循線候選；物件依不同frame量測移動，不外推虛構速度。

既有validation嚴格定位P61.11%／R55.93%、3D Recall@IoU.25為25.42%，是同資產／model-selection validation。感知stage含上游p50 1213.68ms，錄製序列5Hz只表示timestamp間隔。主畫面固定研究／未校準提示，unknown不填free；Town04岩石／植被被預測為卡車的錯例照樣顯示並保存。

## 啟動與重現

在專案根目錄執行 `./Start-WorldViewer.ps1`，再開啟 `http://127.0.0.1:8766/`；Ctrl+C停止。既有本機bundle002約88.6MB含256張原RGB，在ignored artifacts內。Git證據不包含原256RGB或權重；新機須先重建Phase4依賴與資料再export。Three.js0.180.0與MIT固定vendor，執行不連CDN，不啟動CARLA。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/export_world_viewer.py `
  --run artifacts/runs/phase5-world-v1-002 --integrity artifacts/reports/phase5-integrity-v1-002.json `
  --phase4 artifacts/runs/phase4-bev-pilot-002 --phase4-integrity artifacts/reports/phase4-artifact-integrity-001.json `
  --rgb-root artifacts/runs/phase4-pilot-selection-002/rgb_only --rgb-inventory artifacts/runs/phase4-pilot-selection-002/rgb_only_inventory.json `
  --evaluation-report artifacts/runs/phase5-world-v1-002-evaluation/report.json --output artifacts/runs/my-new-viewer
# 新輸出不覆寫；用自選目錄時以--bundle指定。
.\.venv\Scripts\python.exe -X utf8 scripts/serve_world_viewer.py --bundle artifacts/runs/my-new-viewer --feed artifacts/runs/phase6-feed
# 另一終端發布既有推論pointer；heartbeat不是重新推論。
.\.venv\Scripts\python.exe -X utf8 scripts/publish_viewer_frame.py --bundle artifacts/runs/my-new-viewer --key town04-00/246278 --heartbeat-seconds 10
```

同scene監看必須依timestamp向前；重新啟動服務才重設監看序列狀態。一般時間軸回放不受限制。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/check_world_viewer.py --output artifacts/reports/my-viewer-integrity.json
node scripts/check_world_viewer.mjs artifacts/runs/phase6-viewer-bundle-002 artifacts/reports/my-viewer-integrity.json
node --test web/world-viewer/tests/*.test.mjs
.\.venv\Scripts\python.exe -X utf8 scripts/check_viewer_http.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Node需現代Node.js（本輪使用Codex bundled Node）。受限環境首次測試在系統TEMP清理被拒，改設**測試子程序專用**TEMP/TMP到artifacts/tmp，235項通過，沒有改系統環境或skip。loopback服務／HTTP與publisher在正常主機環境驗證，未調整防火牆。所有測試服務與有界publisher在交付前停止。

## 完成界線

本輪要完成Phase5可驗證的World Model功能與Phase6本機互動介面／串流接線，並交付正確的功能／品質矩陣。若準確率、長程追蹤、即時感知或fine geometry未過，保留未完成狀態，不以可轉動的方塊把它們算完成。新完整無近牆路線、完整速度／預測／目的地路由／車輛控制屬仍待實測或後續Phase7–9。
