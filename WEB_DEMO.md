# 網頁展示與GitHub Pages部署

本文件適用於 `docs/` 的獨立靜態展示。完整模型／CARLA閉環的部署另見 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 直接展示

- 公開網址：<https://annahung306.github.io/senior-ai-driving-demo/>
- 原始碼：<https://github.com/AnnaHung306/senior-ai-driving-demo>
- 預設第一章：`phase7-viewer-bundle-v1-001` 的完整 `town01-00` 60幀，約12秒；左前鏡頭同時可見車、機車、行人。這是既有訓練場景感知重播，沒有駕駛控制。
- 第二章：`phase10-drive-long-red-probe-v1-001`，236幀／47.2模擬秒、50.8m；紅燈停等、RGB綠燈辨識、左轉抵達的完整真閉環紀錄。
- 第三章保留原 `phase10-drive-pedestrian-probe-v1-001`，28m、67幀／13.4秒。
- 按「探索駕駛過程」，選章節，再按播放或重點書籤。可切八鏡頭、拖時間軸、按左右單幀、選0.5／1／2倍，拖曳或滾輪調整3D；「重設視角」恢復預設。
- 下方有59.2秒16:9兩章展示影片（完整60＋236幀、5fps、模擬時間1×、無聲），支援下載。分享公開網址即可，不需要分享此電腦或啟動模擬器。

這是保存結果回放。模型沒有放在GitHub Pages上執行，也不接受線上駕駛控制或新的起終點；新路線請在有模型權重與CARLA的本機操作。固定成功路線符合目前展示目標，不代表完整自駕品質已通過。

## 從乾淨clone開啟（不需要任何Python套件）

先安裝Git與Python 3.10以上，然後：

```bash
git clone https://github.com/AnnaHung306/senior-ai-driving-demo.git
cd senior-ai-driving-demo
python -m http.server 8080 --directory docs
```

開啟 <http://localhost:8080/>；Ctrl+C停止。Windows也可以用 `py -3 -m http.server 8080 --directory docs`。如果8080被占用，改8081並使用對應網址。不要直接雙擊index.html：瀏覽器會限制本機file模式的模組／fetch，SHA驗證也需要HTTPS或localhost。

`docs/`已包含所需資料與授權vendor，不需要npm install、CUDA、Torch、CARLA、原始artifacts或模型權重。也可單獨複製整個docs資料夾，進該資料夾後執行 `python -m http.server 8080`。

## GitHub Pages設定

本專案採用公開repository＋**Deploy from a branch／main／docs**。只有docs內容成為展示網站，整個研究程式與歷史證據仍在repository供查閱。

1. 開啟repository → Settings → Pages。
2. Build and deployment的Source選 **Deploy from a branch**。
3. Branch選 **main**，Folder選 **/docs**，按Save。
4. 到Actions等內建「pages build and deployment」完成，再開上方公開網址。GitHub官方說明指出更新可能需要數分鐘。

`.nojekyll`使內容以靜態資產提供；全部URL使用相對路徑，適用 `/senior-ai-driving-demo/` 子路徑，不假設網站位於domain根目錄。GitHub Pages不執行Python後端；它只需提供HTML／CSS／JS／JSON／JPEG／WebM。

官方參考：[建立Pages網站](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)、[Pages限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)。如改repository名稱，請同步更新首頁／README連結；程式資產的相對路徑仍可用。GitHub帳號既有的首頁與lane-nav專案不受本部署影響。

## 修改與更新

```bash
# 修改docs的HTML/CSS/JS後，刷新manifest（只用Python標準函式庫）。
python scripts/build_static_demo.py --manifest-only
python scripts/check_static_demo.py

# Node 22以上；不需要安裝npm套件。
node --test web/static-demo-tests/replay.test.mjs

git add docs scripts web/static-demo-tests
git commit -m "Update driving demonstration"
git push
```

Push後GitHub會自動重新部署。CI會檢查全部檔案SHA、八鏡頭與363影格、HTML連結及回放契約。只要修改了docs任一檔案，就要先刷新manifest；不要手動改bundle_sha256規避資料不一致。

重建原章節需完整 `artifacts/runs/phase10-drive-pedestrian-probe-v1-001`、舊影片證據與Pillow，再執行 `python scripts/build_static_demo.py`。重建新兩章需完整 `phase7-viewer-bundle-v1-001`、`phase10-drive-long-red-probe-v1-001` 及凍結motion設定，再執行 `python scripts/build_showcase_demo.py`，最後刷新manifest。以上只更新docs輸出，不改原始audit或evidence。

影片用實際瀏覽器逐幀點下一幀、等同幀RGB與3D資料提交、透過CUA截圖，再以 `scripts/encode_showcase_video.py --captures <captures.json>` 合成；重製需原截圖收據及OpenCV／Pillow／NumPy，不是用模型真值重畫動畫。號誌小窗為同幀JPEG依模型ROI裁切放大。此維護流程不影響觀眾直接播放既有影片。新增其他路線應擴充exporter、驗證並保存新來源。

## 資料與程式導覽

| 路徑 | 用途 |
|---|---|
| `docs/index.html`／`style.css` | 展示內容與桌機／手機排版 |
| `docs/app.js` | Three.js場景、同幀RGB提交、播放／seek／相機操作；重要生命週期有註解 |
| `docs/replay.js`／`record-contract.js` | 純回放驗證與原導航幾何契約；可在Node測試 |
| `docs/data/showcases.json` | 三章目錄、種類、SHA、預設鏡頭及重點書籤 |
| `docs/data/perception.json` | 60份原始感知record，保留獨立schema，沒有導航或控制 |
| `docs/data/signal-turn.json` | 236份未修改的真號誌左轉record與成功摘要 |
| `docs/data/route.json` | 原67份模型／控制record及來源 |
| `docs/images/<frame>/<camera>.jpg` | 2,904張網頁預覽，從640×360原PNG轉JPEG；不作準確率重算 |
| `docs/manifest.json` | 自排除manifest、全部靜態檔案SHA與bundle SHA；證明一致性，不是簽章 |
| `docs/vendor/` | 既有Three.js 0.180.0／OrbitControls、MIT授權及provenance，無CDN依賴 |
| `scripts/build_static_demo.py`／`build_showcase_demo.py` | 原章與新章的原始record／PNG來源驗證與匯出；前者可只刷新manifest |
| `scripts/audit_showcase_sources.py` | 可選本機audit：逐record數值、原檔SHA及原PNG核對，需要完整artifacts |
| `scripts/encode_showcase_video.py` | 真瀏覽器截圖與DOM收據合成影片，逐幀解碼驗證 |
| `scripts/check_static_demo.py` | 乾淨clone也能執行的標準函式庫完整性檢查 |
| `web/static-demo-tests/` | 時間、跨幀、GT污染、越界URL與壞bytes的測試 |

RGB圖片與模型輸出相同frame才一起顯示；新的seek與章節切換使用generation拒絕晚到結果；載入完成前停用播放及書籤。章節分開載入，首次不下載其餘兩段大型JSON。道路實例緩衝容量按章節增長並釋放舊緩衝，保持單一WebGL renderer。每張當前JPEG都核對SHA並解碼，驗證失敗時停止並隱藏畫面。WebGL不可用時，仍可改看下方影片。切換背景頁會暫停回放，避免回來時突然跳到終點。

## 怎麼展示記憶與紅綠燈

第一章按「暫失0.2秒」：T000003、T000005變成金色線框；「重現／記憶0.6秒」時T000005重新出現在當前ID列表，而T000003仍暫失0.6秒；下一個「過期移除」時T000003已移除。原追蹤器上限0.65秒，沒有延長或以未來影格補物件；同ID重新關聯不等於GT身份驗證。這段60幀中30幀同時輸出car/person/motorcycle，37幀有暫失軌跡；這些是輸出數量，不是準確率。

第二章可依序點紅燈減速、靜止停等、RGB辨識綠燈、重新起步、左轉中、抵達。號誌欄是當時的RGB模型輸出，局部放大只裁切當前鏡頭的模型框；切其他鏡頭時沒有對應框就隱藏。獨立測試director固定燈色，停住約15秒後切綠；driver不讀燈色GT。F72097首次輸出綠燈、F72177解除SIGNAL_STOP，約8秒延遲完整保留。起步時機較慢仍是限制，不宣稱一般交通都能正常處理。

影片第0–12秒為感知記憶，第12–59.2秒為完整號誌左轉；兩段各自保持原順序，沒有把兩段拼成同一次自主駕駛，也沒有略掉停等。影片框數與解碼、原始數值核對、瀏覽器與部署收據見 [展示證據](evidence/web-demo/multiscene-showcase-v1-001/README.md)。

## 常見問題

- **GitHub網站404**：確認Pages設定是main／docs、Actions已完成，而且網址有repository子路徑；不要把整個root設成網站。
- **播放器停在載入**：使用HTTPS或localhost；檢查瀏覽器WebGL是否可用、資產是否完整。不要使用file://。Windows的`.mjs` MIME可能是text/plain，因此瀏覽器模組使用`.js`。
- **SHA不一致**：先確認部署是同一次commit；如是修改了本地docs，刷新manifest再驗證及推送，勿關閉驗證。
- **下載GitHub ZIP後啟動失敗**：先解壓整個ZIP，再於解壓後專案根目錄執行命令，保留images/data/vendor路徑。
- **影片不能播放**：使用新版Chrome／Edge／Firefox；或直接下載WebM。完整互動回放不依賴video解碼。
- **希望自己選新的起終點**：依DEPLOYMENT.md準備本機模型／CARLA，再使用Start-NavigationDemo.ps1。公開網站只提供保存好的三章片段。

## 研究界線

主展示不加入其他actor GT，導航地圖與自車pose是明示先驗。模型可能錯類／漏檢，且實際推論約秒級；1×回放／流暢渲染不等於即時模型。原六條新路線2/6抵達等限制保留於[Phase10](PHASE10.md)與[問題登錄](PHASE10_ISSUES.md)。本次只改展示／文件／部署，沒有改四個模型權重或舊證據。
