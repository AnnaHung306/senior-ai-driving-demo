# Phase 7：因果追蹤、運動預測與風險展示

2026-10-20使用者要求「phase7，這是展示核心，要謹慎製作並驗證，一次完成」。本輪完成實作、固定資料重播、獨立GT評估、3D展示與瀏覽器驗證後再交付。沿用晴天、不追Wall／depth、不開development／lockbox；不訓練新感知模型，不發車，不冒稱閉環。

## 推論前固定設計

- 輸入：原始RGB凍結模型的cuboid／RGB crop外觀、timestamp、calibration hash、RGB估計previous→current SE2；不讀actor ID、GT速度、世界pose、未來frame或depth。輸出current-ego座標，物件速度是扣除自車運動後在當前ego軸表達的地面速度，與相對接近速度分開。
- 追蹤：4維XY/VXY Kalman、Joseph covariance更新、SE2旋轉state/cov/history、class／尺寸／Mahalanobis gating、全域Hungarian assignment與RGB顏色hist輔助。tentative／confirmed／coasting／expired生命週期；短遮擋保留ID但不填occupancy，異常時間／motion／跨scene／calibration重設。歷史與診斷有界。
- 預測：至少3次量測且跨0.4秒才提供速度；未校準covariance及可信程度明示。固定速度baseline，0.2秒取樣至3秒，另以固定煞車假設給敏感度分支，無神經網路意圖／機率宣稱。軌跡全部在**當前固定ego座標系**，不可誤當未來ego相對座標。
- 風險：自車由RGB SE2估算速度／yaw rate，在constant-turn假設下與物件預測比較；保守包圍半徑、連續分段接近／碰撞時間、距離及uncertainty-aware可能衝突。低資訊為unknown；無觸發只表示假設內未發現衝突，不能標成安全。路面候選線不當成無障礙規劃，無油門煞車控制。
- 展示：保留Phase6三視角／RGB／時間軸，增加當前速度向量、過去軌跡／未來虛線／不確定範圍、物件運動狀態、暫失標示、風險列表與TTC。關閉圖層可回看原始量測；不得使用GT軌跡繪製主場景。預設線條克制，選取物件突出完整資訊。

## 固定資料與驗收（先於新結果）

1. 真RGB四場景：原train town01-00／town01-01、原validation town04-00／town04-01；canonical training index每scene排序後**[20:80]全60組**，5Hz／每段11.8秒，共240組／1920RGB。不依新模型結果篩選。既有32組只是其中前8組，新增長時窗不當未見資料泛化；不宣稱完整無近牆路線。
2. 凍結Phase3 v6與Phase5 best權重重跑RGB→Phase4→Phase5；獨立RGB-only staging與SHA，來源snapshot在run前凍結，推論期間不改被凍結source。新Phase7以同一物件量測與舊WorldTracker比較，不把感知更新當追蹤收益。
3. 獨立GT：全部當前正例2m class-gated配對，漏檢／FP與追蹤ID switches、fragmentation、IDF1/MOTA（定義與分母）均報告。速度與ADE/FDE @1/2/3秒，用當前capture座標固定投影GT未來bbox中心；不讀未來去修正當前輸出。條件配對分數與所有GT的預測覆蓋／miss／不可評數分開，場景末尾右設限不能冒充成功。
4. 合成幾何診斷另標示非AI／非CARLA：固定seed，跟車、轉彎、切入、突然煞車、同類交叉／短遮擋、自車旋轉、空量測、跳幀／倒序／NaN／GT污染。真值只用於診斷評分；不得拿這些成功取代真RGB模型品質。
5. 事前工程目標：正確SE2補償與velocity <1e-6解析誤差；Hungarian對小矩陣窮舉一致；因果prefix重播exact；無GT／未來洩漏；遮擋0.6秒連續ID；全部Python／JS／HTTP回歸。合成定速已收斂ADE@1≤0.75m、FDE@2≤1.5m；固定明確碰撞／明確分離案例不得漏警／誤警。轉彎／急煞有模型失配便逐案報告，不事後移門檻。
6. 真RGB探索品質目標：速度MAE≤1.5m/s、ADE@1≤1.5m、FDE@2≤3m、ID switches不高於同量測舊tracker。任何未過保留false，不調整GT域或刪難例，不以已見短片宣稱泛化。
7. 真browser核對每一影格ID／速度／軌跡／風險與source一致，三視角、seek／倒放後重設展示狀態、選取／圖層／RGB、手機版、資料污染／context loss恢復。顯示CPU套用p95<200ms、idle render median≥30FPS；獨立列感知／Phase7計算及原timestamp頻率。

## 技術依據

### 第一版關聯失敗後的固定修正

v1真RGB評估（保留phase7-motion-v1-001／evaluation-v1-001）：全部ID switches66對舊50，validation7對舊2；39/66來自行人，近鄰／重複量測與3D jitter使單純Mahalanobis關聯不穩。速度與預測已有訊號，不能因而略過ID退步。下一版**只固定一個關聯修正**：把Mahalanobis按原gate正規化權重2.0，加入同一相機的原2D bbox IoU代價權重3.0；histogram權重與品質門檻不變，不讀GT actor ID當特徵，不改量測或刪重來美化指標。無共同鏡頭不假造overlap。config另存v1.1、run002，不覆寫001；以同一已知模型選擇資料作開發診斷，不稱獨立驗收或泛化。

v1.1實測58 switches仍高於既有50，因此**不採用兩個新關聯器作主幹**。最終v1.2採工程降級：保留Phase5已由RGB產生且來源hash綁定的追蹤ID，逐ID維持causal Kalman運動狀態；離群量測重暖速度filter，不為維持ID假造高可信速度。新實驗關聯器保留為明確未採用的研究對照，非把舊ID當新演算法收益。run003檢查發現離群重暖會重計觀測次數並重設identity age，改用獨立filter_hits／總hits後另跑**run004及evaluation004**；003原樣保存，沒有evaluation003。未改門檻或重跑GPU。ID和舊版相同只表示未造成回歸，不能稱ID改善。

[SORT原論文](https://arxiv.org/abs/1602.00763)支持Kalman＋Hungarian的簡單追蹤基準；本版另做地面SE2與RGB histogram，並非DeepSORT或學習式re-ID。原論文也指出偵測品質影響追蹤，不能用追蹤掩蓋Phase5錯例。[nuScenes官方預測指標](https://www.nuscenes.org/prediction)的ADE/FDE定義作參考；本版單一CV預測，沒有best-of-K或挑選真值最接近分支主評分。

## 本輪交付與驗收結果

**Phase7工程展示研究版完成，正式模型品質仍未通過。** 原規格的跨時追蹤、速度／方向、未來位置、接近風險、四類情境均有實作、量測及可操作展示；四類指定情境目前是分開標示的合成幾何診斷，尚非四種新CARLA交通實測。主畫面則是實際1920張RGB經凍結感知模型推論的240組長序列，沒有GT geometry。行為預測採運動學baseline，沒有學習意圖或保證急煞／轉彎預知能力。Phase8/9導航控制未開始。

| 驗收項目 | 結果及界線 |
|---|---|
| 真資料來源 | 原train兩段、validation兩段，各60組／11.8秒，共240組47.2秒、1920RGB；事前固定索引，已知模型選擇資料，非未見路線 |
| 凍結感知重播 | Phase4／5各240組真正CUDA推論完成；1248／767輸出檔SHA、有限值及來源核對，不冒稱新增BEV品質驗收 |
| 動態結果 | 1091個當前物件、889次可用預測；260筆暫失外推出現次數，非260個獨立物件；不進當前occupancy |
| 因果與傳遞 | 240組全量CPU重算exact，80組prefix exact；原ID／cuboid／age與occupancy bytes不變；2401個展示來源檔核對 |
| 單元／回歸 | Python256項0fail／0skip，15.050s；JS40項0fail／0skip；22項真HTTP checks；新語意污染fixture真browser拒絕 |
| 實際Three.js／browser | 240組／1091物件及889組實際Float32軌跡buffer與來源一致；1920RGB核對；4段完整1×timestamp播放，第二輪geometry drift0，cache16 |
| 互動／失效 | 三視角、選取、時間軸／seek、動態五圖層、八鏡頭、390px主頁與診斷頁無水平overflow；真WEBGL_lose_context停止／reload恢復 |
| 合成情境 | 跟車、轉彎、切入、急煞各4個固定seed，共16案；4種解析circle風險測試全符，含高速穿越；診斷頁全部204組DTO／SVG位置核對 |
| 視覺檢查 | 8面板／7個獨立actor-frame，兩contact頁實看；保留每scene最差、val中位／最佳及兩輪例；最差FDE2為7.616m，未刪除 |
| 本機效能 | Phase7含RGB解碼p50 43.13／p95 52.25ms；browser CPU套用p95 44.4ms、render median60FPS。原資料5Hz及上游感知約秒級，均不是端到端即時能力 |

### 原validation兩序列的主要數字

以下120組是原模型選擇validation，v1／v1.1開發對照也看過，不能稱獨立測試。當前GT412、預測393（25m域）、TP273／FP120／FN139，class-gated 2m配對P69.47%／R66.26%。ID switches **2／245個連續matched transitions**，fragmentation22，IDF1 .6385、MOTA .3665；與相同量測的原tracker完全一致。

259組已配對且有速度的樣本：地面速度**向量MAE .2899m/s**，速度大小MAE .2208m/s。此條件分數不包括漏檢／暖機／不可估速物件。

| 未來時距 | ADE / FDE (m)，只限有預測配對 | GT覆蓋 | 含無預測的所有eligible GT miss | 右設限GT |
|---|---|---|---|---|
| 1秒 | .9502 / 1.0026 | 235/378，62.17% | 40.48% | 34 |
| 2秒 | 1.0347 / 1.2047 | 210/343，61.22% | 47.52% | 69 |
| 3秒 | 1.1579 / 1.4861 | 186/309，60.19% | 54.37% | 103 |

miss定義為該horizon內最大位置誤差>2m或沒有可用匹配預測，不是只看終點。條件miss率依序4.26%／14.29%／24.19%，不能取代上表all-GT數字。事前四項探索門檻通過，但覆蓋及ID／上游量測不足，**formal_quality_passed仍false**。

### 混合train／validation與風險診斷

混合240組僅補充：IDs50（舊50，新v1為66、v1.1為58）；648速度配對向量MAE .4346m/s，2秒ADE .9426／FDE1.2866m，同配對保持原位置baseline FDE2.7241m；自車RGB估速向量MAE .15195m/s。不同評估域不能混成validation成績。

風險GT為3秒**保守circle接近**，不是模擬器碰撞事件：nominal TP38／FN38／FP9／TN660，另11個無GT匹配警報；recall50%，含未匹配警報precision65.52%。寬uncertainty／braking potential為TP41／FN35／FP358／TN311，另124個無匹配警報，誤警很多。介面提示「可能接近／不確定域重疊」，不顯示碰撞機率，不拿無觸發當安全，不接煞車控制。

合成第一seed跟車／轉彎／切入／急煞FDE2分別.280／1.548／1.304／2.688m。跟車工程門檻通過；CV不能預知突然改變意圖。主畫面早期jitter還曾造成反向預測，最差實際FDE2 7.616m；因此不確定速度明標「待確認」、未顯著運動不畫速度箭頭，未改輸出／評分來遮掩錯誤。

## 使用與重現

```powershell
# 從專案根目錄啟動，瀏覽器開啟 http://127.0.0.1:8766/；Ctrl+C停止。
.\Start-WorldViewer.ps1
# 指定既有Phase6 bundle仍可用，動態資料為optional。
.\Start-WorldViewer.ps1 -Bundle artifacts/runs/phase6-viewer-bundle-002
```

預設使用 `artifacts/runs/phase7-viewer-bundle-v1-001`，四序列各60組，點物件查看速度／方向／TTC。上方「動態診斷情境」開啟四種明示非RGB AI的合成示範。時間軸跳回只重播已計算結果，不在browser使用未來資訊修正推論。監看功能沿用Phase6已驗證同bundle pointer；尚未接CARLA連續新key producer。所有測試服務交付前已停止，沒有常駐job。

權威bundle SHA256：`d2a92c8cfe389f9c308b0bbfa98248d93f341d9b8a688de43a4ff0c516232db8`。本機完整RGB／權重及上游NPZ留ignored artifacts/data；Git證據封存JSON/bin、完整分數、失敗版本、sources、瀏覽器收據及圖，不含1920原RGB或模型權重。新機不能只解Git證據就宣稱能重新推論，須有原核准007資料與checkpoint。

```powershell
# 檢查既有權威輸出；全量CPU含RGB重算，輸出診斷到ignored reports。
.\.venv\Scripts\python.exe -X utf8 scripts/verify_phase7_artifacts.py
node scripts/check_world_viewer.mjs artifacts/runs/phase7-viewer-bundle-v1-001 artifacts/reports/phase7-integrity-final.json
.\.venv\Scripts\python.exe -X utf8 scripts/check_phase7_browser_receipt.py
node --test web/world-viewer/tests/*.test.mjs
# 受限測試環境把TEMP/TMP僅設於此PowerShell子程序，勿修改系統全域值。
$env:TEMP = Join-Path $PWD 'artifacts/tmp/phase7-tests'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

需要新動態重播時，以下使用**新的輸出目錄**，不覆寫004或權威evaluation。原始RGB staging與CUDA結果可重用，身份來源先逐SHA驗證。最早staging由`prepare_phase7_inputs.py`固定，Phase4用`run_phase4_pilot.py`＋`configs/phase4_bev_v1.json`（內容版本v1.1），Phase5用`run_phase5_world.py`＋其凍結checkpoint；各run的contract/config/source snapshot為實際版本依據。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_phase7_motion.py `
  --world artifacts/runs/phase7-world-v1-001 --phase4 artifacts/runs/phase7-bev-v1-001 `
  --inputs artifacts/runs/phase7-inputs-v1-001 --integrity artifacts/reports/phase7-world-integrity-001.json `
  --phase4-integrity artifacts/reports/phase7-BEV-integrity-001.json `
  --config configs/phase7_motion_v1_2.json --output artifacts/runs/my-new-phase7-motion
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_phase7_motion.py `
  --motion artifacts/runs/my-new-phase7-motion --world artifacts/runs/phase7-world-v1-001 `
  --output artifacts/runs/my-new-phase7-evaluation
```

本輪未訓練／安裝新套件。主推論仍用專案Python3.12／Torch CUDA；靜態QA圖另用已存在的系統Python3.13 Matplotlib產生，沒有修改系統環境。舊Phase3/4/5/6模型與證據不改。

完整封存見[evidence/phase7/causal-motion-v1-001/README.md](evidence/phase7/causal-motion-v1-001/README.md)。後續先規畫新晴天無近牆完整路線、改善上游漏檢／速度初始化及風險覆蓋，再對未參與開發的資料驗收；Phase8/9需獨立規劃與控制測試，不能用目前展示代替。
