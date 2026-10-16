# 驗收方式與結果

## Phase 3 — 可見 instance 與動態32 sensors工程驗證

最新完成四次工程採集，共640 unique raw PNG（160 RGB），原始資料不作訓練集；44 tests通過，兩次Phase0回歸008/009通過。五案static：32 sensors、160圖、各模態frame/timestamp/pose一致；同ID Rider+兩輪車body確認、移除遮擋後行人14→212 pixels、partial48，截斷與near-plane觀測通過。Axial depth支持率99.657%，range假設76.812%。

可見框由實際mask產生，保留amodal幾何；visibility_policy_v1_1對tiny/rider-only/near-plane/未解析前景明確ignore。修正v1對兩張真實邊緣車輛的過度排除，投影與depth用一致0.15m支持容差，舊policy與audit保留。原static160 actor-view pairs=34 positive/4 ignore/122不可見，不是獨立物件數。尚未將ignore契約接入真正detector RPN/ROI，training_approved仍false。

短動態probe自車0.4196m、label span0.8s、ego collision0；160 PNG全部CRC/SHA/同步通過，capture vs snapshot×rig最大平移差1.839e-5m、rotation matrix差4.22e-7，各模態pose差0。保存實際capture-time transform與32sensor attributes；這是模擬內部一致性，不是實體校正精度，且不代表完整30秒自然互動或正式motion coverage。

相同幾何、同一凍結checkpoint、0 optimizer updates：晴天focus mIoU47.56%、夜雨2.37%；pose差0、raw semantic target差0.0982%。這是已見Town01道路上的40 views/condition診斷，不能取代原test26.22%、不能提供泛化信賴區間或分離雨／低光因果。Assistant視覺review及真實GT/模型圖見 evidence/phase3/visibility-v1/REPORT.md。GPU case-end樣本最大3938MiB，非連續峰值；無八鏡頭20Hz保證。

## Phase 3 — 首輪 segmentation baseline，工程完成、品質未驗收

固定 configs/segmentation_v1_1.json 與 source commit a1eff7b，在 CARLA 關閉時實際完成 16 epochs / 4800 optimizer updates，704.96 秒。Tiny U-Net 從零訓練，原解析度 640×360、488,120 parameters。只 train 更新及算 loss weights，validation 選 epoch 12，最後才開 test。32 tests 通過；訓練後再驗全部 7200 PNG、639 metadata、source/config/criteria/rig 未變，checkpoint / test receipt hash 一致，finite weights 與恢復契約通過。

| Test 語意類別 | IoU | Precision | Recall |
|---|---:|---:|---:|
| 道路 | 36.82% | 60.58% | 48.41% |
| 人行道 | 26.23% | 43.68% | 39.64% |
| 行人 | 13.70% | 14.67% | 67.41% |
| 汽車 | 17.01% | 18.87% | 63.24% |
| 機車 | 49.01% | 55.98% | 79.75% |
| 自行車 | 6.13% | 8.10% | 20.21% |
| 車道線 | 34.63% | 52.10% | 50.82% |

Validation focus mIoU=27.50%，test pooled focus mIoU=26.22%，pixel accuracy=68.00%；constant train-majority focus mIoU=0%，train-prior expected-counts 基準=2.69%。Other_annotated IoU=65.92%，不混入 focus 平均。晴天 scene-04=49.04%，夜雨 scene-05=11.83%；不同幾何／天氣有混淆，不能作單一因果結論。夜雨道路 IoU=1.13%，自行車 recall=3.03%，屬明顯失敗。

保留 validation+test 全 1200 筆 per-view 指標、14 張固定規則錯誤對照圖與 84 筆逐類 FP / FN triage entries。Bicycle 在 test 的 700 張 GT-negative views 上誤報 47,257 pixels；100 張 positive views 總 recall=20.21%。這些影格高度相關，不能當成 1200 個獨立場景或以像素 bootstrap 給泛化信賴區間。

GPU peak allocated / reserved=1382.4 / 1538.0 MiB。Batch1 network p50 / p95=9.07 / 9.85ms；warm-cache 載圖到 mask+confidence=25.64 / 32.04ms。八鏡頭單 bundle 順序處理=217.09ms，不含感測器、CARLA render 或寫檔，不代表 20Hz 閉環。

真實 best checkpoint 另在完全無 labels / audit / plan / indexes 的 RGB+calibration fixture 成功執行獨立推論。成果與限制見 evidence/phase3/segmentation-v1-002/MODEL_CARD.md；原圖、checkpoint、source snapshot、review.html 在 artifacts/runs/segmentation-v1-002。尚無 detection AP／mAP、號誌狀態、完整長尾類別或正式品質門檻；整個 Phase 3 未通過。錯誤分析見同 evidence 下 ERROR_ANALYSIS.md。

## Phase 2 v2 — 基礎資料版本驗收通過

新版 `data/town01-motion-v2-001` 六案收集均通過，各 200 個 0.05 秒 world tick、每四 tick 存一組（5Hz），共 300 組、7200 PNG。每案五個受控角色：一台汽車、一台自行車、一台機車、兩名行人。自車使用已標記來源的 BasicAgent 規則，不是 AI 感知或 AI 駕駛。

| 場景／split | 天氣 | 自車位移 m | 自車 collision callbacks |
|---|---|---:|---:|
| scene-00／train | ClearNoon | 17.6586 | 0 |
| scene-01／train | ClearNight | 17.2544 | 0 |
| scene-02／train | HardRainNoon | 20.4886 | 0 |
| scene-03／validation | WetCloudyNoon | 16.5617 | 0 |
| scene-04／test | ClearNoon | 15.8407 | 0 |
| scene-05／test | HardRainNight | 17.5501 | 0 |

單場景標籤跨度約 9.8 秒；每個 participant 都移動超過 1m。這是 ego collision 檢查，未宣稱所有 NPC 相互間沒有碰撞、交通規則完全受測或有 AI 導航能力。每個 split 的道路、人行道、行人、汽車、機車、自行車與車道線，必須各有至少 1000 個語意像素／10 張正樣本，並由獨立磁碟 auditor 確認。

已完成 19 項單元測試。v2 磁碟 auditor 256.141 秒通過全部 7200 PNG（1,231,370,833 bytes），CRC 錯誤與 timestamp 誤差均 0；所有固定 cadence／motion／class coverage／split／K/T 門檻通過。觀測跨 split ego 最小距離 147.36m；每 required class 最少 86 張正樣本，最少 actor 位移 7.56m。訓練清單 train／validation／test=1200／400／800 view，96 個首尾樣本用 Pillow 獨立解碼通過，舊資料的 regression 也通過。證據見 evidence/phase2/motion-v2/audit-001.json、training-format-001.json。這是有限單城、固定演出的基礎版本，不是泛化或模型品質通過。

失敗保留：full-probe-002 的導航由首次 tick 前 `(0,0,0)` client 座標初始化，導致偏向對向車道並有十次 collision callbacks。經單車實測確證座標生命週期，改從暖機 snapshot 初始化；相同場景 full-probe-003 通過，正式六案也全部零 ego collision。

## 前輪結果 — Phase 0、初始 Baseline 與資料管線 pilot

最終使用本地 background 排程修正的 CARLA 0.9.16 + DX12。

目前預設 **DX12／Epic**：phase0-dx12-epic-004 原門檻通過（140 幀、14 PNG、位移 6.3031m、峰速 5.1616m/s、最終 0、時間戳誤差 0、碰撞 0）；phase1-baseline-005-epic 五路線 5/5。相同場景／rig／seed／門檻另收 `data/town01-pilot-v1-epic`，六案各 30 組、共 4320 PNG 完整收集；其磁碟稽核另存該目錄 audit.json 與 evidence/phase2/dataset-pilot-epic-audit.json，不和下表的 Low 數值混用。

Low 雨天抽樣有黑天空與明顯雜訊；固定場景 Epic 五幀／24 感測器／六 NPC 通過，人工對照看到灰色天空與光照恢復，仍有雨霧／反射／模糊。這支持畫質設定影響部分現象，沒有宣稱所有渲染品質問題均已解決。Low 與 Epic 各抽 48 RGB 的探索性 profile 分別保存，不設定事後畫質通過門檻。Epic probe 後總 GPU 取樣 3382MiB，不是峰值。

下表是先前 **DX12／Low** 的完整性與速度測量，保留作比較；最新單元測試為 14 項，另加入錯誤焦距／反向外參拒絕。Epic auditor 也逐相機檢查 K／T 數值與凍結 rig 的 FOV／安裝位置一致。

| 實測 | 結果與範圍 | 證據 |
|---|---|---|
| Phase 0 v2 | 140 幀、14 PNG、位移 7.6264m、峰速 5.1616m/s、最終 0、碰撞 0、時間戳誤差 0、清理成功 | evidence/phase0/dx12-background-acceptance.json |
| Phase 1 初始規則 Baseline | 預先固定的五條 40m 路線 5/5；碰撞／跨線／off-driving-lane 均 0，終速均 ≤0.2m/s；晴天無 NPC、非路口，號誌未刺激 | evidence/phase1/baseline-004-dx12.json |
| 八 RGB rig | 20 完整組、160 PNG、時間戳誤差 0、清理成功；約 4.34 組／牆鐘秒，平均提交延遲 212.5ms、最大 281ms | evidence/phase2/rig-eight-003-dx12.json |
| 24 感測器資料 pilot | 六場景各 30 組，共 180 組／4320 PNG；RGB、原始語意 ID、24-bit 深度、NPC 3D GT、道路 GT、K/T | evidence/phase2/dataset-pilot-v1b.json |
| 獨立磁碟稽核 | 全 4320 PNG 解壓、尺寸／CRC／逐圖 SHA256、全部 manifest 同幀與標籤配對通過；PNG CRC 錯誤 0；約 683.7MB 影像 | evidence/phase2/dataset-pilot-v1b-audit.json |
| 群組切分 | train 90、validation 30、test 60 組；scene group 交集 0；原點最小間距 164.88m | 同上 |
| 單元測試 | 13 項通過；含原生 callback 資料所有權、同步、PNG 編碼、GT 輸入拒絕、路徑隔離、NaN、群組洩漏與 depth channel 次序 | tests/ |

八 RGB rig 每五組取樣的全 GPU 用量最高 2824MiB；並非連續量測峰值，也不是模型 VRAM。20Hz 是模擬 fixed timestep，不是實時處理速度。

Pilot 只有六個共享 Town01 的停駐自車場景，各 1.5 模擬秒；天氣含晴／夜／雨，NPC 為 0／3／6 輛單一車型。共 630 筆 NPC 3D box 標註，包含不可見／遮擋物件，未做逐相機可見度過濾。語意共有 19 種實際出現的 class IDs；沒有行人、騎士、兩輪車或大型車類別樣本。群組不交集與原點分離不等於所有可見內容完全互斥，更不等於 unseen-town 評估。**此為前輪 pilot，當時完整 Phase 2 尚未通過；本輪 v2 基礎資料驗收見頂部，仍不宣稱模型品質。**

八鏡頭 DX11 兩次失敗均保留；第二次確認 GPUCrash／D3D device lost 0x887A0020，沒有將它誤稱 OOM。NPC 資料清理的兩次 native abort 亦保留，經有紀錄的順序修正後 probe 與六場景正式收集通過。

最終重跑回歸：同一 DX12／Epic server 的 005a 初次載入、005b 沿用 Town01，兩次 doctor／140 幀原門檻均通過，見 evidence/phase0/repeated-map-regression.json。

## 歷史結果 — 首次 Phase 0 v2 通過

`phase0-native-fixed-001`：140 影格、14 PNG，位移 6.3031m、峰速 5.1616m/s、最終 0m/s、最大時間戳誤差 0s、碰撞 0、清理成功，全部門檻通過。
doctor 同時通過。來源程式 `33a4aa2`；數值門檻 SHA256 仍為下方原值，沒有降低。
使用 **本地修正的 CARLA 0.9.16**，不是未修改的官方 binary。Patch hash / 原檔備份 / source diff 見 D005。
v2 protocol 在正式測試前保存為 `configs/phase0_protocol_v2.json`：地圖 preroll、sensor warmup、每 tick RGB，以及一個 tick 的 readback pipeline。
影像必須配對其原本 snapshot；額外最後一個全煞車 tick 只排空 GPU pipeline，不能用來改善量測到的最終車速。
原 v1 的失敗仍是失敗，不回寫為通過。後續 Phase 1 與資料 pilot 結果見本檔頂部。

## Phase 0，protocol v1（實機測試前凍結）

固定輸入：CARLA 0.9.16、Town01、ClearNoon、第一個可用 spawn（記錄 index）、單一 Tesla Model 3、前向 RGB 640×360 / FOV 90°。
同步 fixed delta=0.05s；physics substep≤0.01s。重新載入 world，沒有其他交通或 tick client。
20 settle + 40 accelerate + 80 brake = 140 ticks，油門 0.55 / 全煞車。
此 protocol 是加速煞停與感測驗證，沒有宣稱能跑導航路線。

| 驗收項 | 固定門檻 |
|---|---|
| Python / client / server | Python 3.10–3.12；client=server=0.9.16；本專案使用 3.12 |
| 自車位移 | 從 settle 結束到最後影格 ≥1m |
| 峰值車速 | ≥1m/s |
| 最終車速 | ≤0.2m/s |
| 完整觀測 | 140 個同步影格 |
| 影像存檔 | ≥10 張；每張 640×360 BGRA sensor，存為 PNG |
| 最大 timestamp 誤差 | ≤0.0001s |
| 碰撞事件 | 0 |
| 清理 | sensors stop、actors destroy、world settings restore 均成功 |

缺幀 / future frame / server timeout / 非有限數值 / 清理失敗一律不通過。
門檻：`configs/phase0_acceptance.json`，SHA256 `db9649f3e6cf5c5b9d3c09e2ddc6a43f83810ed94e1cfbe6cc49ff7084ea8def`。
每次 run 保留門檻副本、程式 Git revision、來源標記、逐幀遙測與集中報告。

## 已執行結果

| 檢查 | 結果 | 證據 |
|---|---|---|
| CARLA API import | 通過 | Python 3.12.14 / carla 0.9.16 |
| 初始環境 doctor | 未通過；server 未啟動 | `artifacts/reports/environment-initial.json`；其餘 5 項檢查通過 |
| 6 項單元測試 | 通過 | 同步舊幀丟棄、future frame 拒絕、timeout、門檻失敗案例、投影、NaN 拒絕 |
| 無 server 的故障路徑 | 預期失敗，成功保存報告 | `artifacts/runs/phase0-unavailable-server/report.json`；port 29999，timeout 1s |
| 安裝 / ZIP CRC | 通過 | `evidence/phase0/installation.json` |
| 啟動後 doctor | 曾全部通過 | `evidence/phase0/environment-rpc-ready.json`；這是崩潰前的時間點，不代表 server 現在仍運行 |
| 真正 CARLA 車輛 / 影像驗收，D3D12 | 失敗；第一張影像前 server 崩潰 | `evidence/phase0/simulation-attempt-001.json` |
| 真正 CARLA 車輛 / 影像驗收，D3D11 | 失敗；第一張影像前 server 崩潰 | `evidence/phase0/simulation-attempt-002.json` |
| Tesla 無攝影機物理 probe | 失敗；第一個 tick 後 server 不再回應 | `evidence/phase0/isolation-tesla.json` |
| Lincoln 無攝影機物理 probe | 失敗；第一個 tick 後 server 不再回應 | `evidence/phase0/isolation-lincoln.json` |

初次執行結論（歷史）：未修改的 v1 **未通過**。當時無有效 RGB 存檔或加速 / 煞停指標。
兩個完整 smoke 都維持相同門檻 SHA256；沒有事後降低標準。Probe 僅供隔離故障，不能替代正式 smoke。
當次 server 已退出；最新執行狀態請看本檔頂部與 CURRENT_TASK。

上述為初次 Phase 0 的歷史狀態；當時感知 mAP / mIoU、3D 誤差與 AI 導航均未測，沒有模型或資料集。最新 segmentation 結果以本檔頂部 Phase 3 為準；偵測 mAP、3D / AI 駕駛仍未完成。

## 後續正式評估原則

Phase 1 須先固定路線與事件定義；Phase 2 須先固定場景群組 split。
Phase 3 起，訓練只用 train、調參只用 validation、正式評估只用凍結 test。
逐類呈現結果，保留失敗樣本；模型比較使用同一測試輸入與硬體條件。
探索性 Baseline 可用於訂定未來標準，但不得用同一批結果事後降低門檻宣稱通過。
