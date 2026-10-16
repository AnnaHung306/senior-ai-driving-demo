# 驗收方式與結果

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

結論：Phase 0 **未通過**。無有效 RGB 存檔或加速 / 煞停指標，不能進入 Phase 1 或訓練。
兩個完整 smoke 都維持相同門檻 SHA256；沒有事後降低標準。Probe 僅供隔離故障，不能替代正式 smoke。
目前 server 已退出；doctor 若再執行應回報 server 未就緒。

感知 mAP / mIoU、3D 誤差、導航成功率與真實影像 FPS：**未測試，沒有數據**。
目前無模型、無資料集、無 AI 辨識畫面；模擬器自車遙測清楚標記為 GT。

## 後續正式評估原則

Phase 1 須先固定路線與事件定義；Phase 2 須先固定場景群組 split。
Phase 3 起，訓練只用 train、調參只用 validation、正式評估只用凍結 test。
逐類呈現結果，保留失敗樣本；模型比較使用同一測試輸入與硬體條件。
探索性 Baseline 可用於訂定未來標準，但不得用同一批結果事後降低門檻宣稱通過。
