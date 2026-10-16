# 目前任務

更新：2026-10-16（Asia/Taipei）。

任務：Phase 0 — 在本機 CARLA 0.9.16 驗證車輛加速、煞停與 RGB 擷取。
狀態：已定位 PixelReader 把 GPU 等待任務錯排到 StatsThread，完成有原檔備份的一個 byte 排程修正。修正後 1 tick 延遲、原始幀號／時間戳配對的 RGB probe 通過（3 張）；正準備正式 140 影格驗收，尚未宣告 Phase 0 通過。

已完成：

- 原始規格完整保存；建立模組架構、路線圖、風險與初始驗收標準。
- 硬體評估：Core Ultra 7 165H、22 邏輯核心、63.43GiB RAM、RTX 4070 Laptop 8188MiB。
- 獨立 Python 3.12.14 venv；CARLA 0.9.16 API 已安裝並成功 import。
- 環境 doctor、同步 RGB / 控制 smoke、安裝與集中驗收程式。
- 6 項單元測試通過；無模擬器時的失敗報告已實際驗證。

實際執行：7.3GiB 官方 archive 完整下載、CRC 驗證、約 18.07GiB 解壓完成。D3D12 / D3D11 的完整 smoke 均失敗；無攝影機的 Tesla / Lincoln 物理隔離測試也失敗。沒有有效相機影像或行駛指標，不宣稱完成車輛控制驗收。
驗收門檻：`configs/phase0_acceptance.json`，首次測試前已固定。

完成條件：實際 server RPC 正確回報 0.9.16，140 個影格完整、影像時間戳誤差 ≤0.0001 秒、儲存影像 ≥10、位移 ≥1m、峰值速度 ≥1m/s、最終速度 ≤0.2m/s、碰撞 0、資源清理成功。

下一個單一任務：依 DECISIONS.md D004，在獨立資料夾與 client 環境評估 CARLA 0.9.15 是否可完成相同物理 / RGB probe；先核實官方 wheel / server 可用性，不混用 0.9.16 client，不改動本次門檻。不再重跑已失敗的 0.9.16 設定。替代版本尚未下載或測試。

當前執行：`scripts/diagnose_landscape.py batch --output artifacts/runs/landscape-v1`；每案獨立 server / log / report，避免重跑既有相同條件，成功即停止矩陣。沒有進行中的訓練或排程。Phase 1–10 均未開始。

接手時只需先讀此檔、EVALUATION 最新結果、相關程式；不需重讀整份規格或逐秒詢問訓練狀態。
