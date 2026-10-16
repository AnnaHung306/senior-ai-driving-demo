# 目前任務

更新：2026-10-16（Asia/Taipei）。

任務：Phase 0 — 在本機 CARLA 0.9.16 驗證車輛加速、煞停與 RGB 擷取。
狀態：進行中，官方模擬器下載 / 安裝尚待完成；未宣告 Phase 0 通過。

已完成：

- 原始規格完整保存；建立模組架構、路線圖、風險與初始驗收標準。
- 硬體評估：Core Ultra 7 165H、22 邏輯核心、63.43GiB RAM、RTX 4070 Laptop 8188MiB。
- 獨立 Python 3.12.14 venv；CARLA 0.9.16 API 已安裝並成功 import。
- 環境 doctor、同步 RGB / 控制 smoke、安裝與集中驗收程式。
- 6 項單元測試通過；無模擬器時的失敗報告已實際驗證。

當前執行：官方完整下載；程式等待完成後自動核對 ZIP、解壓、啟動 server、doctor + smoke。
驗收門檻：`configs/phase0_acceptance.json`，首次測試前已固定。

完成條件：實際 server RPC 正確回報 0.9.16，140 個影格完整、影像時間戳誤差 ≤0.0001 秒、儲存影像 ≥10、位移 ≥1m、峰值速度 ≥1m/s、最終速度 ≤0.2m/s、碰撞 0、資源清理成功。

下一項：通過後固定 Phase 1 scenario / route / events，再建立規則式導航 Baseline。未通過時先保存診斷，只修復有證據的問題。

接手時只需先讀此檔、EVALUATION 最新結果、相關程式；不需重讀整份規格或逐秒詢問訓練狀態。
