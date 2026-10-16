# 技術決策與風險

## D001 — 2026-10-16：先驗證 CARLA 預編譯版

選擇 CARLA 0.9.16 UE4 Windows 預編譯包，client 同版。硬體：Core Ultra 7 165H、22 邏輯核心、約 63.4GiB RAM、RTX 4070 Laptop 8188MiB、初始可用 346.5GiB。
CARLA 官方 quickstart 建議 8GB GPU；這只是啟動可行性依據，吞吐量必須實測。
不先編譯 Unreal，不先建立整個 AI 模型。原生 Windows 能先減少額外 WSL 圖形環境變數。

比較：DRIVE Sim 需先確認授權與可用環境，目前無此證據；Unity / 自製 Unreal 需要額外重做道路、感測與標註，先不採用。縮小實驗以 CARLA Low / 小影像 / 無交通開始；若 GPU 不足，可離線收集、關閉 CARLA 後訓練，或日後指定遠端 CARLA host。這些替代方案都不能替代本機 Phase 0 的實測結果。

## D002 — Python 與 CUDA 分階段驗證

現有系統 Python 3.13.5 不適合官方 CARLA 0.9.16 wheel。已使用 Codex bundled Python 3.12.14 建立專案 `.venv`，沒有修改系統 Python。
bundled runtime 的 ensurepip 失敗，改由系統 pip 指定 `--python .venv/Scripts/python.exe` 安裝 pip / carla。
此 venv 依賴 bundled base runtime 路徑；長期重現應改由獨立維護的 Python 3.12 執行 `scripts/setup.ps1` 重建，不搬移 venv。
目前不安裝 PyTorch / MMDetection3D；CUDA driver 可查詢不等於 PyTorch CUDA 已驗證。

## D003 — 感知候選與八鏡頭風險

BEVFormer 與 BEVDet 有官方開源實作。BEVFormer tiny config 標示至少約 6700M GPU memory，不能據此保證與 CARLA 共用 8GB GPU 可行；其資料相機數與本案八鏡頭也需要調整並重新驗證。
先完成 2D 單鏡頭小批次感知，再使用自採八鏡頭 calibration / grouped split，逐步處理幾何、BEV、時序與 occupancy。
關鍵風險：顯存競爭、Windows 舊 CUDA extension 編譯、模擬域過擬合、scene 洩漏、鏡頭同步延遲、遮擋深度不確定、真值進入推論、sensor timeout 未停止車輛。
對應措施：分時訓練、鎖版與獨立環境、保留場景測試集、資料 API 分離、逐幀 frame ID 檢查及超時失敗；完整閉環安全停止待 Phase 9 實作。

## 研究來源（2026-10-16 查閱）

- [CARLA 0.9.16 release](https://carla.org/2025/09/16/release-0.9.16/)：Python 3.10 / 3.11 / 3.12 wheel；固定採此版本。
- [CARLA quickstart](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)：預編譯包與啟動需求；頁面下載推薦仍寫 0.9.15，版本選擇以 0.9.16 release / PyPI wheel 為準。
- [Official release downloads](https://github.com/carla-simulator/carla/releases/tag/0.9.16)：Windows 官方下載連結。
- [PyPI carla](https://pypi.org/project/carla/0.9.16/)：cp312 win_amd64 wheel。
- [CARLA synchrony](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/)：同步、固定 timestep、單一 tick client 與 GPU sensor 延遲。
- [CARLA sensors](https://carla.readthedocs.io/en/0.9.16/ref_sensors/)：RGB frame、timestamp、BGRA 與相機參數。
- [BEVFormer official](https://github.com/fundamentalvision/BEVFormer)、[tiny config](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py)：相機 BEV / 時序模型候選。
- [BEVDet official](https://github.com/HuangJunJie2017/BEVDet)：多鏡頭 3D / BEV 候選。
- [指定影片](https://www.youtube.com/watch?v=--cGYatMESY)：已嘗試存取但工具無法讀取內容；未據此捏造影片分析，僅採用需求書列出的公開概念。

官方 Windows archive 約 7.3GiB，需另留解壓空間；下載 checksum 僅記錄本地 SHA256，不宣稱已比對發布者 checksum。
