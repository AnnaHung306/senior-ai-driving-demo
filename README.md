# 多鏡頭三維環境感知與自主駕駛研究系統

長期需求見 [PROJECT_SPEC.md](PROJECT_SPEC.md)，執行狀態見 [CURRENT_TASK.md](CURRENT_TASK.md)。
目前是 Phase 0：已建置獨立 Python 3.12 / CARLA API、硬體檢查、固定車輛控制與攝影機驗收程式。
實測狀態：CARLA 0.9.16 已完整安裝，API / RPC 曾通過，但本機車輛物理 tick 會造成 server 崩潰（D3D12 / D3D11、兩個車型均失敗）。**Phase 0 未通過**，不可依下列可重現指令宣稱系統已可行駛。
尚未完成自主導航、八鏡頭資料集、AI 模型、BEV / occupancy 或 3D 導航介面。下一步相容性對照見 [DECISIONS.md](DECISIONS.md)。
只有實際 `doctor` 與 `smoke` 都通過，才算 Phase 0 通過；單元測試不能代替模擬器驗收。

## 環境

- Windows 11；本機 RTX 4070 Laptop 8GB、64GB RAM。
- Python 3.12 x64、CARLA server / client **0.9.16**。
- 現有 `.venv` 使用 bundled Python 3.12.14，無須啟用 venv，直接執行下列命令。
- 新環境：`./scripts/setup.ps1 -Python312 '你的 Python 3.12 python.exe 絕對路徑'`；此腳本預設透過系統 `python -m pip --python` 安裝依賴，需要系統 pip 支援此參數。
- `.venv` 不可搬移；大型模擬器、影像、權重與原始 log 不納入 Git。

## 安裝與執行（PowerShell，工作目錄為專案根目錄）

```powershell
# 官方下載約 7.3GiB；已存在檔案不覆寫。
python scripts/download_carla.py
# 中斷下載可選 --resume；若官方 Range 請求停滯，保留原檔並改用新 --directory 完整下載。

# 核對檔案大小、SHA256（本地追溯）、每個 ZIP member CRC 並安全解壓。
.\.venv\Scripts\python.exe scripts/install_carla.py

# 啟動專案專用 server，DX11 / Low / offscreen / 無音效（此設定仍有已知崩潰）。
.\scripts\start_carla.ps1

# 自動等待 RPC，集中執行環境與控制驗收。每次使用新的 run-id。
.\.venv\Scripts\python.exe scripts/verify_phase0.py --run-id ("phase0-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
```

下載大小是 2026-10-16 官方 HEAD 的 7,812,252,810 bytes。若官方檔案改變，先核實 release 與 metadata，再明確傳入 `install_carla.py --expected-size`，不要跳過 ZIP 驗證。沒有取得發布者 checksum，因此本地 SHA256 不等於已完成來源 checksum 比對。

`smoke` 會重新載入 Town01，須使用本專案專用 CARLA server，且只能有一個 tick client。
啟動腳本使用隱藏背景程序，不需要桌面操作；結束後可依工作管理員確認本專案的 CARLA server 程序並關閉，避免占用 GPU。

## 個別檢查

```powershell
.\.venv\Scripts\python.exe scripts/run.py doctor
.\.venv\Scripts\python.exe scripts/run.py smoke --output artifacts/runs/my-new-run
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

退出碼：doctor 0=環境通過、2=依賴未就緒；smoke 0=實測通過、1=失敗。
集中 verifier 將原始輸出保存為 `artifacts/reports/<run-id>-<stage>.log`，每階段最多 300 秒；只輸出精簡狀態。超時屬失敗，不能視為通過。
`smoke` 140 個同步 tick、每 tick 0.05 秒：20 tick 煞車穩定、40 tick 油門 0.55、80 tick 全煞車。
測量自車移動與煞停，保存 14 張 640×360 RGB、內外參、逐幀遙測、事件及 JSON 報告。
門檻在 `configs/phase0_acceptance.json` 先固定，每次 run 複製並雜湊，不在看到結果後降低門檻。

## 產物

- `artifacts/reports/environment.json`：硬體、Python、CARLA API / RPC 與資源檢查。
- `artifacts/reports/installation.json`：archive 來源、大小、SHA256、解壓結果。
- `artifacts/reports/phase0-pipeline.json`：集中驗收結果。
- `artifacts/runs/<id>/report.json`：通過與否、核心數據、錯誤與程式版本。
- `sensors/front/`、`sensor_manifest.json`、`calibration.json`：影像與參數。
- `evaluation/ego.jsonl`、`evaluation/collisions.json`：模擬器自車真值驗收資料；不是 AI 感知結果。

固定程序自行記錄，不需要 LLM 逐幀監看。完整研究方向、未完成項目與風險分別保存在 ROADMAP、ARCHITECTURE、EVALUATION、EXPERIMENTS、DECISIONS。
