# 目前任務

更新：2026-10-16（Asia/Taipei）。原始長期需求完整保存在 PROJECT_SPEC.md。

目前：Phase 2 小型資料管線 pilot 收集與磁碟稽核均通過；稽核見 data/town01-pilot-v1-epic/audit.json。完整 Phase 2 尚未通過，沒有開始模型訓練。

已實測完成：

- Phase 0：原數值門檻通過。正式 phase0-dx12-epic-004：140 幀、14 PNG、位移 6.3031m、峰速 5.1616m/s、最終 0、碰撞 0、時間戳誤差 0、清理成功。
- Phase 1 初始規則 Baseline：五條預先固定的 40m 晴天、無 NPC、非路口路線全部通過；最新 phase1-baseline-005-epic。碰撞、車道線跨越與偏離可行駛車道均為 0。號誌／停車標誌沒有受測，非 AI。
- 八鏡頭 RGB：rig-eight-003-dx12，20 組完整同步影格、160 PNG、時間戳誤差 0、清理成功；約 4.34 組／牆鐘秒，非 20Hz 實時推論。每 5 組取樣的總 GPU 使用量最高 2824MiB，並非連續量測峰值。
- DX12／Epic 的 24 感測器（8 方位 × RGB／語意／深度）：五幀 probe 通過，六場景正式 pilot 收集 180 組、4320 PNG，包含晴天、夜間、雨天與 0／3／6 輛 NPC。
- 收集前凍結整群組切分：train 3 群組／90 組、validation 1 群組／30 組、test 2 群組／60 組。RGB 輸入與 GT 標籤分開；14 項單元測試通過。
- 同一 DX12／Epic server 連續兩次 Phase 0 也通過：005a 初次載入、005b 沿用地圖，避免再次觸發冗餘 map load。

關鍵修正與不要重犯：

1. 不因為「車輛 tick 時 crash」就推斷物理引擎故障。官方 PDB 顯示初期錯誤涉及地形渲染；Town01 只需初次 load，不要對已載入地圖再次 reload。
2. 官方 PixelReader 把 priority bit 512 當成完整 thread enum，誤排 StatsThread。當前使用 AnyBackgroundHiPriTask=2815，只有兩個 byte 不同；原檔備份、固定 hash、source patch 與還原腳本齊全。
3. DX11 單鏡頭通過不代表八鏡頭可用；八鏡頭失敗留下 GPUCrash／D3D device lost 0x887A0020。當前選用 DX12／Epic，已通過單鏡頭、24 感測器與規則回歸；Low 版八 RGB 通過但雨天畫質有問題，保留作歷史對照。
4. NPC 資料曾完整寫入，卻在清理切換時序時 native abort。現在先取消 autopilot 註冊，保持 TM 同步直到 world 恢復完成；同一 NPC probe 與六場景均完整通過。
5. Low／DX12 的雨天 RGB 有黑天空與明顯雜訊；CRC 正確仍不代表適合訓練。固定場景 Epic 對照改善天空／光照，已另收 Epic pilot；仍須量測跨場景畫質與長序列穩定性。
6. 保留失敗結果，只修改有證據的新假說；不能用 GT 冒充模型輸出，不能把延遲影像誤配成當前影格。數值門檻沒有降低。

原生 build：stock ba05ec36...，當前 background b0370d6d...；完整值與診斷見 DECISIONS D005–D009。這是本地實驗修正版。大型 CARLA、影像與 raw logs 不進 Git；精簡證據在 evidence/。

下一個單一任務：擴充 Phase 2 的可訓練資料覆蓋。新的版本化計畫須先固定較長的移動自車 episode、路口、行人／兩輪車、類別與場景分佈及 split，再收集／稽核。本 pilot 保留作回歸；每場景只有 1.5 秒、共用 Town01、NPC 僅單一車型，不足以驗收完整資料集或支撐準確率宣稱。

沒有 AI 模型、checkpoint、BEV／occupancy、3D 導航介面或完整 AI 閉環。模擬器已在集中測試後關閉，沒有訓練、監控排程或其他背景工作。

接手先讀本檔、EVALUATION 最新結果與當前任務程式；不必重讀全部原始 log，也不需逐幀讓 LLM 監督。
