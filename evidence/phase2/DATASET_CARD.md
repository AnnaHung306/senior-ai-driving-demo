# Town01 多鏡頭資料 pilot v1 — Epic

用途：驗證擷取／同步／校正／GT 邊界／群組切分與磁碟完整性，不是完整 Phase 2 或模型品質驗收。

- 本地修正 CARLA 0.9.16、DX12／Epic、640×360、八方位 RGB／語意／深度。
- 六個固定場景各 30 組，共 180 組、4320 PNG，影像約 677.4MB。
- train／validation／test 為 90／30／60 組，整個道路場景群組不跨 split。
- 預先按幾何選六個不同道路，原點最小距離 164.88m；共享同一 Town01，不保證可見內容完全不重疊。
- 晴天、夜間、雨天，0／3／6 輛 NPC；自車停駐，每場景 1.5 模擬秒。
- 630 筆 NPC 3D box GT；包含遮擋／視野外物件，未標逐相機可見度。
- RGB inputs 與 labels 分存；正式 RGB-only 推論 API 不讀標籤。
- 原始語意 red class ID、24-bit packed RGB depth，不是調色盤／灰階展示圖。
- audit.json：4320 PNG CRC／解壓／尺寸與逐圖 hash、每圖 frame／timestamp、K/T 與凍結 rig 數值、GT 對齊及 split 均通過。

限制：短序列高度相關、無移動自車、NPC 單一車型，缺行人／騎士／兩輪車／大型車。雨天有雨霧與模糊；Epic 對照改善 Low 黑天空／雜訊，不代表所有渲染品質已驗證。沒有訓練 checkpoint 或感知準確率。

正式訓練前須版本化擴充覆蓋、固定資料分佈／split／品質標準，並驗證較長移動序列。此 pilot 保留作管線回歸。原始需求見專案 PROJECT_SPEC.md。
