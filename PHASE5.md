# Phase 5：RGB 三維物件與 World Model

**2026-10-20補充：已依授權與Phase6一併完成工程再驗證。** 891原始裁切／GT targets與232筆新CUDA reload exact，沒有更新權重。32份WorldFrame已接本機互動3D，147物件的Python／Three.js／真browser座標一致，235項Python與20項JS測試通過。詳見[聯合結果](PHASE6.md)與[證據](evidence/phase6/world-viewer-v1-001/README.md)。原模型、GT評分與舊證據不改；既有品質未通過仍成立。

2026-10-20 使用者授權「很好，繼續phase5」。Phase4 的幾何 BEV 原型可作依賴，但其完整路線、延遲及物件可靠度尚未驗收；開始 Phase5 不把那些限制改成通過。沿用晴天、平坦、無近牆循線範圍，不重訓 Wall、不採 depth auxiliary，不開 development／lockbox，不啟動 CARLA／控制。

## 第一個實作里程碑

1. **真正學習的 3D box head**：在既有自訓 2D detector 後接小型 RGB crop CNN，輸入裁圖、2D box、校準幾何及預測類別；學習相對地面交點的 camera-local 中心殘差、長寬高及朝向 sin/cos。類別尺寸中位數僅從 train 算，另報純幾何／類別先驗 baseline。不把固定車型尺寸或 GT位置冒充網路預測；不是 Mousavian 方法的完整重製，也不是端到端八鏡頭 transformer。
2. **八鏡頭 cuboid 融合**：轉回 ego x前/y右/z上，完整長寬高（非 extent）、yaw 以弧度；依類別、不同鏡頭、完整群組距離限制去重。保留每個來源與未校準偵測分數；朝向向量互相抵消時標記不可靠，不以平均角度處理 ±π。
3. **基本連續 ID**：用 Phase4 的 RGB previous→current SE(2) 補償自車運動，以同類、有限距離、一對一匹配維持 ID。只有當幀實際量測輸出可佔據，短暫漏檢軌跡可留 metadata 但不得當成目前已觀測物件。斷流、回時、標定／場景切換或配準失敗重設，ID 保持命名空間唯一。Phase7 才處理完整速度／預測，這輪不宣稱已完成。
4. **可追溯 semantic occupancy**：0.5m 三維格點，將可見地面與預測 cuboid 體積融合；以可見地面射線作有限 free 證據，未見處維持 unknown。明確標示這是 ground-plane＋cuboid 近似，不是從影像重建精細表面或學習式 occupancy network。路緣只作道路／人行道語意邊界，不假稱量測其高度。
5. **WorldFrame 契約與圖像**：時間戳、ego 座標、單位、source hash、模型／來源／假設、追蹤 ID、cuboids、道路／候選路徑、occupancy 檔案與未知率，能供 Phase6 接入。輸出真正推論的 cuboid 投影與三維格點圖，與 GT 圖分開。

## 固定的有限訓練與驗證

新模型從零訓練；2D／Phase4 三模型權重不動。主資料007原 train 的8 scenes與validation的3個Town04 scenes，每 scene 按排序取 stride8 全程固定樣本；只取可見 positive、類別1..6、未截斷且地面交點存在／range≤30m、GT中心≤25m 的 crops。GT中心以 `world_vertices_m` 平均，不用 actor 原點；真實 camera capture transform 只轉換訓練／評估 targets，推論只讀固定 rig。尺寸為2×extent；yaw保留 bbox-local rotation。類別無足夠 train 支持則不輸出學習式3D，報 not_assessed，不捏造可靠度。

固定一次12 epochs、batch64、AdamW lr0.001、seed20261020、無超參數搜尋；validation選best後不再加輪。保存資料／GT／RGB inventory、分組、每類數量、原始targets、增強、梯度、CUDA／checkpoint reload 證據與每輪log。橫向翻轉須同時轉換幾何／yaw，第一版不作不一致的 flip augmentation。

分開評：(a) GT 2D crop條件下的 head 中心／尺寸／yaw及先驗baseline；(b) 原 Phase4 固定32 bundles 的真2D detector→3D→融合→追蹤→occupancy。不把條件指標當端到端品質，不把已見 train／model-selection validation 當新 heldout。完整已知短片先做 baseline，無事前可靠品質數值時不事後定門檻宣稱正式pass。

工程驗收在實跑前固定：座標／corners round-trip<1e-6、相同box IoU3D=1／不相交=0、yaw環繞正確、去重不跨類／同鏡頭、不讀GT與唯一ID、跳幀重設、unknown不算free、box優先於free、資料split無scene交集、有限非零梯度、checkpoint重載exact；完整既有回歸必須通過。

品質報告：各類中心3D／XY／Z誤差、L/W/H MAE、yaw與π對稱yaw誤差、IoU3D與BEV IoU；端到端P/R@2m與IoU3D@0.25、FP/FN、GT支持去重及ID switches／配對數；occupancy逐類IoU、free誤判與unknown覆蓋。GT正例全域與預測評分域不得依預測遮罩縮小，無支持類別明記。Ground-depth表面與完整GT actor cuboid分開評，不拿表面GT宣稱完整體積重建。真實計算時間與Phase4上游成本分開。

## 依據與限制

- [Mousavian et al., CVPR2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Mousavian_3D_Bounding_Box_CVPR_2017_paper.html)：以影像學習尺寸／朝向並結合2D幾何的可行方向；本專案另做輕量殘差回歸，不冒稱該論文結果。
- [CARLA 0.9.16 bounding boxes](https://carla.readthedocs.io/en/0.9.16/tuto_G_bounding_boxes/)及[官方 BoundingBox 原始碼](https://carla.org/Doxygen/html/d2/dfe/LibCarla_2source_2carla_2geom_2BoundingBox_8h_source.html)：extent為半尺寸，world vertices包含bbox local transform及actor transform。

不以本輪小模型、cuboid佔據或短片追蹤宣稱完成精細geometry、即時world model、長程追蹤、未知路線泛化或AI駕駛。每個失敗保留原因與原始報告，只有具體證據才修改重跑。

## 首輪實測結論

固定12epochs／132updates訓練已完成（659 train／232 validation crops、六類、94,424參數），真CUDA replay與獨立GT評估完成32組；223項回歸測試全部通過。16張contact頁及6組完整prediction／GT錯例均已實看。可追溯成果見[圖文報告](evidence/phase5/rgb-cuboid-world-v1-001/README.md)與[模型卡](evidence/phase5/rgb-cuboid-world-v1-001/MODEL_CARD.md)。

工程基準可輸出WorldFrame，但**Phase5品質仍未通過**。validation16組的嚴格定位P/R=61.11%／55.93%、IoU3D≥.25 P/R=27.78%／25.42%；混合train／validation32組的75.35%／78.68%不能冒充泛化成績。另保存既有2D ignore政策補充（validation P67.35%，R不變）與不改動的主評分。共同25個validation配對上，先驗→學習中心3D .649→.831m、yaw29.70→49.11°、IoU3D .248→.211，部分退步明記。

232 GT crop條件下中心平均.931m／yaw72.05°，尺寸先驗已能記住相同模擬資產；未證明可靠朝向或未知車型尺寸泛化。基本追蹤11次ID切換／86連續成功配對transition。occupancy car65.22%、ped4.77%、bike51.12%、motor18.85%，GT已知域覆蓋19.31%、GT occupied誤判free4.23%；不是可靠可通行空間。全部replay stage加總p50約1.214秒（含上游，未含檔案與控制），未即時。

地面slab被射線誤標free的歧義已在預測與GT兩邊修正，原run001 report/source保留。run002全部32組3D物件／prior／tracking對001 exact，沒有加訓或改模型。3572原始RGB／GT檔重新核對、232條件指標與32組全部計分／occupancy／tracker IDs重算一致。原始pending字樣不改bytes，實看與決策以獨立收據為準。

本輪停止於已驗證的研究基準，不自動追加訓練。後續先改善方向對稱／行人幾何／重複ID競爭與free證據，凍結新路線及門檻再驗收；Phase6可接此schema展示低可信研究輸出，但不宣稱精細3D或駕駛可靠性。
