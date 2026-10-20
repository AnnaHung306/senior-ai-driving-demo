# RGB CuboidNet v1 模型卡

狀態：**有限研究基準，未通過可靠 3D／正式 Phase5 品質驗收**。本次新增的 3D 模型與先前已拒絕的 segmentation depth auxiliary 不同；不採用該深度分支，也不變更 Phase3 三個既有模型。

- 從零訓練的 94,424 參數 RGB crop CNN（4層 stride2 conv／GroupNorm／SiLU＋geometry MLP）；無外部預訓練權重。
- 輸入：64×64 RGB、2D bbox 幾何、固定 K／相機安裝／地面交點、2D 類別 one-hot。沒有 GT depth／pose／actor ID 或模擬器物件資料。
- 輸出：camera-local 中心殘差3、log尺寸殘差3、heading cos/sin2；解碼為 ego 公尺／完整 LWH／弧度 yaw。地面 z=0、upright pitch/roll 和 train class尺寸中位數是明確先驗；不是精細重建。
- 主007資料：原train 8 scenes，659 eligible positive crops；Town04 validation 3 scenes，232 crops。按 scene 排序 stride8，未截斷、最短邊≥6px、ground anchor≤30m、GT bbox中心≤25m。整個 scene 不跨split，未開 development／lockbox。尺寸先驗只從train獨立actor算。
- 固定12 epochs、batch64、AdamW lr0.001／weight_decay0.0001、seed20261020、FP32、按類反頻率有放回取樣，無 flip augmentation。132更新，best epoch11（從0開始），encoder／head初始梯度 .0478117／.3272012，重載232 validation outputs exact，CUDA allocated peak92.74MiB。耗時為模型訓練內部計時，不含資料準備與程序啟動。
- best.pt SHA256：`1ad8ccbf8e5409b476219b8ecaf0924922a756c19b845043b9cbeae99fd32f6f`。原權重與 optimizer／epoch checkpoints 只在本機 ignored `artifacts/runs/phase5-cuboid-v1-001/`；Git 保留 hash、來源與結果，沒有假稱包含權重。

232 GT crop/class條件驗證：中心3D平均1.204→0.931m、平均IoU3D .195→.231；heading平均絕對誤差 **53.03→72.05°**，不是全面改善。行人中心 .651→1.239m 退步。L/W先驗在這些相同資產上原本誤差為0，網路沒有證明尺寸學習優於先驗；行人height更有變化。同一車型／資產跨train/val重複，不能宣稱未知車型形狀泛化。

端到端16組validation短片的嚴格定位P/R=61.11%／55.93%、IoU3D≥.25 P/R=27.78%／25.42%；主分數不略過2D ignore。補充遵循原2D ignore政策，排除5個全部來源均屬ignore且不重疊已知positive的未匹配假設後，定位P為67.35%，R不變；原始分數保留。這些仍是validation選模資料、兩個1.4秒已知序列，不是獨立heldout。

兩方法共同配對的25個validation actor-frame：中心3D .649→.831m、heading29.70→49.11°、IoU3D .248→.211，均顯示先驗基準在這個子集合更好。不能只拿混合train/val的P75.35／R78.68宣稱模型成功。

`orientation_reliable`欄位只表示heading向量長度／融合一致性的啟發式有效檢查，**不是已校準正確率**；WorldFrame limitations與圖上明標低可信度。現在不把它當可靠朝向、車道規劃障礙或安全控制依據。下一版先改善對稱方向建模、跨角度／尺寸多樣性及行人幾何，再做事前固定的新資料驗證；本輪不自動加訓或搜尋超參數。
