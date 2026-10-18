# Phase3 外觀統計增強候選：未整合、未開始訓練

現有v5分割對28個刻意選出的seen train contexts wall IoU .96965，完整val只有.09380；補景已改善Fence，但Wall外觀及岩面negative仍不足。優先依PHASE3_DOMAIN_RECOVERY.md確認實際外觀覆蓋；本文件保留另一個有依據、待驗證的train-only增強候選，不將原論文效果當成本模型成績。

依據為 Zhou et al. 的 [Domain Generalization with MixStyle（ICLR2021）](https://openreview.net/forum?id=6xHJ37MVxxp)與[官方實作說明](https://github.com/KaiyangZhou/mixstyle-release)。原作者以低階特徵的每樣本均值／標準差混合增加style變化，並建議在較低層使用；原始驗證任務不是本專案MobileUNet語意分割。因此這裡只是把已觀察到的泛化缺口轉成可測假說。

獨立prototype位於scripts/experimental_feature_style.py，尚未改任何src或正在跑的trainer。預先指定MobileNet encoder3/6兩層、p=.5、Beta alpha=.1；沿用RGB空間位置與原GT，不混合標籤。每通道做正向仿射轉換不等於語意內容必然不變，必須以全部原gate驗證，尤其稀少動態類不能退步。採FP32 statistics、population variance及原dtype輸出，是本專案BF16穩定性調整，不宣稱精確重現原論文設定。

CPU原型測試涵蓋：eval/p0/singleton的bit-exact identity及RNG不變、恢復CPU RNG後增強exact一致、無新參數／state keys、每通道空間順序保持、constant BF16及反傳finite、實際MobileUNet掛鉤後eval exact/訓練gradient finite。這些只證明工程性質，尚無CUDA容量、全模型GPU determinism、正式恢復或品質證據。

啟用前須等目前唯一GPU batch完全退出；另凍結新source/config/run，source manifest包含實際augmentation helper，正式optimizer/RNG/epoch resume完整保存。可只在正式train forward接入，eval identity，固定view overfit仍明記不含隨機train augmentation；GPU容量另強制啟用分支測finite/backward。用真正production無掛鉤loader重新載入並exact比對，初始完整GPU val應重現parent結果。舊完成run維持原source與權重，不能改active來源或覆寫best。

若執行，保持loss、taxonomy、quality gates、train-only資料及heldout隔離；繼承v5後增加的updates另外記錄。額外訓練算力與增強同時存在，不能聲稱matched-compute因果ablation。資料修復優先；此候選只作有界實驗，無改善就保存失敗並回到確認過的資料缺口，不以調整gate或GT分類宣告完成。
