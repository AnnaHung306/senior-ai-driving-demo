# 深度輔助A/B：準備與CPU對齊證據，GPU尚未執行

使用者已同意小GPU檢查＋兩組各1輪的建議。本次設計固定32步GPU/BF16容量／finite gradient／reload預檢（更新丟棄、重建初始化），再control與depth各1epoch、每組13696sampledviews／1712updates。相同parent v6 best0a3fe...、原train與兩核准補景、原class loss／LR／photometric與幾何，只有depth分支／監督不同；head6945參數、loss係數.2／headLR.003。保留原門檻，另列排除Wall的15類研究均分，不打開heldout或正式採用。

主GPU仍由detector原批次使用。一次性當前程序exec78878等待整個batch002及既定postQA都結束，才require_idle_gpu並執行；非排程服務，不自動追加第二輪／改batch／CPU fallback。最多等6h，own awake綁同一parent instance、退出即清，無改powerplan。**此封存只有準備／CPU證據，不是GPU已跑或A/B已完成。**

4項focused CPU tests已通過：三種scale、多seed下RGB／semantic／後續Python RNG與原production流程bitwise exact；depth／valid nearest同步、不按影像scale縮放米值、padding無監督；空mask無gradient；nonWall診斷完整15類及screening不自動擴大。另實讀main／old scenery／new reserved expansion各2張，共6個真train視角，原始depth／semantic SHA核對、cache上限2時正確逐出、augmented RGB/semantic及RNG exact。完整資料support仍待GPU階段正式載入後與parent重新核對。

CUDA使用nearest loss上採樣以沿用現有模型已驗證的deterministic backward路徑；先前CPU小測loss用bilinear，差異已明記，不把它當相同數值。head構造額外包住CPU＋全部CUDA RNG fork，之後每arm重設同seed；全epoch逐batch雜湊實際augmented RGB／semantic bytes、indices與geometry，兩組不一致就拒絕比較。GPU32步將驗證真BF16 eligible convolution、FP32 depth head與depth-only梯度；現在不假稱已通過。

每組完整3600val、原gates／逐類／逐scene／margin、checkpoint重載及原batch錯例重播；QA選每scene最大nonWall改善與退步，待assistant實看。研究screening預先定nonWall gain≥.002、margin降幅≤.01且無新nonWall單類gate失敗，僅表示可供review，不代表統計顯著或正式採用。現有branch為一次one-epoch研究，**不自動跑第二輪**。weights只在ignored local artifacts/runs，來源原bytes封存於tar。
