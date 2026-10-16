# Phase 3 訓練基礎與實驗協議

本協議與 configs/segmentation_v1.json 在首次訓練前建立。第一個實驗是從零訓練的小型單視角語意分割網路，八個方向共用權重。Phase 3 的物件偵測、號誌狀態與更廣類別覆蓋仍須另做，不以本實驗代替整個階段。

## 資料與任務定義

- 保留 Phase 2 的整場景群組切分：train 3、validation 1、test 2 個獨立場景。時間相鄰幀與八個鏡頭互相關聯，2400 張不是 2400 個獨立場景。共享 Town01；不能宣稱跨城鎮、真實道路或未知車型泛化。
- 重驗全部影像 SHA256，並對影像清單、input/label/calibration JSON、split、資料規格建立完整指紋。跨 split 相同 RGB hash 必須阻擋；沒有相同 hash 仍不能證明沒有視覺重疊。
- 模型輸入只有 RGB / 255。深度、相機位姿、模擬器 actor 與 GT 不進入 forward。原圖 640×360，不縮圖以免自行車／車道線消失。GT 使用原始整數 ID，禁止雙線性插值。
- 七類 focus：道路、人行道、行人、汽車、機車、自行車、車道線；其餘已標註類別合併 other_annotated。raw 0（未知）忽略。other 不是 free space；大型車與號誌等未個別學習，輸出不可用作導航。
- 訓練前列出逐 split 支持像素／正樣本數，僅 train 的統計產生 loss weights。固定 inverse-sqrt frequency，上限 20。保留原類別映射、逐鏡頭與逐場景結果，避免整體準確率掩蓋少數類失敗。

## 可重現與工程驗收

- 固定 PyTorch 2.10.0 CUDA 12.8、Python 3.12、NumPy / Pillow 版本。PyTorch 官方安裝來源：<https://pytorch.org/get-started/previous-versions/>。CARLA 關閉後才使用 GPU；不靜默退回 CPU。記錄 driver、CUDA、裝置、套件、Git、source/config/data hashes、耗時及 GPU 峰值。
- Tiny U-Net：16/32/64/128 channels、GroupNorm、skip connections、8 logits；不下載預訓練權重。AdamW、16 epochs、batch 4、固定 cosine LR、AMP float16／float32 CE、梯度裁切。只有 train 做亮度／對比增強；validation / test 無增強。無空間增強，為後續幾何任務保留清楚契約。
- 正式訓練前 CUDA forward/backward finite 檢查、獨立兩張訓練圖 80 steps overfit sanity check：loss 至少降低 25%。只作工程驗收，不等於品質達標；正式模型重設 seed / 權重 / optimizer。
- 每 epoch 記錄 loss、LR、validation metrics、梯度／非有限檢查。last 與 best 原子 checkpoint 包含 optimizer、scheduler、scaler、所有 RNG、config / data / code 指紋及下一個 epoch。恢復時拒絕不一致來源；只保證同環境 epoch 邊界恢復，不保證跨硬體位元一致。
- CUDA deterministic algorithms（發現不支援則明確失敗），禁用 TF32 / cuDNN benchmark。無 NaN／Inf、checkpoint 可重載且 forward 相同、RGB-only 推論成功、資料邊界與指標單元測試為工程門檻。OOM 或中斷保留 last 與失敗收據，不以降低設定覆蓋同一實驗。
- 首次 GPU 測試發現 PyTorch 2.10.0 CUDA NLLLoss2d 的 mean / sum reduction 使用 atomicAdd，strict deterministic 模式拒絕；依[官方該版源碼](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/NLLLoss2d.cu)改用 reduction=none，再以 tensor.sum / 有效像素權重和求 weighted mean，保持同一 loss 定義與 deterministic 門檻。另測 ignore 像素零梯度、weighted mean 等價與無標籤 batch 拒絕。

## 評估與選模

- 只用 validation 的 supported focus macro IoU 選 best（相同分數取較早 epoch）。test 不進 optimizer、類權重或選模。完成訓練後對 best 作一次最終 test；失敗續跑與已產生的 test 收據不得靜默覆寫。日後根據 test 改模型，必須宣告 test 已成開發資料，補收新 lockbox。
- 保存真實列 × 預測欄 confusion matrix、逐類 IoU / precision / recall、GT / predicted support、overall pixel accuracy、focus macro IoU、frequency-weighted IoU。沒有 GT 也沒有預測的類別為 null，不能當作 IoU=1；focus macro 附上納入的類別數。背景多數類常數預測、train prior 隨機預測的解析期望作為對照。先記錄 untrained validation control；不用它挑 seed。
- 額外按 scene、camera、weather 分解，列出驗證集及測試集最差幀、各 focus 類最差正樣本與真實 RGB/GT/prediction。樣本選擇依固定規則，不能只挑漂亮結果。兩個 test groups 不足以提供可靠泛化信賴區間，不做像素 bootstrap 假精確。
- batch=1 CUDA 同步、固定 warmup、50 次量測 p50/p95，分開 network-only 與載圖到結果端到端；輸出 checkpoint hash。八方向序列推論需實測，不從單張延遲聲稱八鏡頭即時 20Hz。
- 本次無預設感知品質通過線，所有品質數字標示 exploratory。先建立基準與錯誤清單，之後以新場景設定正式品質門檻；不能用本次 test 結果反推本次通過線。

## 必須補收／補作的下一步

1. 以獨立長 episode、城鎮、天候／時間、車型、人流密度擴充；加入無目標負樣本、遮擋、逆光、彎道／路口、交通燈狀態、罕見大型車，場景層級 split。train 的類別不足先改資料，而非以 test 重採樣或複製影格製造樣本數。
2. 物件偵測需有可見性正確的 2D instance labels。目前 actor 3D box 含畫外／遮擋物件，直接投影會產生錯標。補 instance segmentation / actor ID、深度遮擋、near-plane 裁切與畫面截斷規則、可見像素門檻、人工抽查後，才凍結 detection 資料版本；AP50 / AP50:95、逐類 PR、small-object AP、可見性分層評估另立協議。語意連通區不可假裝 instance GT。
3. 行人／兩輪車召回、車道線連續性、夜雨失敗與 confidence calibration 應優先分析。semantic confidence 是 softmax 最大值，不是經校準的安全信心或碰撞機率。後續補 ECE / reliability、OOD、失效退讓政策，再接 BEV。
4. 本階段不提供追蹤、速度估計、3D 位置、號誌相位、道路可通行性或駕駛安全保證。推論 artifact 明確標示 learned_prediction，不混入 GT；檢視報告中的 GT 僅為評估用途。

## 命令

```powershell
python -m pip --python .venv/Scripts/python.exe install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe scripts/train_segmentation.py --config configs/segmentation_v1.json --run artifacts/runs/segmentation-v1-001
# 同環境中斷後僅續接完整 epoch：
.venv/Scripts/python.exe scripts/train_segmentation.py --config configs/segmentation_v1.json --run artifacts/runs/segmentation-v1-001 --resume
# inference 不需要 labels：
.venv/Scripts/python.exe scripts/predict_segmentation.py --checkpoint artifacts/runs/segmentation-v1-001/best.pt --root data/town01-motion-v2-001 --manifest inputs/frames/scene-04/FRAME.json --camera front --output artifacts/runs/prediction-001
```
