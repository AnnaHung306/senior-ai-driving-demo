# v5 批次工程前置實測

完整GPU suite94/94零skip通過。720x1280 detector batch2容量3350.97MiB allocated/3924MiB reserved，3個dummy更新finite；production load_model重建resize後RGB輸出exact一致。synthetic.pt有保留dummy權重，僅供工程重載，未套用到正式初始best；不是實際訓練成果。

segmentation完整資料verify與CPUcache成立：9600main+1024unique static每輪重複4次，實際sampled13696。80步sanity損失比.423368<.8通過，之後重建模型/optimizer/seed。正式更新前完整val報告與v4 selected best完全一致（含SHA），證明初始化，不代表修復後模型過關。
