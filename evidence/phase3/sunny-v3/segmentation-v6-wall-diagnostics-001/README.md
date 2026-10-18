# v6 Wall 診斷：跨場景表現仍不足

完成v6模型best 0a3fe...，原GPU全3600val Wall IoU .15645，FP4324560/FN1051136；逐scene與完整混淆重算exact。此部分只讀保存GPU結果，不重推論。

沿用先前train-only外觀/negative選例固定28張，完成v6在CPU FP32的Wall IoU .97739、P .98703、R .99011。兩頁六個最高FP/FN情境已實看：磚石/水泥Wall大致吻合，仍有少量grass/sidewalk與Wall/background誤差。選例有偏、時間相關，不能稱全train accuracy，更不是泛化或正式GPU驗收。選例原報告、契約、exact tool與此次收據一併保留。

此落差支持優先處理外觀/場景轉移，不能斷言純粹容量不足、class weighting是唯一原因，或增加相同磚牆必能解決。Wall-only CPU九候選仍在全資料診斷；原batch後的完整GPU錯例QA仍待完成，未使用heldout、未變動品質門檻。
