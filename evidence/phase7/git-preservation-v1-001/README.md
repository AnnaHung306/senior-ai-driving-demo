# Phase7 Git換行保存補充

主證據`../causal-motion-v1-001`封存後，提交前額外核對發現Git原有`*.json text eol=lf`會將v1.1／v1.2設定CRLF改為LF，造成checkpoint-free replay的config SHA guard在新checkout不符。原演算法、設定值、推論及封存bytes均未變。

因此在根`.gitattributes`追加`configs/phase7_motion*.json -text`，保留三份事前設定原bytes。此補充晚於主證據source-snapshot.zip，**使用Git最終attributes或本補充attributes-final.txt**，不要以ZIP內較早的attributes取代。這是Git保存規則修正，沒有重跑模型／改分數，也沒有重寫主封存。

提交前已核對final motion contract的`motion_forecast.py`、`run_phase7_motion.py`及`phase7_motion_v1_2.json`三者工作目錄與Git暫存SHA完全一致。所有evidence的raw Git bytes另以`check_evidence_git_bytes.py`檢查；完整診斷保留本機ignored reports，不把自我參照proof放進自身。
