# Observable light-state v1-002 — GT crop通過，detector v3聯合驗證失敗

- Run：`artifacts/runs/sunny-light-state-v1-002`；16epochs/832updates，best第9輪。最佳SHA `e4fdbc431dcf0db5d772ba0724ee74a00f4ea0fb525025c42db12493859ee60b`，exact reload通過。
- 四類unknown/unobservable、red、yellow、green；native phase需另有至少2個語意lamp mask內RGB相容顏色pixel支持，否則unknown，不使用背面隱藏相位當已知target。
- RGB 32×64裁切，四層GN卷積8/16/32/64、global mean與4類head。AdamW lr.001，batch64，class weights inverse frequency clamp.1..10，train brightness/contrast.9..1.1。
- 原v1-001完成更新後JSON收尾失敗，failure原樣保留。修正單一NumPy bool→Python bool後新版本從頭重跑，best/last模型tensor均與舊run完全相同；未修改訓練數學或品質門檻。
- 共同主資料007與Town切分；train3269 crops，validation2368 crops。每類原route群組數與counts見coverage.json。相同route的連續frame不是獨立場景。

## GT crop驗證

Macro recall **0.94381170**；accuracy **0.93623311**。紅→綠10/753=**1.328%**，≤凍結2%門檻。所有GT crop門檻通過，逐張混淆與state SHA核對通過。

| 類別 | GT crops | Precision | Recall |
|---|---:|---:|---:|
| unknown_or_unobservable | 1429 | 0.934592 | 0.969909 |
| red | 753 | 0.959004 | 0.869854 |
| yellow | 31 | 0.885714 | 1.000000 |
| green | 155 | 0.868263 | 0.935484 |

## 尚待完成與限制

- **actual detector-v3 boxes joint已驗證並失敗**：含漏偵測的正確狀態macro recall .36020448 < .4，matched red→green0%；[完整證據與實看錯例](../light-state-v1-002-joint-validation/)。GT框分類通過不等於實際物件框系統合格，整體號誌與Phase3未通過。
- Yellow只有31個validation crops/2原groups；紅→綠仍有可見錯例。11類型錯例與3頁實際視覺檢查見相鄰visual_review.json。
- 主lockbox黃燈4crops/1group不足；預先凍結的Town02 Off/R/Y/G補收已完成68triplets／204PNG、independent audit與實看批准。controlled unknown38/R14/Y14/G14、各2原groups，只供後續heldout號誌評估，不列本run訓練；尚未對模型打開heldout，也不是新獨立路線。
- 未開development/lockbox。GPU訓練6.171s不含全資料verify與crop前置，也不是推論benchmark。置信值未校準，無駕駛或3D驗收。
