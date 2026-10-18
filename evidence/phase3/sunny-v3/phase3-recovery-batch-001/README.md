# v5完整批次：仍未通過Phase3

三模型已完成訓練、原門檻統一val/joint、八RGB demo、三種warm benchmark與export。未開任何新heldout。det mAP50 .70852、唯一bike R .33745失敗；seg mIoU .59003、wall .09380/margin .49447失敗；phase crop macro .94381、joint .46671且matched red→green0通過。原11 stages return0，batch state quality_failed（shell包裝exit1，非stage crash）。

GT/ignore/逐場景與size-strata完整detector重新計算exact，見detection_artifact_check.json。五頁15個saved GPU det錯例及三display zoom已實看。原export號誌joint precision使用legacy v1忽略缺陷；**應參照joint-ignore-v2-addendum**：289個unscorable排除後FP42，與det AP50一致，GTmatrix/recall/quality完全不變。四contact32個修正後phase錯例已實看；此修正是評分修正，不是模型改善。原報告保留供追溯。

8view warm integrated p50 699.08ms/p95 741.78ms，排除sim/render/control，不能宣稱20Hz。Demo已實看並保存demo_visual_review.json；仍見兩輪車多類重框及pole-like誤報；任何導出存在不等於Phase3通過。
