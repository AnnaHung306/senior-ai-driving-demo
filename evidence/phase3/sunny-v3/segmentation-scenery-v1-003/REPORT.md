# 分割訓練專用靜態補景003

1024個獨特視圖、128個相關道路錨點、4096PNG，原train Town01/Town03；完整audit及5頁38情境實際視覺檢視通過。另有先前6個picket全尺寸預覽。真正RGB/GT包含白色尖柵欄、木板斜撐圍欄、金屬網、石牆、草地與道路背景；只有所列例子被人工式看圖，不聲稱全資料材質辨識或2個獨立木板地點。

原007不改、不是新增動態episodes，Town03成功512view逐位元沿用001，Town01完整重收；兩次失敗344/59view及64view probe不供train。捕捉207個不完整候選均被拒絕；所有收下影像同真實F/time，並在F+1後讀回，世界pose最大差4.52e-5。

Wall/Fence/Terrain支持各10162284/2774763/13078119pixels，>=512px視圖611/427/728。raw0缺標與真實錯類分開，resolved最大差8.71e-6，全raw差異及29x29對照仍保存。full audit原始bytes壓在audit.json.gz，可依audit-summary與review SHA還原核對；原圖保留本機data。

v5每輪主動態9600view加1024靜態各重複4次，13696sampledviews，仍是1024獨特靜態views，不把4096採樣算新場景。僅供17類2D segmentation，不能混入detector/phase/heldout/3D；模型品質尚須完整驗證。
