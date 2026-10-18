# 靜態補景採集重試修正

Town03完整512view保留；Town01第一版失敗於5次連續同偶數F，實際只收future F+1。兩tick重試保留奇偶性，不能視為5個不同排程機會。修正為拒絕整組後另前進一tick，再重新要求四模態exactF/time/pose；不接受future重標。單元測試明確驗證101失敗後104成功、奇偶不同及有界失敗。

root002只逐位元複製完整Town03，再重收整個Town01，失敗344view不供train。post-capture source archive是實際未變動來源的事後存檔，非捏造採集當下寫檔。Town01新collector會在採集前寫完整來源快照。仍須全部1024view/4096PNG獨立audit與實際QA，不能將複製當成新增場景。


後續實測否定奇偶假說足以解釋問題：002僅59view後仍失敗。改逐幀保留future後，未paced probe仍有semantic/depth/instance輪流缺F。加入每tick .05s牆鐘節奏，64view/256PNG原生短測完成，獨立CRC/frame/time/pose/alignment檢查passed_probe_only；43個不完整候選被丟棄。max pose error4.37784e-5，semantic-instance偏差最大1.7096%<2%。不能據此保證所有view；完整採集每view最多24個真實候選，仍exactF並只在F+1後接收，無common frame會有界失敗。完整003只沿用512view Town03，另重收512view Town01；probe及失敗344/59view不進train。
