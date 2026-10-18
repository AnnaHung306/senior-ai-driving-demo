# 靜態影像的跨模態標註覆蓋

Town03原始512view全掃描，最差全圖raw差異2.00651%。實際看圖並逐像素查4623個不一致，全部是semantic9植被對instance0未標註，非wall/fence錯類；raw0在原映射就是unlabelled/ignore255。圖中instance raw0顯示黑色，不冒充terrain。

跨模態類別一致性應只在雙方有非零標註的像素比較，仍要求不一致率≤2%，並另外保留全圖raw差異、未解析支持量及29x29原始類別對照。此最差raw影像resolved差異為0；非零錯類仍會計入。獨立CRC、exactF/time、世界pose、矩陣完整、真實QA與主模型逐類門檻均保留。

這不是把2%改成2.1%，也沒有改動semantic target；語意植被仍是已標註other0，不能因instance缺標就抹除semantic。static補景不教物件ID或3D，未知Dynamic仍ignore；完整資料仍須正式audit/review。
