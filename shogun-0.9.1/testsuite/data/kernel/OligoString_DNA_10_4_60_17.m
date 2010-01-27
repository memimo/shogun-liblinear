kernel_alphabet = 'DNA';
kernel_data_train = {'CCCATATAAGGCGCGGCTGCTTGATAACGTGTAAGTGAAACATGGAAGTGTTAAGCCCCT', 'ACGGGTTGACGTTGGAAAACCGGGCCTATCGAACGCGCCTTGAATATCTCCGTTGTGTTC', 'GGGTCACACTGTATGAGTAGTAGCCGATACGCCTTTGTTCCACTGGAACTCTAGGTAAAT', 'AAGGTTGTGCTGCGCAGTGATACCACGTTTCGTCTCCGGGCGTTAAGCACTTCCGAGACC', 'GGCACGAAGTCCTTCTGTCTTCTCACTGGGACGTACAACGAATTCGCTGTAGGCCGGAGA', 'AGCCCGCTCGCGGCCCAAGTCCAATGACTACGGCTAGAGTCGTAGTGAACAGAACCGGAA', 'ACAGATCCTGGTCTCTCCGAACGTGTGAAGCACGCTGGTCGTGTCGATTTGAATCCGCGC', 'GTACCTCCGCTTGGAGGCATCGACTCGACTGTTTTATAAACGCACCTCACCGCCCCTATG', 'ATTGCGCATTATGACCTTAGTTAGTCCTATCAAAAGATGCCGCATCAGGCTGGCTCGCGA', 'ATCCGGAATGCTGAACTAATAGAGCAGGGGGGGGGGAACTTGCATTATCCGGTACCCGCC', 'CGGGACAACAGTATAGGTACACTTGGTCATCAAGGCGTTGTTCCCACCAACGTGCCAATC'};
kernel_arg0_size = 10;
init_random = 42;
kernel_feature_type = 'Char';
kernel_accuracy = 1e-09;
kernel_arg1_k = 4;
kernel_matrix_train = [1, 3.14059559e-06, 0.0255638799, 0.0169461979, 6.91395626e-05, 0.00311680482, 2.72642172e-07, 0.0198812947, 7.59564446e-10, 6.23328272e-08, 0.00439567516;3.14059559e-06, 1, 0.00445807713, 0.0256036868, 0.0043987449, 3.07079674e-06, 0.012361058, 0.0248933178, 0.0372320727, 0.0386866501, 0.0124122412;0.0255638799, 0.00445807713, 1, 2.27562234e-09, 0.00438890828, 0.0247858781, 0.000774593156, 6.81627956e-08, 0.0263578294, 0.000710335329, 0.0206976518;0.0169461979, 0.0256036868, 2.27562234e-09, 1, 0.00439574266, 1.64677042e-17, 0.00444530378, 7.22094824e-05, 1.26929414e-11, 0.0227287898, 9.28068954e-06;6.91395626e-05, 0.0043987449, 0.00438890828, 0.00439574266, 1, 0.024824407, 6.89061553e-05, 0.00517481448, 0.000917292532, 2.86609163e-06, 6.91395626e-05;0.00311680482, 3.07079674e-06, 0.0247858781, 1.64677042e-17, 0.024824407, 1, 0.0248535997, 0.0140426432, 6.14032041e-06, 6.233261e-08, 2.70563777e-36;2.72642172e-07, 0.012361058, 0.000774593156, 0.00444530378, 6.89061553e-05, 0.0248535997, 1, 3.05735403e-06, 0.0174635723, 0.011346407, 3.05814377e-06;0.0198812947, 0.0248933178, 6.81627956e-08, 7.22094824e-05, 0.00517481448, 0.0140426432, 3.05735403e-06, 1, 0.0175400064, 6.32508406e-05, 0.00517481449;7.59564446e-10, 0.0372320727, 0.0263578294, 1.26929414e-11, 0.000917292532, 6.14032041e-06, 0.0174635723, 0.0175400064, 1, 0.0160148647, 0.0350804437;6.23328272e-08, 0.0386866501, 0.000710335329, 0.0227287898, 2.86609163e-06, 6.233261e-08, 0.011346407, 6.32508406e-05, 0.0160148647, 1, 6.23328056e-08;0.00439567516, 0.0124122412, 0.0206976518, 9.28068954e-06, 6.91395626e-05, 2.70563777e-36, 3.05814377e-06, 0.00517481449, 0.0350804437, 6.23328056e-08, 1];
kernel_name = 'OligoString';
kernel_seqlen = 60;
kernel_arg2_width = 1.7;
kernel_feature_class = 'string';
kernel_matrix_test = [0.000917484305, 0.000779203652, 0.0219209439, 0.00957055245, 0.0130293618, 0.049714739, 0.00517501424, 0.0175438597, 0.0518956365, 0.0383223828, 0.0220550034, 0.000772809236, 0.00155840882, 0.00439567516, 7.59681304e-10, 0.00439572917, 0.0177445985;0.0140400611, 3.13983591e-06, 0.035265297, 0.0183230621, 0.012261056, 1.22823848e-05, 3.07080618e-06, 0.0044080256, 0.017725398, 0.0216721812, 0.00155847536, 0.00434210918, 7.22103644e-05, 0.00233761156, 0.0299540287, 0.0175827107, 0.00873281334;1.64421563e-17, 4.22527837e-12, 3.20116179e-06, 0.0299095268, 2.08346676e-27, 1.22632935e-05, 0.024785886, 0.0299785588, 0.000777992166, 6.82589359e-05, 4.23705907e-12, 7.49297336e-10, 3.06603218e-06, 3.06679072e-06, 6.81673823e-08, 0.0219114878, 0.0123850636;0.000138279125, 6.94816756e-05, 4.2400615e-12, 0.0175439279, 0.0347284435, 0.0175426476, 8.04578584e-24, 0.00963969201, 0.0292201586, 0.00160771461, 7.52849694e-05, 0.00875554938, 0.0248244842, 0.0372398653, 0.00440465057, 0.00155840252, 4.20364824e-12;1.1504331e-20, 0.0175871209, 6.89710314e-08, 0.00077934019, 0.00013659408, 0.00439850937, 6.9789216e-08, 0.0140397889, 0.0423683445, 6.74378649e-08, 4.23187412e-12, 0.0260142937, 0.0475091737, 0.0300024345, 0.0227171671, 0.00879439259, 0.0369885049;0.0043956606, 0.0547804837, 0.0175289391, 0.0124122031, 0.0346600377, 2.35969714e-14, 0.00446787043, 3.13907497e-06, 0.013966183, 0.00308489074, 0.0527037568, 3.1008144e-06, 0.000782273526, 0.0183921451, 0.00517464059, 7.63949091e-10, 3.01862112e-09;0.00515226418, 1.14580646e-20, 4.20979492e-12, 0.0059969002, 0.000766344838, 0.00437636846, 0.0263364356, 0.0306027065, 6.88374317e-05, 6.7999338e-05, 0.0247160691, 2.07759233e-27, 3.05738738e-06, 0.0176479404, 0.00155155787, 0.00162043113, 0.0304428282;6.92077421e-05, 3.13903781e-06, 0.0307090231, 0.0263826357, 0.0260140401, 0.0300227968, 0.0043956201, 0.0139721976, 0.00595400857, 0.0245277428, 0.043262484, 0.0173419938, 0.0351564285, 6.82689384e-08, 6.82642188e-08, 8.46361889e-12, 0.00437241965;0.00098641776, 6.91942458e-05, 0.0177758455, 6.91259859e-05, 0.0122614798, 0.0131833774, 0.0131887095, 0.0307721846, 0.0206563902, 0.00434120912, 0.00956856806, 0.0173264798, 0.0248884619, 0.000779042093, 1.36501805e-07, 6.91272928e-05, 0.000776967219;0.0339985566, 6.23328056e-08, 0.0161391658, 0.00802682329, 6.3508122e-28, 6.31225878e-05, 0.0194227824, 0.000714369884, 0.0555217094, 0.0113195092, 8.41196958e-06, 0.000702771958, 0.0113356359, 6.93653993e-10, 0.0226642829, 0.0339983905, 0.0166808141;3.07080661e-06, 2.70564627e-36, 1.51815653e-09, 0.0124122444, 0.0216721827, 0.00440151248, 8.46373178e-12, 0.0416355397, 0.0124123103, 0.0044134399, 0.017612998, 0.013170392, 3.13907639e-06, 0.0744734525, 0.012411384, 2.27915893e-09, 1.17207179e-14];
kernel_data_test = {'TGAGCCGTTTAAACCGGTTATCCTATGTTGAACATCTGACCCGAGCTTAAGTCCACCCGC', 'ACTCTGCAGGGTGATGCGGACCCAAACTACCTAAATGACAATCGCGCCGAGTATACGGAT', 'TATGTATATGCATGCCTCATCCATAGTATCGCGCATATACTATCCGCCGCATTCGGGCTA', 'GTGTCCTCGATGCCGAGGATCCCGTACATCGCTGTAGTGATCACGTCCGCTTCATAAAAT', 'GTACTTGATCGGGGGACGCCACTCGGATCTAACCTAGATTCAGAAGTTGGTGCTAAGACC', 'GTACAGGCAGGACCTTGTGTACGGACCCCGTGTTCCATAGCTCTCCGTGTTTCCATTGCA', 'TATCCTACTGTTAACCCTAGAACTAGGAGTGAGGAAAACCCTGTGACTAACCCACGCGGG', 'AGACGACAACTCCGGTTTTAGTGTACTCGAGTCAATGAATTACTGTCGGTCACTCCGAAC', 'GGTTCGAAACACGCGCAAAAGTCCTAAGGGGACATTCACACGTCAAAATATGGCGCCCTC', 'CCAAAGCTGAGGGGGAGCGATCGTTGATGGTAAGTCGCGTAGTCCATCCCCGTCGGCATG', 'GATGTTTTATATCGATACTGACAGTGACTAGCTTCGGCCGATAACACCGTTATCCCCTCG', 'ATTCGGGGGAGCCTACAACTCGAGTCTTGCGTACCCTCCGAAGCCAGTTAGTCTTACAAT', 'TAGGCGTAAACCCGTCCTTACTACCAACTTTAATGCACCATATTCGGACGGGCCCCGTGG', 'GGGATACAATCTCCCGTCCTACCACATGTGGGTACGTTGAATCATAGGACAGCATCCAAT', 'CCTGCTGAGTCGTGAAGCTCCGGCAGTGCATTCGCGTAGACGAGCGAGGTTCAGGAAGTT', 'ACACTTCACTACGGTCGCTTTATCAAGCTGAGACGTCTTCGGATTATATCTCGACATCGT', 'TATAGGAACCTCTCTGTAGACGGAGGAGGATGTATACATGAACAGGTTGACGCGTTACTT'};
