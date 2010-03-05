init_random = 42;
kernel_feature_type = 'Real';
kernel_accuracy = 1e-08;
kernel_data_train = [0.5838937, 0.9790694, 0.5719511, 0.9490415, 0.4441329, 0.4358411, 0.5000074, 0.6974255, 0.367616, 0.863886, 0.0292814;0.4397642, 0.5109918, 0.6070322, 0.2760719, 0.42748, 0.4845378, 0.68691, 0.7735442, 0.7378273, 0.3415831, 0.9994505;0.8486653, 0.2600234, 0.1951058, 0.3140213, 0.4523449, 0.8507994, 0.7783623, 0.8512461, 0.2719571, 0.5934612, 0.2832491;0.1090232, 0.6841977, 0.1429976, 0.6676978, 0.6479643, 0.8102484, 0.1395546, 0.7834801, 0.9992765, 0.8510424, 0.7518121;0.6621654, 0.4217815, 0.7305002, 0.724178, 0.6314983, 0.7962784, 0.4487639, 0.1212527, 0.5590677, 0.2650435, 0.3844331;0.0174149, 0.3827906, 0.8088011, 0.5427224, 0.0098005, 0.4698438, 0.2914516, 0.4775507, 0.4046927, 0.9319792, 0.0411773;0.683588, 0.8510266, 0.9592386, 0.1776362, 0.5313048, 0.0435896, 0.4991835, 0.1254581, 0.8378967, 0.11387, 0.5132884;0.5210961, 0.9716505, 0.6549807, 0.2878403, 0.2407993, 0.1282482, 0.7565147, 0.6907567, 0.6098771, 0.0408404, 0.7628289;0.6418876, 0.9638889, 0.5872021, 0.3039519, 0.3815206, 0.1758407, 0.7229142, 0.7077217, 0.20065, 0.1584893, 0.0770049;0.9119757, 0.2330698, 0.3441635, 0.2805045, 0.4136246, 0.7140488, 0.8705956, 0.1825987, 0.6864123, 0.3525621, 0.9805272;0.2989192, 0.007193, 0.51564, 0.1507311, 0.0444944, 0.6982692, 0.7322593, 0.460699, 0.6768855, 0.5382394, 0.6298011];
kernel_matrix_train = [1.02005147, 0.890957458, 0.843084335, 0.613365633, 0.660098901, 0.778535985, 1.02201221, 0.769153443, 0.838217278, 0.574169168, 0.787101883;0.890957458, 1.27783742, 1.0097911, 0.786477689, 0.711592241, 0.666581735, 0.946331462, 0.96558289, 0.957634903, 0.702398123, 0.753642672;0.843084335, 1.0097911, 1.08569871, 0.687783749, 0.599632445, 0.708054808, 0.949093809, 0.785917392, 0.944160959, 0.679002866, 0.756112347;0.613365633, 0.786477689, 0.687783749, 0.715257221, 0.534511999, 0.699690418, 0.638874914, 0.693989534, 0.704761123, 0.709366255, 0.512984894;0.660098901, 0.711592241, 0.599632445, 0.534511999, 0.556931938, 0.611841268, 0.637438252, 0.597759954, 0.699142088, 0.493315186, 0.599013902;0.778535985, 0.666581735, 0.708054808, 0.699690418, 0.611841268, 1.001014, 0.863979065, 0.818105448, 0.893061483, 0.823978973, 0.795680678;1.02201221, 0.946331462, 0.949093809, 0.638874914, 0.637438252, 0.863979065, 1.15363385, 0.928664928, 0.952922426, 0.686287638, 0.922228088;0.769153443, 0.96558289, 0.785917392, 0.693989534, 0.597759954, 0.818105448, 0.928664928, 1.05467759, 0.866402263, 0.805563115, 0.757515161;0.838217278, 0.957634903, 0.944160959, 0.704761123, 0.699142088, 0.893061483, 0.952922426, 0.866402263, 1.15369914, 0.773784656, 1.03114386;0.574169168, 0.702398123, 0.679002866, 0.709366255, 0.493315186, 0.823978973, 0.686287638, 0.805563115, 0.773784656, 0.899852792, 0.567497434;0.787101883, 0.753642672, 0.756112347, 0.512984894, 0.599013902, 0.795680678, 0.922228088, 0.757515161, 1.03114386, 0.567497434, 1.08134585];
kernel_name = 'SparseLinear';
kernel_normalizer = 'AvgDiagKernelNormalizer';
kernel_feature_class = 'simple';
kernel_matrix_test = [0.917522285, 1.12678605, 0.651447336, 0.849047814, 0.662254514, 0.733519482, 0.810208951, 0.709643044, 0.660466489, 0.677644667, 0.716246907, 0.767879012, 0.40293166, 0.994471339, 0.818906341, 0.616367553, 0.631889536;1.09643358, 1.29824462, 0.609105088, 0.81847631, 0.884530779, 1.0040578, 1.1060583, 0.781038691, 0.664138942, 0.524286745, 0.917472318, 0.75249617, 0.445579104, 0.824028269, 1.03383954, 0.6267669, 0.741397732;0.985379639, 1.25067163, 0.605749737, 0.930365388, 0.817623678, 1.03020409, 0.950810406, 0.808478174, 0.666523357, 0.53352811, 0.945984784, 0.645239785, 0.601403494, 0.776237263, 0.986191057, 0.569640086, 0.632080839;0.780020454, 0.972513489, 0.563365734, 0.637690317, 0.676509873, 0.690683598, 0.653728183, 0.655798126, 0.370975238, 0.461053617, 0.704157676, 0.59144331, 0.401265862, 0.652014504, 0.766021002, 0.483085709, 0.536832297;0.687203699, 0.874597968, 0.545702852, 0.559596547, 0.539829208, 0.620903277, 0.5964182, 0.643853222, 0.435845248, 0.444722408, 0.593857231, 0.567487241, 0.265121232, 0.686718678, 0.665159187, 0.381015632, 0.431155429;0.981170327, 1.08481594, 0.750397242, 0.84478506, 0.765216313, 0.79088029, 0.680803299, 0.842534688, 0.539837258, 0.648660487, 0.836015471, 0.777169149, 0.453060018, 0.878666724, 0.881742372, 0.492546382, 0.535865468;1.12679492, 1.22272297, 0.682886841, 1.00644328, 0.835410022, 0.911644159, 0.99792005, 0.775593279, 0.776802244, 0.692159832, 0.862696497, 0.865180517, 0.503298533, 0.992623677, 0.952025268, 0.670662514, 0.726759356;1.17500785, 1.13219689, 0.592859237, 0.852540969, 0.909613962, 0.936517293, 0.963099897, 0.770689826, 0.617155203, 0.535423853, 0.847562223, 0.809487054, 0.397471367, 0.812742281, 0.956933999, 0.513356997, 0.746190129;1.0734495, 1.25910579, 0.742450483, 0.849114941, 0.853237578, 1.10266367, 0.961841183, 0.975140643, 0.731825258, 0.589096343, 0.998622055, 0.818043328, 0.495756713, 0.857807574, 1.01378008, 0.565160473, 0.56112637;0.921533135, 0.992145162, 0.554503008, 0.747354547, 0.71652247, 0.807162545, 0.650940129, 0.713183895, 0.426744946, 0.498044144, 0.798217574, 0.686387188, 0.458764371, 0.689793732, 0.815716242, 0.460414683, 0.519497088;0.924832871, 1.0701706, 0.682572101, 0.776674637, 0.762419245, 0.925560497, 0.830048295, 0.822377382, 0.704913821, 0.573590172, 0.797820365, 0.803603553, 0.330918313, 0.78962515, 0.801460265, 0.549646514, 0.519500032];
kernel_data_test = [0.596704, 0.6658228, 0.1996306, 0.2687066, 0.4399631, 0.446128, 0.7744117, 0.3673937, 0.1625052, 0.251466, 0.2147335, 0.73166, 0.4703396, 0.7086307, 0.4162197, 0.8897356, 0.9835203;0.6698425, 0.8638126, 0.5079333, 0.9346208, 0.8315865, 0.9917786, 0.6230937, 0.7792067, 0.5428681, 0.2591201, 0.3547552, 0.8940054, 0.0559479, 0.5914799, 0.372484, 0.3866925, 0.8845712;0.8538224, 0.5988424, 0.2729188, 0.4754563, 0.4082934, 0.353618, 0.1027856, 0.5431139, 0.3018313, 0.6608079, 0.3192039, 0.4311635, 0.0523604, 0.9925381, 0.5076104, 0.0242376, 0.5412307;0.870354, 0.7725521, 0.7009067, 0.1394647, 0.6577371, 0.7662984, 0.5724191, 0.892749, 0.2769898, 0.2421447, 0.8695697, 0.6253982, 0.0014472, 0.3531314, 0.857856, 0.0273195, 0.0032479;0.2812593, 0.914, 0.9014776, 0.5187619, 0.5532369, 0.332008, 0.2437367, 0.8152219, 0.1062949, 0.4873544, 0.6343973, 0.1036803, 0.5196071, 0.5775213, 0.7424961, 0.234836, 0.2933275;0.5695211, 0.9831264, 0.2174932, 0.9859038, 0.7143425, 0.8260821, 0.0896413, 0.3004202, 0.0577363, 0.371547, 0.9812897, 0.1052731, 0.5623728, 0.0060706, 0.7126828, 0.1971927, 0.1295845;0.3370178, 0.7710443, 0.1066131, 0.229839, 0.0084644, 0.9684149, 0.5921456, 0.7198996, 0.6513832, 0.1443646, 0.5949399, 0.1314156, 0.4012365, 0.5689631, 0.5588811, 0.1190734, 0.034436;0.7982343, 0.7259224, 0.0818189, 0.1274155, 0.8925838, 0.7691289, 0.9961603, 0.2539874, 0.487803, 0.2364596, 0.6228001, 0.1154142, 0.1272432, 0.0718879, 0.6935861, 0.4940726, 0.6177312;0.879301, 0.7403285, 0.4617229, 0.984356, 0.4562561, 0.1124507, 0.9926547, 0.0228207, 0.5498836, 0.3090955, 0.5035522, 0.558713, 0.2400959, 0.5568588, 0.7560611, 0.2665098, 0.3834261;0.2682427, 0.8205597, 0.7249823, 0.8711964, 0.1700499, 0.0735796, 0.1193835, 0.1610183, 0.5330827, 0.9709121, 0.3572354, 0.9585083, 0.0382008, 0.9268722, 0.1811526, 0.912475, 0.0562274;0.9482194, 0.2007983, 0.3027141, 0.4909423, 0.3807209, 0.667528, 0.9748078, 0.6064265, 0.7214854, 0.0880053, 0.6278124, 0.4538498, 0.8857824, 0.3882111, 0.6794541, 0.2281537, 0.2593725];