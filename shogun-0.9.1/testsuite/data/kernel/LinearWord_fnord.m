init_random = 42;
kernel_feature_type = 'Word';
kernel_accuracy = 1e-08;
kernel_data_train = [15, 32, 26, 0, 15, 4, 17, 40, 33, 24, 41;33, 14, 3, 3, 24, 40, 12, 41, 24, 2, 34;8, 13, 21, 8, 23, 22, 36, 14, 35, 13, 19;38, 35, 26, 16, 25, 40, 3, 2, 18, 40, 39;38, 6, 35, 18, 11, 19, 37, 11, 28, 29, 19;3, 1, 29, 22, 5, 17, 21, 36, 3, 10, 40;34, 10, 33, 8, 30, 41, 31, 34, 10, 16, 35;1, 23, 14, 27, 2, 28, 12, 21, 30, 6, 25;30, 37, 37, 3, 34, 32, 2, 18, 8, 18, 18;25, 12, 21, 7, 23, 25, 40, 40, 2, 6, 1;32, 11, 11, 9, 40, 26, 17, 3, 39, 22, 20];
kernel_matrix_train = [1.20709758, 0.715698002, 0.935459033, 0.358907726, 0.98022797, 1.16278238, 0.78527135, 0.828830315, 0.791321206, 0.775289087, 1.00321742;0.715698002, 0.743224848, 0.709345653, 0.276932173, 0.662610513, 0.805992108, 0.431808495, 0.636747377, 0.63326871, 0.578820003, 0.813251935;0.935459033, 0.709345653, 1.07142955, 0.432564727, 0.805235876, 1.00185621, 0.833065215, 0.891295082, 0.713429306, 0.74019992, 1.03119801;0.358907726, 0.276932173, 0.432564727, 0.312928818, 0.284494493, 0.490189608, 0.37977973, 0.371612424, 0.376452309, 0.313987543, 0.49850816;0.98022797, 0.662610513, 0.805235876, 0.284494493, 0.951339906, 1.02983679, 0.692708548, 0.751845894, 0.739141195, 0.63493242, 0.875262963;1.16278238, 0.805992108, 1.00185621, 0.490189608, 1.02983679, 1.38239217, 0.851971016, 1.00503238, 0.867398149, 0.741863631, 1.17957074;0.78527135, 0.431808495, 0.833065215, 0.37977973, 0.692708548, 0.851971016, 0.990058986, 0.882976529, 0.709496899, 0.532538602, 0.794043641;0.828830315, 0.636747377, 0.891295082, 0.371612424, 0.751845894, 1.00503238, 0.882976529, 1.27470473, 0.689229881, 0.496541957, 1.08383176;0.791321206, 0.63326871, 0.713429306, 0.376452309, 0.739141195, 0.867398149, 0.709496899, 0.689229881, 0.997621306, 0.636898624, 0.939845179;0.775289087, 0.578820003, 0.74019992, 0.313987543, 0.63493242, 0.741863631, 0.532538602, 0.496541957, 0.636898624, 0.669416601, 0.800093498;1.00321742, 0.813251935, 1.03119801, 0.49850816, 0.875262963, 1.17957074, 0.794043641, 1.08383176, 0.939845179, 0.800093498, 1.39978551];
kernel_name = 'LinearWord';
kernel_normalizer = 'AvgDiagKernelNormalizer';
kernel_feature_class = 'simple';
kernel_matrix_test = [1.07218579, 0.907932187, 0.855903422, 0.864978207, 0.659131846, 0.558250492, 1.11892093, 0.606498096, 0.810227007, 0.673197762, 0.616934098, 0.738687456, 0.501533089, 0.825502894, 0.845921159, 0.73475505, 0.684994981;0.816276863, 0.672895269, 0.652930743, 0.737779978, 0.740956152, 0.514842772, 0.688171156, 0.314290036, 0.516960222, 0.45978908, 0.403979155, 0.566871537, 0.401105474, 0.460242819, 0.69724594, 0.635234913, 0.690742345;1.073547, 0.810983239, 0.943021353, 0.915343261, 0.860894554, 0.534807298, 1.02166949, 0.70148084, 0.756988272, 0.768785491, 0.588499773, 0.776196565, 0.546302025, 0.746854762, 0.87072557, 0.701329593, 0.765609317;0.552805621, 0.320491138, 0.469166357, 0.345598042, 0.412146462, 0.344236824, 0.435589655, 0.310055136, 0.294476756, 0.322457342, 0.364806336, 0.423641189, 0.391274457, 0.367377525, 0.451924267, 0.436950872, 0.254396458;1.00382241, 0.856659654, 0.708589421, 0.740351167, 0.69558223, 0.419406289, 1.01803957, 0.532841095, 0.727192729, 0.594852122, 0.585626091, 0.759408214, 0.503953031, 0.663971731, 0.862709511, 0.573980118, 0.637654856;1.26502496, 0.934097815, 1.01970328, 0.938786454, 0.948163731, 0.695128491, 1.15884998, 0.69180107, 0.788750017, 0.769087984, 0.756685779, 0.941357643, 0.808563297, 0.895227488, 1.12300458, 0.861197047, 0.741712384;1.08050434, 0.702085825, 0.80296718, 0.693616027, 0.627521346, 0.400349242, 1.08383176, 0.756080793, 0.893110039, 0.675012718, 0.560216695, 0.775742826, 0.605136878, 0.783456393, 0.781641436, 0.563846609, 0.516657729;1.24218675, 0.823234198, 0.887513922, 0.780431465, 0.737779978, 0.478241142, 1.11468603, 0.766819288, 0.863012003, 0.724016555, 0.610732995, 0.920334392, 0.788598771, 0.950432427, 0.939996425, 0.511666598, 0.603926907;1.10576249, 0.697850926, 0.836543882, 0.587289802, 0.69346478, 0.47325001, 1.08746167, 0.501684335, 0.82777159, 0.58275241, 0.535412284, 0.93727399, 0.608918038, 0.726285251, 0.818696806, 0.773776623, 0.648695843;0.765609317, 0.622530215, 0.662761759, 0.638562334, 0.601355718, 0.449958063, 0.783758886, 0.409575273, 0.576097568, 0.51408654, 0.476728678, 0.533446081, 0.330473401, 0.515296512, 0.559914202, 0.624798911, 0.620110272;1.29950914, 0.847433623, 1.09804892, 0.807655818, 0.949222456, 0.626462622, 1.23961556, 0.731276382, 0.842291246, 0.844408695, 0.769087984, 1.06008607, 0.833065215, 0.982496666, 0.996865074, 0.811588225, 0.8379051];
kernel_data_test = [35, 29, 20, 22, 20, 7, 38, 12, 39, 21, 6, 23, 4, 25, 9, 10, 39;37, 10, 29, 3, 1, 12, 38, 16, 22, 9, 12, 41, 38, 41, 28, 15, 2;35, 3, 35, 15, 40, 5, 41, 27, 35, 22, 11, 41, 30, 11, 31, 24, 28;18, 21, 23, 32, 35, 39, 15, 1, 7, 12, 29, 0, 11, 6, 13, 41, 34;27, 11, 39, 23, 0, 16, 37, 27, 32, 23, 5, 20, 3, 32, 9, 35, 12;35, 8, 30, 3, 19, 7, 26, 35, 8, 23, 36, 40, 39, 29, 25, 3, 0;19, 27, 23, 27, 27, 4, 31, 29, 20, 41, 15, 7, 11, 31, 24, 2, 25;35, 16, 26, 21, 37, 34, 6, 2, 5, 12, 14, 25, 32, 14, 38, 40, 13;18, 27, 23, 40, 32, 11, 9, 6, 0, 14, 0, 19, 1, 2, 35, 15, 28;35, 38, 2, 36, 11, 19, 22, 21, 31, 6, 21, 1, 13, 15, 23, 6, 3;36, 40, 4, 1, 4, 2, 41, 7, 25, 9, 33, 35, 11, 19, 28, 13, 4];
