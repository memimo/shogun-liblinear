library("sg")

size_cache <- 10

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))

# Linear
print('Linear')

scale <- 1.2
dump <- sg('set_kernel', 'LINEAR', 'REAL', size_cache, scale)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')
