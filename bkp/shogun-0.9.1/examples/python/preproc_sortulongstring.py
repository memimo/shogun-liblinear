def sort_ulong_string ():
	print 'CommUlongString'

	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	from sg import sg
	sg('add_preproc', 'SORTULONGSTRING')
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')

	sg('set_kernel', 'COMMSTRING', 'ULONG', size_cache, use_sign, normalization)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	sort_ulong_string()
