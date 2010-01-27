f = open('label_train_twoclass.dat')
labels=f.readline().split(' ')

f=open('fm_test_sparsereal.dat')
out=open('test.dat','w')
i=0
for line in f:
	out.write(str(labels[i]) + ' ' + line)
	i+=1


