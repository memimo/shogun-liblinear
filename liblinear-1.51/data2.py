f = open('australian_scale')
out_data=open('australian_scale.dat','w')
out_label=open('australian_scale.label', 'w')
i=0
for line in f.readlines():
	item = line.split()
	out_label.write(item[0] + "\n")
	del item[0]
	out_data.write(" ".join(item) + "\n")
	i+=1


