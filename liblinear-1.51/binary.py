f = open('out.txt')
fo = open('out.b.txt','w')
lst = f.readlines()
for item in lst:
	if float(item.rstrip('\n')) >= 0.0:
		fo.write("-1\n")
	else:
		fo.write("1\n")
