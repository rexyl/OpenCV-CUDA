with open('uspsdata/uspscl.txt', 'r') as content_file:
    content = content_file.read()
f = open('uspsdata/uspscl_ext.txt','w')
for i in range(0,300):
	f.write(content)
f.close()


with open('uspsdata/uspsdata.txt', 'r') as content_file:
    content = content_file.read()
f = open('uspsdata/uspsdata_ext.txt','w')
for i in range(0,300):
	f.write(content)
f.close()