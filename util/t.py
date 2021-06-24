import numpy as np
a = np.load("/home/xxx/AAAI/My-ZSSBIR/sketchy_acc_im_em.npy")
b = np.load("/home/xxx/AAAI/My-ZSSBIR/sketchy_acc_cls_im.npy")
num = 0
tmp = b[0]
data=[]
ind = [0]
for index, i in enumerate(b):
	if(i==tmp):
		num = num + 1
	else:
		ind.append(index)
		data.append(num)
		num = 0
		tmp = i
print(data)
print(ind)
aa = a[0:80]
for inn, ii in enumerate(ind):
	if(inn>0 and inn<10):
		aa = np.concatenate((aa, a[ii:ii+80]), axis = 0)
print(aa.shape)		
np.save("/home/xxx/AAAI/My-ZSSBIR/tsne/sketchy_img_em_tsne.npy",aa)