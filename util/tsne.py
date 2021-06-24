from sklearn.manifold import TSNE 
from pandas.core.frame import DataFrame
import pandas as pd  
import numpy as np 
 
import km as k 
#用TSNE进行数据降维并展示聚类结果
 
tsne = TSNE()
tsne.fit_transform(k.data_zs) #进行数据降维,并返回结果
tsne = pd.DataFrame(tsne.embedding_, index = k.data_zs.index) #转换数据格式
 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
 
#不同类别用不同颜色和样式绘图
d = tsne[k.r[u'聚类类别']== 0]     #找出聚类类别为0的数据对应的降维结果
plt.plot(d[0], d[1], 'r.')
d = tsne[k.r[u'聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')
#d = tsne[k.r[u'聚类类别'] == 2]
#plt.plot(d[0], d[1], 'b*')
plt.savefig("data.png")
plt.show()
a = np.load("/home/xxx/AAAI/My-ZSSBIR/berlin_acc_im_em.npy")
b = np.load("/home/xxx/AAAI/My-ZSSBIR/berlin_acc_cls_im.npy")
