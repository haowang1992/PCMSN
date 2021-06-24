from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import math
 
cm = plt.cm.get_cmap('RdYlBu')
a = np.load("/home/xxx/AAAI/My-ZSSBIR/tsne/danbu_Berlin_ske_em_tsne.npy")
X_tsne = TSNE(learning_rate=100).fit_transform(a)
Y = []
for i in range(800):
	Y.append(math.ceil(i/80))
hh = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, vmin=0, vmax=100, s=35, cmap=cm)
plt.colorbar(hh)
plt.show() 