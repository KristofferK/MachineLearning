from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
 
k = 2
X, y = make_moons(n_samples=1000, noise=0.1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_
clusters = kmeans.cluster_centers_
 
print(clusters)
 
cmap_bold = [ListedColormap(['#FF0000', '#0000FF'])]
                             
pyplot.scatter(X[:, 0], X[:, 1], c=labels, edgecolor='black', cmap=cmap_bold[0], s=20)