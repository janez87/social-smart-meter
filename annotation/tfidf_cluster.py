from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import AffinityPropagation

import numpy as np
import matplotlib.pyplot as plt

with open('out3.txt', 'r') as f:
    documents = list(f.readlines())
 
 
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
X = vectorizer.fit_transform(documents)

print(X.toarray())

distortions = []
silhouette = []

"""
K = range(2, 21)
for k in K:
    print("Clustering with k = "+str(k))
    kmeanModel = KMeans(n_clusters=k,init='k-means++').fit(X.toarray())
    kmeanModel.fit(X)
    print("Computing the errors")
    distortions.append(sum(np.min(
        cdist(X.toarray(), kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    print("Computing the average silouette score ")
    silhouette.append(silhouette_score(
        X.toarray(), kmeanModel.labels_, metric='euclidean'))
    

print(silhouette)

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

kMeansVar = [KMeans(n_clusters=k).fit(X) for k in range(1, 10)]
centroids = [X.cluster_centers_ for X in kMeansVar]

print(centroids)

k_euclid = [cdist(X, cent) for cent in centroids ]
dist = [np.min(ke, axis=1) for ke in k_euclid]
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(X.toarray())**2) / X.shape[0]
bss = tss - wcss



# Plot the elbow
plt.plot(bss)
plt.xlabel('k')
plt.ylabel('Betweennes')
plt.title('The Elbow Method showing the optimal k')
plt.show()
"""


