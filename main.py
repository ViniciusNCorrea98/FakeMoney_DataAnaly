import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv('banknote-dataset.csv')
dataset = pd.DataFrame(dataset)
dataset.sort_values('V1', inplace=True)

print('--------------Dataset------------------')
print(dataset)

print('--------------Means------------------')
mean_V1 = np.mean(dataset['V1'])
print(mean_V1)
mean_V2 = np.mean(dataset['V2'])
print(mean_V2)


print('--------------standard deviation------------------')
std_dev_V1 = np.std(dataset['V1'])
std_dev_V2 = np.std(dataset['V2'])
print(std_dev_V1)
print(std_dev_V2)

plt.xlabel('Variance of Wavelet')
plt.ylabel('Skewness of Wavelet')
plt.scatter(dataset['V1'], dataset['V2'], s=20, alpha=0.9)

v1_v2 = np.column_stack((dataset['V1'], dataset['V2']))
km_results = KMeans(n_clusters=3).fit(v1_v2)
clusters = km_results.cluster_centers_
plt.scatter(clusters[:,0], clusters[:,1], s=1000, alpha=0.4)
plt.show()