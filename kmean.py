"""
This program is used for finding the level of return on investment
using kmean++ algorithm
@author: tyeh3
"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

movies_meta_data = pd.read_csv("movies_meta_data_after_preprocessing.csv")
roi_pos_df = movies_meta_data[movies_meta_data['return_on_investment'] > 0]
roi_pos = movies_meta_data[movies_meta_data['return_on_investment'] > 0]['return_on_investment'].values.reshape(-1, 1)

# Using the elbow method to find the optimal number of clusters
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 1).fit(roi_pos)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# Fitting K-Means to the dataset
optimal_clusters = 3
kmeans = KMeans(n_clusters = optimal_clusters, init = 'k-means++', random_state = 1)
roi_cluster = kmeans.fit_predict(roi_pos)

# Include the values with negative ROI in the graph
# plt.scatter(roi_pos[roi_cluster == 0, 0], np.zeros(np.count_nonzero(roi_cluster == 0)), s = 10, c = 'red', label = 'Very Low')
# plt.scatter(roi_pos[roi_cluster == 1, 0], np.zeros(np.count_nonzero(roi_cluster == 1)), s = 10, c = 'yellow', label = 'Low')
# plt.scatter(roi_pos[roi_cluster == 2, 0], np.zeros(np.count_nonzero(roi_cluster == 2)), s = 10, c = 'green', label = 'Medium')
# plt.scatter(roi_pos[roi_cluster == 3, 0], np.zeros(np.count_nonzero(roi_cluster == 3)), s = 10, c = 'blue', label = 'High')
# plt.scatter(kmeans.cluster_centers_[:, 0], np.zeros(optimal_clusters), s = 50, c = 'black', label = 'Centroids')
# plt.title('Clusters of movies')
# plt.xlabel('Ruturn on investment')
# plt.legend()
# plt.show()

movies_meta_data['return_on_investment_label'] = -1
movies_meta_data['return_on_investment_label'][roi_pos_df.index] = roi_cluster
# movies_meta_data['return_on_investment_label'] = movies_meta_data['return_on_investment_label'].map({-1: 0, 0: 1, 1: 2, 2: 3, 3: 4})
movies_meta_data['return_on_investment_label'] = movies_meta_data['return_on_investment_label'].map({-1: 0, 0: 1, 1: 2, 2: 3})

#print level ranges
# print()
# print('level0')
# print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 0]['return_on_investment'])))
# print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 0]['return_on_investment'])))

# print('level1')
# print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 1]['return_on_investment'])))
# print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 1]['return_on_investment'])))

# print('level2')
# print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 2]['return_on_investment'])))
# print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 2]['return_on_investment'])))

# print('level3')
# print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 3]['return_on_investment'])))
# print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 3]['return_on_investment'])))

# print('level4')
# print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 4]['return_on_investment'])))
# print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 4]['return_on_investment'])))

