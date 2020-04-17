"""
This program is used for finding the level of return on investment
using kmean++ algorithm
@author: tyeh3
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

movies_meta_data = pd.read_csv('data/movies_meta_data_after_processing.csv')

def create_label_cluster(movies_meta_data): 
    roi_pos = movies_meta_data['return_on_investment'].values.reshape(-1, 1)
    
    # Using the elbow method to find the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 1).fit(roi_pos)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    # plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    # Fitting K-Means to the dataset
    optimal_clusters = 4
    kmeans = KMeans(n_clusters = optimal_clusters, init = 'k-means++', random_state = 1)
    roi_cluster = kmeans.fit_predict(roi_pos)
    
    unique, counts = np.unique(roi_cluster, return_counts=True)
    dict(zip(unique, counts))
    movies_meta_data['return_on_investment_label'] = -1
    movies_meta_data['return_on_investment_label'] = roi_cluster
    movies_meta_data['return_on_investment_label'].value_counts()
    movies_meta_data['return_on_investment_label'] = movies_meta_data['return_on_investment_label'].map({0: 0, 1: 2, 2: 1, 3: 3})
    # movies_meta_data['return_on_investment_label'].value_counts()
    
    plt.scatter(movies_meta_data[movies_meta_data['return_on_investment_label'] == 0]['return_on_investment'], np.zeros(np.count_nonzero(movies_meta_data['return_on_investment_label'] == 0)), s = 10, c = 'red', label = 'Not Profitable')
    plt.scatter(movies_meta_data[movies_meta_data['return_on_investment_label'] == 1]['return_on_investment'], np.zeros(np.count_nonzero(movies_meta_data['return_on_investment_label'] == 1)), s = 10, c = 'yellow', label = 'Slightly Profitable')
    plt.scatter(movies_meta_data[movies_meta_data['return_on_investment_label'] == 2]['return_on_investment'], np.zeros(np.count_nonzero(movies_meta_data['return_on_investment_label'] == 2)), s = 10, c = 'green', label = 'Profitable')
    plt.scatter(movies_meta_data[movies_meta_data['return_on_investment_label'] == 3]['return_on_investment'], np.zeros(np.count_nonzero(movies_meta_data['return_on_investment_label'] == 3)), s = 10, c = 'blue', label = 'Highly Profitable')
    plt.title('Clusters of movies')
    plt.xlabel('Ruturn on investment')
    plt.legend()
    plt.show()
    
    print('Not Profitable')
    print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 0]['return_on_investment'])))
    print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 0]['return_on_investment'])))
    
    print('Slightly Profitable')
    print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 1]['return_on_investment'])))
    print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 1]['return_on_investment'])))
    
    print('Profitable')
    print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 2]['return_on_investment'])))
    print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 2]['return_on_investment'])))
    
    print('Highly Profitable')
    print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 3]['return_on_investment'])))
    print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 3]['return_on_investment'])))
    
    movies_meta_data.to_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv', index=False)

create_label_cluster(movies_meta_data)

'''
def create_label_cluster_custom(movies_meta_data):
    roi_pos_df = movies_meta_data[movies_meta_data['return_on_investment'] > 0]
    roi_pos = movies_meta_data[movies_meta_data['return_on_investment'] > 0]['return_on_investment'].values.reshape(-1, 1)
    
    # Using the elbow method to find the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 1).fit(roi_pos)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    # plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    # Fitting K-Means to the dataset
    optimal_clusters = 3
    kmeans = KMeans(n_clusters = optimal_clusters, init = 'k-means++', random_state = 1)
    roi_cluster = kmeans.fit_predict(roi_pos)
    
    unique, counts = np.unique(roi_cluster, return_counts=True)
    dict(zip(unique, counts))
    movies_meta_data['return_on_investment_label'] = -1
    movies_meta_data['return_on_investment_label'][roi_pos_df.index] = roi_cluster
    movies_meta_data['return_on_investment_label'].value_counts()
    movies_meta_data['return_on_investment_label'] = movies_meta_data['return_on_investment_label'].map({-1: 0, 0: 1, 1: 3, 2: 2})
    movies_meta_data['return_on_investment_label'].value_counts()
    
    plt.scatter(movies_meta_data[movies_meta_data['return_on_investment_label'] == 0]['return_on_investment'], np.zeros(np.count_nonzero(movies_meta_data['return_on_investment_label'] == 0)), s = 10, c = 'red', label = 'Not Profitable')
    plt.scatter(movies_meta_data[movies_meta_data['return_on_investment_label'] == 1]['return_on_investment'], np.zeros(np.count_nonzero(movies_meta_data['return_on_investment_label'] == 1)), s = 10, c = 'yellow', label = 'Slightly Profitable')
    plt.scatter(movies_meta_data[movies_meta_data['return_on_investment_label'] == 2]['return_on_investment'], np.zeros(np.count_nonzero(movies_meta_data['return_on_investment_label'] == 2)), s = 10, c = 'green', label = 'Profitable')
    plt.scatter(movies_meta_data[movies_meta_data['return_on_investment_label'] == 3]['return_on_investment'], np.zeros(np.count_nonzero(movies_meta_data['return_on_investment_label'] == 3)), s = 10, c = 'blue', label = 'Highly Profitable')
    plt.title('Clusters of movies')
    plt.xlabel('Ruturn on investment')
    plt.legend()
    plt.show()
    
    print('Not Profitable')
    print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 0]['return_on_investment'])))
    print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 0]['return_on_investment'])))
    
    print('Slightly Profitable')
    print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 1]['return_on_investment'])))
    print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 1]['return_on_investment'])))
    
    print('Profitable')
    print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 2]['return_on_investment'])))
    print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 2]['return_on_investment'])))
    
    print('Highly Profitable')
    print('min: ' + str(min(movies_meta_data[movies_meta_data['return_on_investment_label'] == 3]['return_on_investment'])))
    print('max: ' + str(max(movies_meta_data[movies_meta_data['return_on_investment_label'] == 3]['return_on_investment'])))
    
    movies_meta_data.to_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv', index=False)

create_label_cluster_custom(movies_meta_data)
'''