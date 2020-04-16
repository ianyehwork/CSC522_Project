"""
Unfinish
@author: gwang25
"""
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(42)
#Using 8 features to clusters
movies_meta_data_after_processing = pd.read_csv('data/movies_meta_data_after_processing.csv')
movies_meta_data_after_processing.columns
kmeans_attributes = movies_meta_data_after_processing[['genres_popularity_score', 'genres_vote_score', 'keywords_popularity_score', 'keywords_vote_score', 'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score', 'directors_vote_score']]

#The statistics of the 8 features
desc_all = kmeans_attributes.describe(include='all')
for d in desc_all:
    print()
    print("{}:\nmean={:.3f}\nstd.dev={:.3f}\nmin={:.3f}\n25%={:.3f}\n50%={:.3f}\n75%={:.3f}\nmax={:.3f}".format(d, desc_all[d]['mean'], desc_all[d]['std'], desc_all[d]['min'], desc_all[d]['25%'], desc_all[d]['50%'], desc_all[d]['75%'], desc_all[d]['max']))
del(d)
# kmeans_attributes_scaled
# scaled_kmeans_attributes = scale(kmeans_attributes)
# n_samples, n_features = scaled_kmeans_attributes.shape