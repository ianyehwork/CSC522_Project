#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tyeh3
@author: wchu2
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def decision_tree(x_train, y_training, x_test, y_testing):
    # Decision Tree
    dt = DecisionTreeClassifier(criterion='gini', random_state=0)
    dt.fit(x_train, y_training)
    dt_y_pred = dt.predict(x_test)

    # Print accuracy and SWK
    print(" Decision Tree")
    print("     Accuracy: " + str(accuracy_score(y_testing, dt_y_pred)))
    print("     SWK: " + str(cohen_kappa_score(y_testing, dt_y_pred, weights="linear")))


def random_forest(x_train, y_training, x_test, y_testing):
    # Random Forest
    rf = RandomForestClassifier(n_estimators=19, criterion='gini', random_state=0)
    rf.fit(x_train, y_training)
    rf_y_pred = rf.predict(x_test)
    # 0.5363
    print(" Random Forest")
    print("     Accuracy: " + str(accuracy_score(y_testing, rf_y_pred)))
    print("     SWK: " + str(cohen_kappa_score(y_testing, rf_y_pred, weights="linear")))


# kNN by Guoyi
def knn(x, y):
    k_acc_scores = []
    k_swk_scores = []
    swk = make_scorer(cohen_kappa_score, weights="linear")
    acc = make_scorer(accuracy_score)
    k_range = range(1, 30)
    k_fold = 10
    for k in k_range:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        acc_scores = cross_val_score(knn_classifier, X=x, y=y, cv=k_fold, scoring=acc)
        k_acc_scores.append(acc_scores.mean())
        swk_scores = cross_val_score(knn_classifier, X=x, y=y, cv=k_fold, scoring=swk)
        k_swk_scores.append(swk_scores.mean())
    print("kNN")
    print("Choose " + str(k_range[np.argmax(k_acc_scores)]) + " as k for accuracy " + str(k_acc_scores[np.argmax(k_acc_scores)]))
    print("Choose " + str(k_range[np.argmax(k_swk_scores)]) + " as k for swk score " + str(k_swk_scores[np.argmax(k_swk_scores)]))


# Clustering
# Separate the dependent and independent variables
cluster_data = pd.read_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv')
cluster_data = cluster_data.drop(columns=['return_on_investment', 'casts_roi_score', 'directors_roi_score'])
X = cluster_data.iloc[:, 0:12].values
y = cluster_data.iloc[:, 12].values

# Create training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 0)

# Perform feature scaling
# https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
comps = 12
pca = PCA(n_components=comps)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# plot
# x = range(1, comps+1)
# plt.plot(x, pca.explained_variance_)
# plt.xlabel('Components')
# plt.ylabel('Explained Variance')
# plt.show()

# select features
selected_cluster_data = cluster_data[['genres_popularity_score', 'keywords_popularity_score', 'keywords_vote_score',
                                     'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score',
                                      'directors_vote_score', 'release_year', 'return_on_investment_label']]
X = selected_cluster_data.iloc[:, 0:8].values
y = selected_cluster_data.iloc[:, 8].values

# Create training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state = 0)

# clustering
print('Clustering')
decision_tree(X_train, y_train, X_test, y_test)
random_forest(X_train, y_train, X_test, y_test)
knn(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))

# # Percentile
# # Separate the dependent and independent variables
# percent_data = pd.read_csv('data/movies_meta_data_after_processing_percentile_4_label.csv')
# percent_data = percent_data.drop(columns=['return_on_investment'])
# X = percent_data.iloc[:, 0:12].values
# y = percent_data.iloc[:, 12].values
#
# # Create training and testing dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 0)
#
# # Perform feature scaling
# # https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# # Applying PCA
# comps = 12
# pca = PCA(n_components=comps)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_
#
#
# # select features
# selected_percent_data = percent_data[['genres_popularity_score', 'keywords_popularity_score', 'keywords_vote_score',
#                                      'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score',
#                                       'directors_vote_score', 'release_year', 'return_on_investment_label']]
# X = selected_percent_data.iloc[:, 0:8].values
# y = selected_percent_data.iloc[:, 8].values
#
# # Create training and testing dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state = 0)
#
# # percentile
# print('Percentile')
# decision_tree(X_train, y_train, X_test, y_test)
# random_forest(X_train, y_train, X_test, y_test)
# knn(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))

