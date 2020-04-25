y#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:03:09 2020

@author: tyeh3
"""
# Reference: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, make_scorer
from sklearn.metrics import cohen_kappa_score

def decision_tree(x_train, y_training, x_test, y_testing):
    # Decision Tree
    dt = DecisionTreeClassifier(criterion='gini', random_state=0)
    dt.fit(x_train, y_training)
    dt_y_pred = dt.predict(x_test)
    # 0.4375
    print(" Decision Tree")
    print("     Accuracy: " + str(accuracy_score(y_testing, dt_y_pred)))
    print("     SWK: " + str(cohen_kappa_score(y_testing, dt_y_pred, weights="quadratic")))


def random_forest(x_train, y_training, x_test, y_testing):
    # Random Forest
    rf = RandomForestClassifier(n_estimators=19, criterion='gini', random_state=0)
    rf.fit(x_train, y_training)
    rf_y_pred = rf.predict(x_test)
    # 0.5363
    print(" Random Forest")
    print("     Accuracy: " + str(accuracy_score(y_testing, rf_y_pred)))
    print("     SWK: " + str(cohen_kappa_score(y_testing, rf_y_pred, weights="quadratic")))


# kNN by Guoyi
def knn(x, y):
    k_acc_scores = []
    k_swk_scores = []
    swk = make_scorer(cohen_kappa_score, weights="quadratic")
    bas = make_scorer(accuracy_score)
    k_range = range(1, 30)
    k_fold = 10
    for k in k_range:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        acc_scores = cross_val_score(knn_classifier, X=x, y=y, cv=k_fold, scoring=bas)
        k_acc_scores.append(acc_scores.mean())
        swk_scores = cross_val_score(knn_classifier, X=x, y=y, cv=k_fold, scoring=swk)
        k_swk_scores.append(swk_scores.mean())
    print("kNN")
    print("Choose " + str(k_range[np.argmax(k_acc_scores)]) + " as k for accuracy " + str(k_acc_scores[np.argmax(k_acc_scores)]))
    print("Choose " + str(k_range[np.argmax(k_swk_scores)]) + " as k for swk score " + str(k_swk_scores[np.argmax(k_swk_scores)]))

# Percentile
# Separate the dependent and independent variables
percent_data = pd.read_csv('data/movies_meta_data_after_processing_percentile_4_label.csv')
percent_data = percent_data.drop(columns=['return_on_investment'])
X = percent_data.loc[:, ['budget', 'casts_popularity_score', 'directors_popularity_score', 'runtime', 'keywords_popularity_score', 'release_year']].values
y = percent_data.iloc[:, 12].values

#
# # Create training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 0)

# # percentile
print('Percentile')
decision_tree(X_train, y_train, X_test, y_test)
random_forest(X_train, y_train, X_test, y_test)
knn(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))

# Clustering
# Separate the dependent and independent variables
cluster_data = pd.read_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv')
cluster_data = cluster_data.drop(columns=['return_on_investment'])
X = cluster_data.loc[:, ['budget', 'casts_popularity_score', 'directors_popularity_score', 'runtime', 'keywords_popularity_score', 'release_year']].values
y = cluster_data.iloc[:, 12].values

# Create training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 0)

# clustering
print('Clustering')
decision_tree(X_train, y_train, X_test, y_test)
random_forest(X_train, y_train, X_test, y_test)
knn(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))


# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# data = pd.read_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv')
# data = data.drop(columns=['return_on_investment'])

# X = data.iloc[:,0:10]  #independent columns
# y = data.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features
# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(X,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Attributes','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("data/movies_meta_data_after_processing_percentile_4_label.csv")
data = data.drop(columns=['return_on_investment'])
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")