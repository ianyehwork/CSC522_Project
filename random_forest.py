#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:14:24 2020

@author: tyeh3
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score
import skll
import graphviz

# Separate the dependent and independent variables
c_dataset = pd.read_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv')
c_dataset = c_dataset.drop(columns=['return_on_investment'])
p_dataset = pd.read_csv('data/movies_meta_data_after_processing_percentile_4_label.csv')
p_dataset = p_dataset.drop(columns=['return_on_investment'])

def rf_all_attributes(dataset):
    X = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1].values

    # Create training and testing dataset
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle=True)
        X_test.drop(columns=['genres_roi_score', 'keywords_roi_score', 'casts_roi_score', 'directors_roi_score', 'popularity','vote_average'])
        rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=1, criterion='entropy')
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        scores.append(accuracy_score(y_test, y_pred))
    print(np.average(scores))

    # visualization - uncomment if needed
    dot_data = tree.export_graphviz(rf.estimators_[1], out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("movie_tree_random_forest") 


    # plt.xlabel("Number of n_estimators in Random Forest")
    # plt.ylabel("Cross Validation Accuracy")
    # plt.show()

    
    # print("Balanced accuracy score")
    # print(balanced_accuracy_score(y_test, y_pred, sample_weight=None, adjusted=False))
    # print("cohen_kappa_score")
    # print(cohen_kappa_score(y_test, y_pred, weights='quadratic'))


def rf_original_attributes(dataset):
    X = dataset.loc[:, ['budget','runtime', 'release_year', 'release_month', 'popularity','vote_average']]
    y = dataset.iloc[:, -1].values

    # Create training and testing dataset
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)
        X_test.drop(columns=['popularity','vote_average'])
        rf = RandomForestClassifier(n_estimators=19, criterion='entropy', random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    print(np.average(scores))

    # # print("Squared weighted kappa accuracy original features")
    # print(skll.metrics.kappa(y_test, y_pred, weights='quadratic', allow_off_by_one=False))
    # print("Balanced accuracy score")
    # print(balanced_accuracy_score(y_test, y_pred, sample_weight=None, adjusted=False))
    # print("cohen_kappa_score")
    # print(cohen_kappa_score(y_test, y_pred, weights='quadratic'))

rf_all_attributes(c_dataset)
rf_original_attributes(c_dataset)

print("Percentile labels")
rf_all_attributes(p_dataset)
rf_original_attributes(p_dataset)