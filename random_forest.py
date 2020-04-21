#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:14:24 2020

@author: tyeh3
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Separate the dependent and independent variables
c_dataset = pd.read_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv')
c_dataset = c_dataset.drop(columns=['return_on_investment'])
p_dataset = pd.read_csv('data/movies_meta_data_after_processing_percentile_4_label.csv')
p_dataset = p_dataset.drop(columns=['return_on_investment'])

def rf_all_attributes(dataset):
    X = dataset.iloc[:, 0:-3]
    y = dataset.iloc[:, -1].values

    # Create training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle=True, random_state = 0)
    X_test.drop(columns=['genres_roi_score', 'keywords_roi_score', 'casts_roi_score', 'directors_roi_score'])
    cv_scores = []
    # for i in range(1, 100):
    #     rf = RandomForestClassifier(n_estimators=i, min_samples_leaf = 1, criterion='entropy', random_state=0)
    #     rf.fit(X_train, y_train)
    #     y_pred = rf.predict(X_test)

    #     cv_scores.append(np.average(cross_val_score(rf, X, y, cv=10)))

    # plt.plot(range(1, 100), cv_scores)
    # plt.xlabel("Number of n_estimators in Random Forest")
    # plt.ylabel("Cross Validation Accuracy")
    # plt.show()

    rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=1, criterion='entropy', random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(np.average(cross_val_score(rf, X, y, cv=10)))



def rf_selected_attributes(dataset):
    X = dataset.loc[:, ['budget','runtime','casts_popularity_score','casts_vote_score']]
    y = dataset.iloc[:, -1].values

    # Create training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 0)

    rf = RandomForestClassifier(n_estimators=19, criterion='entropy', random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # print("Accuracy of RandomForestClassifier after cross validation: ")
    print(np.average(cross_val_score(rf, X, y, cv=10)))

rf_all_attributes(c_dataset)
rf_selected_attributes(c_dataset)

print("Percentile labels")
rf_all_attributes(p_dataset)
rf_selected_attributes(p_dataset)