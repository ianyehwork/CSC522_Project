#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:14:24 2020

@author: tyeh3
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Separate the dependent and independent variables
c_dataset = pd.read_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv')
c_dataset = c_dataset.drop(columns=['return_on_investment'])
e_dataset = pd.read_csv('data/movies_meta_data_after_processing_equal_range_4_label.csv')
e_dataset = e_dataset.drop(columns=['return_on_investment'])
p_dataset = pd.read_csv('data/movies_meta_data_after_processing_percentile_4_label.csv')
p_dataset = p_dataset.drop(columns=['return_on_investment'])

def rf_all_attributes(dataset):
    X = dataset.iloc[:, 0:11]
    y = dataset.iloc[:, 12].values

    # Create training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 0)

    rf = RandomForestClassifier(n_estimators=19, criterion='entropy', random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # print("Accuracy of RandomForestClassifier after cross validation: ")
    print(np.average(cross_val_score(rf, X, y, cv=10)))

rf_all_attributes(c_dataset)


def rf_selected_attributes(dataset):
    X = dataset.loc[:, ['budget','runtime','casts_popularity_score','casts_vote_score']]
    y = dataset.iloc[:, 12].values

    # Create training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 0)

    rf = RandomForestClassifier(n_estimators=19, criterion='entropy', random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # print("Accuracy of RandomForestClassifier after cross validation: ")
    print(np.average(cross_val_score(rf, X, y, cv=10)))

rf_selected_attributes(c_dataset)
rf_all_attributes(e_dataset)
rf_selected_attributes(e_dataset)
rf_all_attributes(p_dataset)
rf_selected_attributes(p_dataset)