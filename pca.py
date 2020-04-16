#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tyeh3
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Separate the dependent and independent variables
dataset = movies_meta_data
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values

# Create training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 0)

# Perform feature scaling
# https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Applying PCA
# pca = PCA(n_components = 8)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()

# Decision Tree
dt = DecisionTreeClassifier(criterion='gini', random_state=0)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
# 0.4375
accuracy_score(y_test, y_pred)

# Random Forest
rf = RandomForestClassifier(n_estimators=19, criterion='gini', random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
# 0.5363
accuracy_score(y_test, y_pred)