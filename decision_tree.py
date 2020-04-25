import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score
import skll
import graphviz

dataset = pd.read_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv').drop(columns=['return_on_investment'])
p_dataset = pd.read_csv('data/movies_meta_data_after_processing_percentile_4_label.csv').drop(columns=['return_on_investment'])


def dt_all_attributes(dataset):
	X = dataset.iloc[:, 0:-1]
	y = dataset.iloc[:, -1].values

	# Create training and testing dataset
	scores = []
	for i in range(10):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle=True)
		X_test.drop(columns=['genres_roi_score', 'keywords_roi_score', 'casts_roi_score', 'directors_roi_score', 'popularity','vote_average'])
	
		dt = tree.DecisionTreeClassifier(criterion='entropy',  max_depth=5, min_samples_split = 35)
		dt.fit(X_train, y_train)
		y_pred = dt.predict(X_test)
	
		scores.append(accuracy_score(y_test, y_pred))
	print(np.average(scores))	
	# for i in range(2, 100):
	# 	dt = tree.DecisionTreeClassifier(criterion='entropy',  max_depth=5, min_samples_split = i)
	# 	dt.fit(X_train, y_train)
	# 	y_pred = dt.predict(X_test)
	# 	cv_scores.append(np.average(cross_val_score(dt, X, y, cv=10)))

	# plt.plot(range(2, 100), cv_scores)
	# plt.xlabel("min_samples_split of decision tree")
	# plt.ylabel("Cross Validation Accuracy")
	# plt.show()
	dot_data = tree.export_graphviz(dt, out_file=None, feature_names=list(X.columns), filled=True) 
	graph = graphviz.Source(dot_data) 
	graph.render("movie_tree") 

	# print("Balanced accuracy score")
	# print(balanced_accuracy_score(y_test, y_pred, sample_weight=None, adjusted=False))
	# print("cohen_kappa_score")
	# print(cohen_kappa_score(y_test, y_pred, weights='quadratic'))

def dt_original_attributes(dataset):
	# X = dataset.loc[:, ['casts_vote_score', 'keywords_vote_score', 'keywords_popularity_score', 'directors_vote_score', 'directors_popularity_score', 'casts_roi_score', 'directors_roi_score']]
	X = dataset.loc[:, ['budget','runtime', 'release_year', 'release_month', 'popularity','vote_average']]
	y = dataset.iloc[:, -1].values
	# Create training and testing dataset
	scores = []
	for i in range(10):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)
		X_test.drop(columns=['popularity','vote_average'])

		dt = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=15)
		dt.fit(X_train, y_train)
		y_pred = dt.predict(X_test)
		scores.append(accuracy_score(y_test, y_pred))

	print(np.average(scores))

	# print("Balanced accuracy score")
	# print(balanced_accuracy_score(y_test, y_pred, sample_weight=None, adjusted=False))
	# print("cohen_kappa_score")
	# print(cohen_kappa_score(y_test, y_pred, weights='quadratic'))

	# dot_data = tree.export_graphviz(dt, out_file=None) 
	# graph = graphviz.Source(dot_data) 
	# graph.render("movie_tree") 

dt_all_attributes(dataset)
dt_original_attributes(dataset)

print("Percentile labels")
dt_all_attributes(p_dataset)
dt_original_attributes(p_dataset)

