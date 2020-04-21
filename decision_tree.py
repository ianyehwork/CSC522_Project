import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import graphviz

dataset = pd.read_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv').drop(columns=['return_on_investment'])
p_dataset = pd.read_csv('data/movies_meta_data_after_processing_percentile_4_label.csv').drop(columns=['return_on_investment'])


def dt_all_attributes(dataset):
	X = dataset.iloc[:, 0:-3]
	y = dataset.iloc[:, -1].values

	# Create training and testing dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle=True, random_state = 0)
	X_test.drop(columns=['genres_roi_score', 'keywords_roi_score', 'casts_roi_score', 'directors_roi_score'])
	# cv_scores = []
	# for i in range(2, 100):
	# 	dt = tree.DecisionTreeClassifier(criterion='entropy',  max_depth=5, min_samples_split = i)
	# 	dt.fit(X_train, y_train)
	# 	y_pred = dt.predict(X_test)
	# 	cv_scores.append(np.average(cross_val_score(dt, X, y, cv=10)))

	# plt.plot(range(2, 100), cv_scores)
	# plt.xlabel("min_samples_split of decision tree")
	# plt.ylabel("Cross Validation Accuracy")
	# plt.show()
	dt = tree.DecisionTreeClassifier(criterion='entropy',  max_depth=5, min_samples_split = 35)
	dt.fit(X_train, y_train)
	y_pred = dt.predict(X_test)
	cv_score = cross_val_score(dt, X_train, y_train, cv=10)
	print(np.average(cv_score))
	

def dt_selected_attributes(dataset):
	# X = dataset.loc[:, ['casts_vote_score', 'keywords_vote_score', 'keywords_popularity_score', 'directors_vote_score', 'directors_popularity_score', 'casts_roi_score', 'directors_roi_score']]
	X = dataset.loc[:, ['budget','runtime','casts_popularity_score','casts_vote_score'] ]
	y = dataset.iloc[:, -1].values

	# Create training and testing dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 0)

	dt = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=15)
	dt.fit(X_train, y_train)
	y_pred = dt.predict(X_test)

	# print(accuracy_score(y_test, y_pred)) 
	cv_score = cross_val_score(dt, X_train, y_train, cv=10)
	print(np.average(cv_score))
	
	dot_data = tree.export_graphviz(dt, out_file=None) 
	graph = graphviz.Source(dot_data) 
	graph.render("movie_tree") 

dt_all_attributes(dataset)
dt_selected_attributes(dataset)

print("Percentile labels")
dt_all_attributes(p_dataset)
dt_selected_attributes(p_dataset)

