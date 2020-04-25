"""
This program is used for finding the best knn of different roi label
@author: gwang25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import scale

#use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
def cv_knn(dateframe, k_range, cv_times, model_evaluation, feature_number=10):
    k_scores = []
    feature_array = ['budget','directors_popularity_score', 'casts_popularity_score', 'runtime', 'keywords_popularity_score', 'release_year', 'casts_vote_score', 'release_month', 'directors_vote_score', 'keywords_vote_score']
    if feature_number == 10:
        knn_attrbiutes = dateframe.loc[:,['budget', 'release_year', 'release_month', 'genres_popularity_score', 'genres_vote_score', 'keywords_popularity_score','keywords_vote_score', 'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score','directors_vote_score']]
    else:
        feature_choose = feature_array[0:feature_number]
        knn_attrbiutes = dateframe.loc[:,feature_choose]
        knn_attrbiutes.info()
        print(feature_choose)
    #.values will change to array, .ravel() convert that array shape to (n,) which is required
    knn_label = dateframe.loc[:, ['return_on_investment_label']].values.ravel()
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X=knn_attrbiutes, y=knn_label, cv = cv_times, scoring = model_evaluation)
        k_scores.append(scores.mean())
    index_max_score = np.argmax(k_scores)
    print("Choose "+str(k_range[index_max_score])+" as k, "+"the "+ model_evaluation + " is "+str(k_scores[index_max_score]))
    # plot to see clearly
    plt.plot(k_range, k_scores)
    label = "Cross-Validated "+ model_evaluation
    plt.xlabel("Value of K for KNN")
    plt.ylabel(label)
    path = 'graph/CV of knn '+ model_evaluation + '.png'
    plt.savefig(path)
    plt.show()

#Out put of knn_new are accuracy, balanced_accuracy, cohen_kappa
def knn_new(dateframe, k_range, k_fold, origin_attribute=False):
    #dataframe
    #'budget'
    if origin_attribute:
        knn_attrbiutes = dateframe.loc[:,['budget','runtime', 'release_year', 'release_month', 'popularity', 'vote_average']]
    else:
        knn_attrbiutes = dateframe.loc[:,['budget', 'runtime', 'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score']]
        #knn_attrbiutes = dateframe.loc[:, ['budget', 'runtime', 'release_year', 'release_month', 'genres_popularity_score', 'genres_vote_score', 'keywords_popularity_score', 'keywords_vote_score', 'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score', 'directors_vote_score', 'genres_roi_score','keywords_roi_score','casts_roi_score','directors_roi_score']]
    #narray
    print(knn_attrbiutes.columns)
    knn_label = dateframe.loc[:, ['return_on_investment_label']].values.ravel()
    #the different of StratifiedShuffleSplit from stratifiedKold is the random index
    shufflesplit = StratifiedKFold(n_splits = k_fold, shuffle=True, random_state=42)
    mean_accuracy_list = []
    mean_balanced_accuracy_list = []
    mean_cohen_kappa_list = []
    for k in k_range:
        knn_classifier = KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="kd_tree")
        accuracy = []
        balanced_accuracy = []
        cohen_kappa = []
        for train_index, test_index in shufflesplit.split(knn_attrbiutes, knn_label):
            scale_attributes_train = scale(knn_attrbiutes.loc[train_index])
            scale_attributes_test = scale(knn_attrbiutes.loc[test_index])
            predict_test = knn_classifier.fit(scale_attributes_train, knn_label[train_index]).predict(scale_attributes_test)
            accuracy.append(accuracy_score(knn_label[test_index], predict_test))
            balanced_accuracy.append(balanced_accuracy_score(knn_label[test_index], predict_test))
            cohen_kappa.append(cohen_kappa_score(knn_label[test_index], predict_test, weights= "quadratic"))
        mean_accuracy_list.append(np.mean(accuracy))
        mean_balanced_accuracy_list.append(np.mean(balanced_accuracy))
        mean_cohen_kappa_list.append(np.mean(cohen_kappa))
    index_max_accuracy = np.argmax(mean_accuracy_list)
    index_max_balanced_accuracy = np.argmax(mean_balanced_accuracy_list)
    index_max_cohen_kappa = np.argmax(mean_cohen_kappa_list)
    print("Choose " + str(k_range[index_max_accuracy]) + " as k, " + "the accuracy is " + str(
        mean_accuracy_list[index_max_accuracy]))
    print("Choose " + str(k_range[index_max_balanced_accuracy]) + " as k, " + "the balanced_accuracy is " + str(
        mean_balanced_accuracy_list[index_max_balanced_accuracy]))
    print("Choose " + str(k_range[index_max_cohen_kappa]) + " as k, " + "the cohen_kappa is " + str(
        mean_cohen_kappa_list[index_max_cohen_kappa]))
    plt.figure(0)
    plt.plot(k_range, mean_balanced_accuracy_list)
    label = "Cross-Validated " + "balanced accuracy"
    plt.xlabel("Value of K for KNN")
    plt.ylabel(label)
    path = 'graph/CV of knn ' + "balanced accuracy" + '.png'
    plt.savefig(path)

    plt.figure(1)
    plt.plot(k_range, mean_cohen_kappa_list)
    label = "Cross-Validated " + "cohen kappa"
    plt.xlabel("Value of K for KNN")
    plt.ylabel(label)
    path = 'graph/CV of knn ' + "cohen kappa" + '.png'
    plt.savefig(path)
    print(mean_cohen_kappa_list[index_max_balanced_accuracy])

#load dataset
#Using percentile to create label of return on investment
movies_processed = pd.read_csv('data/movies_meta_data_after_processing.csv')
knn_features = movies_processed.loc[:,['budget', 'runtime', 'release_year', 'release_month', 'popularity', 'vote_average']]
knn_feature = scale(knn_features)
movies_processed.isnull().sum()
movies_processed_four_percentile_label = pd.read_csv('data/movies_meta_data_after_processing_percentile_4_label.csv')
movies_processed_three_percentile_label = pd.read_csv('data/movies_meta_data_after_processing_percentile_3_label.csv')
#Using clustering to create label of return on investment
movies_processed_four_cluster_label = pd.read_csv('data/movies_meta_data_after_processing_with_4_cluster_label.csv')
movies_processed_three_cluster_label = pd.read_csv('data/movies_meta_data_after_processing_with_3_cluster_label.csv')
movies_processed_three_cluster_label['return_on_investment_label'].value_counts()
movies_processed_four_cluster_label['return_on_investment_label'].value_counts()

movies_processed_three_percentile_label.return_on_investment_label.value_counts()
#set the range of k
k_range = range(1,31)
k_fold = 10
#f1 maybe better for unbalanced data
#accuracy, precision_macro
#scoring = ['accuracy', 'precision'] multiple metric evaluation
# model_evaluation = "accuracy"
#all features
knn_new(movies_processed_four_percentile_label, k_range, k_fold)
knn_new(movies_processed_four_percentile_label, k_range, k_fold, origin_attribute = True)
knn_new(movies_processed_four_cluster_label, k_range, k_fold)
knn_new(movies_processed_four_cluster_label, k_range, k_fold, origin_attribute = True)

movies_processed_four_cluster_label.describe()
movies_processed_four_cluster_label.groupby('return_on_investment_label').size()
movies_processed_four_cluster_label = movies_processed_four_cluster_label.drop(columns=['popularity','vote_average','return_on_investment','genres_roi_score','keywords_roi_score','casts_roi_score','directors_roi_score'])
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(30,10))
movies_processed_four_cluster_label.loc[:, ['budget', 'runtime', 'release_year', 'release_month', 'genres_popularity_score', 'genres_vote_score', 'keywords_popularity_score', 'keywords_vote_score', 'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score', 'directors_vote_score']] = scale(movies_processed_four_cluster_label.loc[:, ['budget', 'runtime', 'release_year', 'release_month', 'genres_popularity_score', 'genres_vote_score', 'keywords_popularity_score', 'keywords_vote_score', 'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score', 'directors_vote_score']])
parallel_coordinates(movies_processed_four_cluster_label, "return_on_investment_label", color = ('Red', 'Yellow', 'Blue', 'Purple'))
plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
plt.xlabel('Features', fontsize=5)
plt.ylabel('Features values', fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.savefig('graph/CV of knn ' + "Parallel Coordinates Plot" + '.png')
plt.show()

import seaborn as sns
plt.figure(1)
sns.pairplot(movies_processed_four_cluster_label, hue='return_on_investment_label', size=4, markers=["o", "s", "D", "+"])
plt.savefig('graph/CV of knn ' + "PairPlot" + '.png')
plt.show()
# #Over-sampling method. It creates synthetic samples of the minority class
# #imblearn python package is used to over-sample the minority classes
# from imblearn.over_sampling import SMOTE
# smote = SMOTE('minorty')
#
# knn_attrbiutes = movies_processed_four_percentile_label.loc[:,['budget', 'genres_popularity_score', 'genres_vote_score', 'keywords_popularity_score','keywords_vote_score', 'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score','directors_vote_score']]
#     #.values will change to array, .ravel() convert that array shape to (n,) which is required
# knn_label = movies_processed_four_percentile_label.loc[:, ['return_on_investment_label']]#.values.ravel()
# x_train = knn_attrbiutes
# y_train = knn_label
# x_train.shape
# y_train.shape
# X_sm, y_sm = smote.fit_sample(x_train, y_train)
# print(X_sm.shape, y_sm.shape)

