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
from sklearn import metrics

#use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
def cv_knn(dateframe, k_range, cv_times, model_evaluation, feature_number=9):
    k_scores = []
    feature_array = ['budget','directors_popularity_score', 'casts_popularity_score', 'runtime', 'keywords_popularity_score', 'release_year', 'casts_vote_score', 'release_month', 'directors_vote_score', 'keywords_vote_score']
    if feature_number == 9:
        knn_attrbiutes = dateframe.loc[:,['budget','genres_popularity_score', 'genres_vote_score', 'keywords_popularity_score','keywords_vote_score', 'casts_popularity_score', 'casts_vote_score', 'directors_popularity_score','directors_vote_score']]
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

#load dataset
#Using percentile to create label of return on investment
movies_processed = pd.read_csv('data/movies_meta_data_after_processing.csv')
movies_processed_four_percentile_label = pd.read_csv('data/movies_meta_data_after_processing_percentile_4_label.csv')
movies_processed_three_percentile_label = pd.read_csv('data/movies_meta_data_after_processing_percentile_3_label.csv')
movies_processed_four_equal_range = pd.read_csv('data/movies_meta_data_after_processing_equal_range_4_label.csv')
movies_processed_three_equal_range = pd.read_csv('data/movies_meta_data_after_processing_equal_range_3_label.csv')
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
model_evaluation = "accuracy"
#all features
cv_knn(movies_processed_three_percentile_label, k_range, k_fold, model_evaluation)
cv_knn(movies_processed_four_percentile_label, k_range, k_fold, model_evaluation)
cv_knn(movies_processed_three_equal_range, k_range, k_fold, model_evaluation)
cv_knn(movies_processed_four_equal_range, k_range, k_fold, model_evaluation)
cv_knn(movies_processed_three_cluster_label, k_range, k_fold, model_evaluation)
cv_knn(movies_processed_four_cluster_label, k_range, k_fold, model_evaluation)

#first three features
cv_knn(movies_processed_three_percentile_label, k_range, k_fold, model_evaluation, 3)
cv_knn(movies_processed_four_percentile_label, k_range, k_fold, model_evaluation, 3)
cv_knn(movies_processed_three_equal_range, k_range, k_fold, model_evaluation, 3)
cv_knn(movies_processed_four_equal_range, k_range, k_fold, model_evaluation, 3)
cv_knn(movies_processed_three_cluster_label, k_range, k_fold, model_evaluation, 3)
cv_knn(movies_processed_four_cluster_label, k_range, k_fold, model_evaluation, 3)

#load dataset
k_range = range(1,31)
k_fold = 10
model_evaluation = "accuracy"
cv_knn(movies_processed_three_cluster_label, k_range, k_fold, model_evaluation)



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

