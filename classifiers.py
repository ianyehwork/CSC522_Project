from kmean import movies_meta_data
import pandas as pd
import sklearn
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import graphviz

# Read in the movies_meta_data from csv
# movies_meta_data = pd.read_csv("movies_meta_data_after_preprocessing.csv")

# Equal frequency binning of ROI
# levels = pd.qcut(movies_meta_data['return_on_investment'], 5, labels=['Very Low', 'Low', 'Average', 'High', 'Very High'])
levels = movies_meta_data['return_on_investment_label']
x_train = movies_meta_data.copy()
del x_train['return_on_investment']
del x_train['return_on_investment_label']
del x_train['Unnamed: 0']

# Normalization?
# X_scaled = preprocessing.scale(x_train)

# Split the dataset into train/test with 2/3 being training data and 1/3 being testing data.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x_train, levels, shuffle=True, test_size=0.2)

# Decision Tree
clf = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=15)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of Decision Tree classifier: ")
print(clf.score(X_test, y_test))
cv_score = cross_val_score(clf, x_train, levels, cv=10)
print("Accuracy of Decision Tree classifier after cross validation: ")
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))



dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("movie_tree") 

# Random Forest
clf_random_forest = RandomForestClassifier().fit(X_train, y_train)
print("Accuracy of Random Forest classifier: ")
print(clf_random_forest.score(X_test, y_test))

print("Accuracy of Decision Tree classifier after cross validation: ")
print(cross_val_score(clf_random_forest, x_train, levels, cv=10))

