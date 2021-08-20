#!/usr/bin/python

import sys
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from tools.feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
sys.path.append("C:/Users/Marcus/PycharmProjects/ud120-projects/tools")



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'exercised_stock_options', 'expenses', 'bonus'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Total number of rows
print("Total number of data points: %i" % len(data_dict))
# POI vs Non-POI
poi = 0
for ppp in data_dict:
    if data_dict[ppp]['poi'] == True:
        poi += 1
print("Total number of POI: %i" % poi)
print("Total number of non-POI: %i" % (len(data_dict) - poi))

# Total number of data points: 146
# Total number of poi: 18
# Total number of non-poi: 128


### We convert the dict to a dataframe as it's easier to work with
df = pd.DataFrame.from_records(data_dict).T

### First thing I notice is NaN values in "numeric" data sets, this tells me that the data is NOT stored as float or int.
### This needs to be changed for the accuracy of the ML algorithm we decide to go with
### Lets get a list of just the numeric columns, then convert those to float inside the dataframe using astype().
num_cols = [i for i in df.columns if i != 'email_address']
df[num_cols] = df[num_cols].astype(float)

### Now we still have NaN in the string columns, lets go ahead and convert those to nan values so we can properly remove them.
df.loc[df.email_address == 'NaN', 'email_address'] = np.nan

### Task 2: Remove outliers

### Here we remove some weird outliers we found in the data set
### Total is the total of all the data points, the travel agency is obviously not an employee, and eugene is full of nulls and thus is not useful.
df = df[(df.index != 'TOTAL') & (df.index != 'THE TRAVEL AGENCY IN THE PARK') & (df.index != 'LOCKHART EUGENE E')]

### Task 3: Create new feature(s)

### We create a new feature
df['exercised_expense_ratio'] = df.exercised_stock_options / df.expenses
df['bonus_expense_ratio'] = df.bonus / df.expenses


### Fill nan / na / null with 0's
df = df.fillna(0)

### Turn our dataframe back to a dict
data_dict = df.T.to_dict()

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Get all the features
features_list2 = ['exercised_expense_ratio', 'bonus_expense_ratio']
features_list3 = features_list + features_list2

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list3, sort_keys = True)
labels, features = targetFeatureSplit(data)

data2 = featureFormat(my_dataset, features_list, sort_keys = True)
labels2, features2 = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier()
param_grid = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 4, 6, 8, 10, 20],
              'max_depth': [None, 5, 10, 15, 20],
              'max_features': [None, 'sqrt', 'log2', 'auto']}

clf_grid = GridSearchCV(clf, param_grid=param_grid, scoring='f1', cv=5)
clf_grid.fit(features, labels)
# Get the best algorithm hyperparameters for the Decision Tree
print(clf_grid.best_params_)

best_tree = clf_grid.best_estimator_

rclf = RandomForestClassifier()

param_grid = {
    "n_estimators": [9, 18, 27, 36],
    "max_depth": [None, 1, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6]}

# Use GridSearchCV to find the optimal hyperparameters for the classifier
rclf_grid = GridSearchCV(rclf, param_grid=param_grid, scoring='f1', cv=5)
rclf_grid.fit(features, labels)
# Get the best algorithm hyperparameters for the Decision Tree
print(rclf_grid.best_params_)
best_random_tree = clf_grid.best_estimator_

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

min_features_to_select=3
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=3003135)
rfecv = RFECV(estimator=best_random_tree, step=1, cv=rkf,
              min_features_to_select=min_features_to_select, n_jobs=-1)
rfecv.fit(features_train, labels_train)
y = rfecv.predict(features_test)


###Print the ranking of features that rfe suggest we use, with 1 representing keep, and 2+ representing what is less impactful in order of impact
print(rfecv.ranking_)

###Print the ideal number of features suggested by the algorithm
print("Ideal number of features : %d" % rfecv.n_features_)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()

###Print the the amount of incorrect predictions, out of the total number of predictions
print("Total incorrect predictions of points out of a total %d points predicted : %d"
     % (y.shape[0], (labels_test != y).sum()))

###Print the precision of the algorithm
print(precision_score(labels_test, y, average='weighted'))

###Print the recall of the algorithm
print(recall_score(labels_test, y, average='weighted'))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(rfecv, my_dataset, features_list)