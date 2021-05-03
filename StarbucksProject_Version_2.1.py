# -*- coding: utf-8 -*-
"""
Application of Machine Learning On International Market Analysis
Can ML Models Classify A Potential Market For An International Company?


Created on Fri Apr  9 09:31:38 2021
@author: Stanislav (Stas) Lukashevich

Trying to classify (aka predict) whether a country (in this example, Ukraine) is a good market for Starbucks

Features (data on countries) used in this case:
    1. GDP per capita
    2. Population
    3. Coffee Consumption Per Capita
    4. Ease of Doing Business Score
    5. FDI net inflows
Target: Binary for whether Starbucks is present in a country or not.

Data Sources: The World Bank, Starbucks Investor Relations, International Coffee Organization
"""

########################### STEP 1: PREPARING DATA ###########################

# Importing important libraries:
import pandas as pd
import numpy as np

# Importing Training Data - all countries & features with Ukraine not included
train = pd.read_csv("SBUCKS2019-Countries-Stores-CoffeeConsumption-Population-GDPpc-Business-FDI-FINALDATA-WITHOUT-UKRAINE.csv")

# Our target Y is the column that indicates Starbucks present in the country or not (1 or 0)
target = train['Starbucks Presence']

# Converting string labels into numbers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
starbucksPresence = le.fit_transform(target)

train.drop(['Country Name', 'Country Code', 'Starbucks Presence', 'Stores',
       'Stores Per 1M Residents', 'Coffee Consumption Total (kg)'],
           axis='columns', inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train,target, test_size=0.20)




################### STEP 2: NAIVE BAYES CLASSIFICATION MODEL #################

from sklearn.naive_bayes import GaussianNB
modelNB = GaussianNB()
modelNB.fit(X_train, y_train)
y_predNB =  modelNB.predict(X_test)

print()
print("Is Ukraine a Starbucks Market? ",modelNB.predict([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print(modelNB.predict_proba([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
print('Naive Bayes Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=modelNB.predict(X_train)).round(3))
print('Naive Bayes Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_predNB).round(3))
print('Cross Validation Accuracy Score For NB: ', cross_val_score(modelNB, train, target, cv=7).mean().round(3))
print()



######################### STEP 3: DECISION TREE MODEL ########################

from sklearn.tree import DecisionTreeClassifier
modelDT = DecisionTreeClassifier(max_depth = 4, random_state = 0)
modelDT.fit(X_train, y_train)
y_predDT =  modelDT.predict(X_test)

print()
print("Is Ukraine a Starbucks Market? ",modelDT.predict([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print(modelDT.predict_proba([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print('Decision Tree Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=modelDT.predict(X_train)).round(3))
print('Decision Tree Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_predDT).round(3))
print('Cross Validation Accuracy Score For DT: ', cross_val_score(modelDT, train, target, cv=7).mean().round(3))
print()

import matplotlib.pyplot as plt
import sklearn.tree as tree
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)
tree.plot_tree(modelDT);
fig.savefig('tree_depth4.png')




######################### STEP 4: RANDOM FOREST MODEL #########################

from sklearn.ensemble import RandomForestClassifier # for random forest classifier
modelRF = RandomForestClassifier(n_estimators=100,max_depth=7)
modelRF.fit(X_train, y_train)
y_predRF =  modelRF.predict(X_test)

print()
print("Is Ukraine a Starbucks Market? ",modelRF.predict([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print(modelRF.predict_proba([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print('Random Forest  Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=modelRF.predict(X_train)).round(3))
print('Random Forest  Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_predRF).round(3))
print('Cross Validation Accuracy Score For RF: ', cross_val_score(modelRF, train, target, cv=7).mean().round(3))
print()



############################ STEP 5: kNN ALGORITHM ###########################

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors = 7)
modelKNN.fit(X_train,y_train)
y_predKNN = modelKNN.predict(X_test)

print()
print("Is Ukraine a Starbucks Market? ", modelKNN.predict([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print(modelKNN.predict_proba([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print('k Nearest Neighbor   Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=modelKNN.predict(X_train)).round(3))
print('k Nearest Neighbor   Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_predKNN).round(3))
print('Cross Validation Accuracy Score For kNN: ', cross_val_score(modelKNN, train, target, cv=7).mean().round(3))
print()




############################# STEP 6:LOGISTIC REGRESSION #####################

from sklearn.linear_model import LogisticRegression
modelLR = LogisticRegression()
modelLR.fit(X_train, y_train)
y_predLR = modelLR.predict(X_test)

print()
print("Is Ukraine a Starbucks Market? ", modelLR.predict([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print(modelLR.predict_proba([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print('Logistic Regression Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=modelLR.predict(X_train)).round(3))
print('Logistic Regression Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_predLR).round(3))
print('Cross Validation Accuracy Score For LR: ', cross_val_score(modelLR, train, target, cv=7).mean().round(3))
print()


####################### STEP 7: SUPPORT VECTOR MACHINES ####################

from sklearn.svm import SVC
modelSVM = SVC(probability=True)
modelSVM.fit(X_train, y_train)
y_predSVM = modelSVM.predict(X_test)

print()
print("Is Ukraine a Starbucks Market? ", modelSVM.predict([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print(modelSVM.predict_proba([[1.79, 3659.03, 44385155, 70.2106, 5833000000]]))
print('SVM Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=modelSVM.predict(X_train)).round(3))
print('SVM Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_predSVM).round(3))
print('Cross Validation Accuracy Score For SVM: ', cross_val_score(modelSVM, train, target, cv=7).mean().round(3))
print()




########################## STEP 8: FEATURE IMPORTANCE ######################

# feature importance for the DECISION TREE model
importancesDT = pd.DataFrame({'feature':X_train.columns,'DecisionTree importance':np.round(modelDT.feature_importances_,3)})
importancesDT = importancesDT.sort_values('DecisionTree importance',ascending=False).set_index('feature')
print(importancesDT.head(10))
print()

# feature importance for the RANDOM FOREST model
importances = pd.DataFrame({'feature':X_train.columns,'RandomForest importance':np.round(modelRF.feature_importances_,3)})
importances = importances.sort_values('RandomForest importance',ascending=False).set_index('feature')
print(importances.head(10))
