# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import matplotlib.pyplot as plt  # some plotting!
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # so pretty!

import sklearn.feature_selection as fs  # feature selection library in scikit-learn
from scipy import stats  # I might use this
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier  # checking if this is available
from sklearn.ensemble import RandomForestRegressor as rfr

# let's set up some cross-validation analysis to evaluate our model and later models...
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.metrics import mean_squared_error

from catEncode import getObjectFeature
from ModelBuild import getModel,plotResults
from Feature_Ranking import Feature_Ranking

# from sklearn import cross_validation
# %matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# import the training data set and make sure it's in correctly...
train = pd.read_csv("DataSets/housing/train.csv")
train_original = pd.read_csv("DataSets/housing/train.csv")
test = pd.read_csv("DataSets/housing/test.csv")
train.info()
# ==========================================================================
# Pre-processing Categorical Features¶
# ====================================
# convert an object (categorical) feature into an int feature
# and test the function...
fcntest = getObjectFeature(train, "LotShape")
print(fcntest.head(10))

# Target Variable Analysis: Is it Normal?
# ====================================
# histogram and normal probability plot
sns.distplot(train["SalePrice"], fit=norm)
fig = plt.figure()
res = stats.probplot(train["SalePrice"], plot=plt)
plt.show()

train["SalePrice"] = np.log(train["SalePrice"])
sns.distplot(train["SalePrice"], fit=norm)
fig = plt.figure()
res = stats.probplot(train["SalePrice"], plot=plt)
plt.show()
# ============================================================================

# define the training data X...
X = train[["MoSold", "YrSold", "LotArea", "BedroomAbvGr"]]
Y = train[["SalePrice"]]
# and the data for the competition submission...
X_test = test[["MoSold", "YrSold", "LotArea", "BedroomAbvGr"]]
print(X.head())
print(Y.head())


# try fitting a decision tree regression model...
DTR_1 = dtr(max_depth=None)  # declare the regression model form. Let the depth be default.
# DTR_1.fit(X,Y) # fit the training data
scores_dtr = cross_val_score(DTR_1, X, Y, cv=10, scoring="explained_variance")  # 10-fold cross validation
print("scores for k=10 fold validation:", scores_dtr)
print("Est. explained variance: %0.2f (+/- %0.2f)"% (scores_dtr.mean(), scores_dtr.std() * 2))

# =================================================================================
# Scientific-ish Feature Analysis to Improve Random Forest Regressors
# ===================================================================
train = pd.read_csv("DataSets/housing/train.csv")  # get the training data again just in case
train["SalePrice"] = np.log(train["SalePrice"])
# first, let's include every feature that has data for all 1460 houses in the data set...
included_features = [
    col
    for col in list(train)
    if len([i for i in train[col].T.notnull() if i == True]) == 1460
    and col != "SalePrice"
    and col != "id"
]
# define the training data X...
X = train[included_features]  # the feature data
Y = train[["SalePrice"]]  # the target
yt = [i for i in Y["SalePrice"]]  # the target list
# and the data for the competition submission...
X_test = test[included_features]
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype == "object":
        X = getObjectFeature(X, col)
# X.head()
# =================================================================================
# Mutual Information Regression Metric for Feature Ranking¶
# ===================================================================
sorted_scores=Feature_Ranking(X,Y)
# =================================================================================
# model Build
# ===================================================================
estimators=[10,20, 30, 40, 50, 60, 70, 80]
# top 15...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X,yt,sorted_scores, 15,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 15,estimators)

# top 20...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X,yt,sorted_scores, 20,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 20,estimators)

# top 30...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X,yt,sorted_scores, 30,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 30,estimators)

# top 40...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X,yt,sorted_scores, 40,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 40,estimators)

# top 50...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X,yt,sorted_scores, 50,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 50,estimators)

# =================================================================================
# The Finale: Building the Output for Submission
# ==============================================
# build the model with the desired parameters...
numFeatures = 40  # the number of features to inlcude
trees = 60  # trees in the forest
included_features = np.array(sorted_scores)[:, 0][:numFeatures]
# define the training data X...
X = train[included_features]
Y = train[["SalePrice"]]
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype == "object":
        X = getObjectFeature(X, col)
yt = [i for i in Y["SalePrice"]]
np.random.seed(11111)
model = rfr(n_estimators=trees, max_depth=None)
scores_rfr = cross_val_score(model, X, yt, cv=10, scoring="explained_variance")
print("explained variance scores for k=10 fold validation:", scores_rfr)
print("Est. explained variance: %0.2f (+/- %0.2f)"% (scores_rfr.mean(), scores_rfr.std() * 2))
# fit the model
model.fit(X, yt)

# testing

# let's read the test data to be sure...
test = pd.read_csv("DataSets/housing/test.csv")

# apply the model to the test data and get the output...
X_test = test[included_features]
y_test=test['SalePrice']
for col in list(X_test):
    if X_test[col].dtype=='object':
        X_test = getObjectFeature(X_test, col, datalength=1459)
y_output = model.predict(X_test.fillna(0)) # get the results and fill nan's with 0

# view this relationship between predictions and actual_values graphically with a scatter plot.
actual_values = y_test
plt.scatter(y_output, actual_values, alpha=.75,color='b')  # alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Random Forest Regression Model')
overlay = 'R^2 is: {}\nRMSE is: {}'.format(model.score(X_test, y_test),mean_squared_error(y_test, y_output))
plt.annotate(s=overlay,xy=(12.2,10.6),size='x-large')
plt.show()

# define the data frame for the results
saleprice = pd.DataFrame(y_output, columns=['SalePrice'])
# print(saleprice.head())
# saleprice.tail()
results = pd.concat([test['Id'],saleprice['SalePrice']],axis=1)
results.head()

# and write to output
results.to_csv('housepricing_submission.csv', index = False)