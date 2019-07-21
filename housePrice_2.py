# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import  train_test_split


# Import the train and test set
train = pd.read_csv('DataSets/housing/train.csv')
test = pd.read_csv('DataSets/housing/train.csv')
# print(train.head())
# print(train.info())
# print(train.describe())
print(train.count())
# Any results you write to the current directory are saved as output.

# Handle missing values
print (train.isnull().sum()[train.isnull().sum()>0])

# -------------------------------------------------------------------------------------

'''Variables PoolQC, Fence, MiscFeature and Alley have lots of missing values.
 In the training set as well as in the test set For better results I remove these variables from the sets.'''
# -------------------------------------------------------------------------------------

train = train.drop(['PoolQC','Fence','MiscFeature','Alley','FireplaceQu'], axis=1)
test = test.drop(['PoolQC','Fence','MiscFeature','Alley','FireplaceQu'], axis=1)
# -------------------------------------------------------------------------------------
# Create a heatmap correlation to find relevant variables
# -------------------------------------------------------------------------------------
corr = train.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr)
plt.yticks(rotation=0, size=7)
plt.xticks(rotation=90, size=7)
plt.show()
# -------------------------------------------------------------------------------------
'''The heatmap is a bit to small to provide good information, 
therefor I select only the variables with a correlation greater then 0.5 and create the heatmap'''
# -------------------------------------------------------------------------------------
rel_vars = corr.SalePrice[(corr.SalePrice > 0.5)]
rel_cols = list(rel_vars.index.values)
print(rel_cols)
corr2 = train[rel_cols].corr()
plt.figure(figsize=(8,8))
hm = sns.heatmap(corr2, annot=True, annot_kws={'size':10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.show()

# Create matrix with independent variables
# -------------------------------------------------------------------------------------
X = train[rel_cols[:-1]].iloc[:,0:].values
y = train.iloc[:, -1].values

# Create training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)

# Fit Random Forest on Training Set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train, y_train)

# Score model
print (regressor.score(X_train, y_train))

# Predict new result
y_pred = regressor.predict(X_test)
# Plot y_test vs y_pred
plt.figure(figsize=(12,8))
plt.plot(y_test, color='red')
plt.plot(y_pred, color='blue')
plt.show()