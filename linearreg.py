import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import os
PATH= "DataSets/housing/"
os.environ['PATH']

#use pandas to read in csv files . The pd,read_csv() method creates a DataFram fro a csv file
train =pd.read_csv(f'{PATH}train.csv',low_memory=False)
test= pd.read_csv(f'{PATH}test.csv',low_memory=False)

print("1 \n")

#checkout  the size of the data
print ("Train data shape:",train.shape)
print("Test data shape:",test.shape)

print("2 \n")

# look at a few rows using the DataFrame.head() method
# train.head()
print(train.head())

#to do some plotting
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

#######################################################
#  2. Explore the data and engineer Features          ###
#######################################################
print("3 \n")

# to get more information like count, mean, std, min, max etc
# train.SalePrice.describe()
print (train.SalePrice.describe())

print("4 \n")

# to plot a histogram of SalePrice
print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
sns.distplot(train['SalePrice'])
plt.show()
print("5 \n")
 # use np.log() to transform train.SalePric and calculate the skewness a second time, as well as re-plot the data
target = np.log(train.SalePrice)
print ("\n Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()
sns.distplot(target)
plt.show()

#######################################################
#   Working with Numeric Features                   ###
#######################################################

print("6 \n")
# return a subset of columns matching the specified data types
numeric_features = train.select_dtypes(include=[np.number])
# numeric_features.dtypes
print(numeric_features.dtypes)

print("7 \n")
# displays the correlation between the columns and examine the correlations between the features and the target.
corr = numeric_features.corr()
# The first five features are the most positively correlated with SalePrice, while the next five are the most negatively correlated.
print (corr['SalePrice'].sort_values(ascending=False), '\n')
print (corr['SalePrice'].sort_values(ascending=False))