import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor as rfr
from scipy import stats  # I might use this
from scipy.stats import norm
from ModelBuild import getModel,plotResults
from Feature_Ranking import Feature_Ranking
from sklearn.model_selection import cross_val_score

PATH= "DataSets/housing/"
os.environ['PATH']

#use pandas to read in csv files . The pd,read_csv() method creates a DataFram fro a csv file
train =pd.read_csv(f'{PATH}train.csv',low_memory=False)
test= pd.read_csv(f'{PATH}test.csv',low_memory=False)

#checkout  the size of the data
# print ("Train data shape:",train.shape)
# print("Test data shape:",test.shape)


# look at a few rows using the DataFrame.head() method
# train.head()
# print(train.head())

#to do some plotting
plt.style.use(style='ggplot')
# plt.rcParams['figure.figsize'] = (10, 7)

#######################################################
#  2. Explore the data and engineer Features          
#######################################################
numeric_features = train.select_dtypes(include=[np.number])
QFact=numeric_features.drop(['Id'],axis=1)
# Side-by-side: skew and Kurtosis
# ==============================================================
fig=plt.figure(1,figsize=(16,8))
title=fig.suptitle("Skewness and Kurtosis in data ", fontsize=16, fontweight='bold')
fig.subplots_adjust(top=0.88,wspace=0.3)
dataSk=QFact.skew()
dataKu=QFact.kurt()
# ----------
# skew #
#----------
ax1=fig.add_subplot(1,2,1)
# ax1.set_xlabel("features")
ax1.set_ylabel("Skewness")

bars1 = ax1.bar(QFact.columns,dataSk,color='darksalmon',edgecolor='darkred', linewidth=1)
ax1.tick_params(axis='both', which='major', labelsize=8)
plt.xticks(rotation=90)
# ----------
# Kurtosis #
# ----------
ax2 = fig.add_subplot(1,2,2) 
# ax2.set_xlabel("features")
ax2.set_ylabel("Kurtosis") 
bars2 = ax2.bar(QFact.columns,dataKu,color='darksalmon',edgecolor='darkred', linewidth=1)
plt.xticks(rotation=90)
ax2.tick_params(axis='both', which='major', labelsize=8)
# ----------
# Save Figure #
# ----------
fig.savefig('Housing-Price-Prediction/skew_kurt.png')
plt.show()
# -------------------------------------------------------------------------------------
# Target Variable Analysis: Is it Normal?
# ---------------------------------------
# histogram and normal probability plot
sns.distplot(train["SalePrice"], fit=norm)
fig = plt.figure()
res = stats.probplot(train["SalePrice"], plot=plt)
plt.show()
# Scale target 
# ----------
train["SalePrice"] = np.log(train["SalePrice"])
sns.distplot(train["SalePrice"], fit=norm)
fig = plt.figure()
res = stats.probplot(train["SalePrice"], plot=plt)
plt.show()
# -------------------------------------------------------------------------------------
# displays the correlation between the columns and examine the correlations between the features and the target.
# -------------------------------------------------------------------------------------
fig=plt.figure(5,figsize=(10,6))
corr = QFact.corr()
ax=fig.add_subplot(1,1,1)
hm = sns.heatmap(corr, 
            ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
            cmap="coolwarm", # Color Map.
            #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
            annot=False, 
            fmt='.2f',       # String formatting code to use when adding annotations.
            #annot_kws={"size": 14},
            linewidths=.05)
plt.show()

#investigate the relationship between OverallQual and SalePrice.
#We set index='OverallQual' and values='SalePrice'. We chose to look at the median here.
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

#visualize this pivot table more easily, we can create a bar plot
#Notice that the median sales price strictly increases as Overall Quality increases.
# ======================================================================================
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

#to generate some scatter plots and visualize the relationship between the Ground Living Area(GrLivArea) and SalePrice
target=train["SalePrice"]
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()

# do the same for GarageArea.
plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
#######################################################
# create a new dataframe with some outliers removed ###
#######################################################
train = train[train['GarageArea'] < 1200]

# display the previous graph again without outliers
plt.scatter(x=train['GarageArea'], y=train['SalePrice'])
plt.xlim(-200,1600)     # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
######################################################
#   Handling Null Values                            ##
######################################################
# create a DataFrame to view the top null columns and return the counts of the null values in each column
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
#nulls
print(nulls)
#to return a list of the unique values
print ("Unique values are:", train.MiscFeature.unique())
######################################################
#   Wrangling the non-numeric Features              ##
######################################################
# consider the non-numeric features and display details of columns
categoricals = train.select_dtypes(exclude=[np.number])
#categoricals.describe()
print(categoricals.describe())
######################################################
#   Transforming and engineering features           ##
######################################################
# When transforming features, it's important to remember that any transformations that you've applied to the training data before
# fitting the model must be applied to the test data.

#Eg:
print ("Original: \n")
print (train.Street.value_counts(), "\n")

# our model needs numerical data, so we will use one-hot encoding to transform the data into a Boolean column.
# create a new column called enc_street. The pd.get_dummies() method will handle this for us
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)

print ('Encoded: \n')
print (train.enc_street.value_counts())  # Pave and Grvl values converted into 1 and 0

# look at SaleCondition by constructing and plotting a pivot table, as we did above for OverallQual
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

# encode this SaleCondition as a new feature by using a similar method that we used for Street above
def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)


# explore this newly modified feature as a plot.
condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
######################################################################################################
#   Dealing with missing values                                                                      #
#   We'll fill the missing values with an average value and then assign the results to data          #
#   This is a method of interpolation                                                                #
######################################################################################################
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

# Check if the all of the columns have 0 null values.
# sum(data.isnull().sum() != 0)
print(sum(data.isnull().sum() != 0))

######################################################
#  3. Build a Random forest model                   ##
######################################################

# separate the features and the target variable for modeling.
# We will assign the features to X and the target variable(Sales Price)to y.
X = data.drop(['SalePrice', 'Id'], axis=1)
# exclude ID from features since Id is just an index with no relationship to SalePrice.

#======= partition the data ===================================================================================================#
#   Partitioning the data in this way allows us to evaluate how our model might perform on data that it has never seen before.
#   If we train the model on all of the test data, it will be difficult to tell if overfitting has taken place.
#==============================================================================================================================#
# also state how many percentage from train data set, we want to take as test data set
# In this example, about 33% of the data is devoted to the hold-out set.
X_train, X_test, y_train, y_test = train_test_split(X, data['SalePrice'], random_state=42, test_size=.33)

# try fitting a decision tree regression model...
DTR_1 = dtr(
    max_depth=None
)  # declare the regression model form. Let the depth be default.
# DTR_1.fit(X,Y) # fit the training data
scores_dtr = cross_val_score(DTR_1, X_train, y_train, cv=10, scoring="explained_variance")  # 10-fold cross validation
print("scores for k=10 fold validation:", scores_dtr)
print("Est. explained variance: %0.2f (+/- %0.2f)"% (scores_dtr.mean(), scores_dtr.std() * 2))

# ============================================================================
# Seeing the Random Forest for the Trees¶
# =======================================
print("Sweeping no of trees ")
estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
mean_rfrs = []
std_rfrs_upper = []
std_rfrs_lower = []
# yt = [i for i in Y["SalePrice"]]  # quick pre-processing of the target
np.random.seed(42)
for i in estimators:
    model = rfr(n_estimators=i, max_depth=None)
    scores_rfr = cross_val_score(model, X_train, y_train, cv=10, scoring="explained_variance")
    print("estimators:", i)
    #     print('explained variance scores for k=10 fold validation:',scores_rfr)
    print("Est. explained variance: %0.2f (+/- %0.2f)"% (scores_rfr.mean(), scores_rfr.std() * 2))
    print("")
    mean_rfrs.append(scores_rfr.mean())
    std_rfrs_upper.append(scores_rfr.mean() + scores_rfr.std() * 2)  # for error plotting
    std_rfrs_lower.append(scores_rfr.mean() - scores_rfr.std() * 2)  # for error plotting
# and plot...
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(estimators, mean_rfrs, marker="o", linewidth=4, markersize=12)
ax.fill_between(
    estimators,
    std_rfrs_lower,
    std_rfrs_upper,
    facecolor="green",
    alpha=0.3,
    interpolate=True,
)
ax.set_ylim([-0.3, 1])
ax.set_xlim([0, 80])
plt.title("Expected Variance of Random Forest Regressor")
plt.ylabel("Expected Variance")
plt.xlabel("Trees in Forest")
plt.grid()
plt.show()

sorted_scores=Feature_Ranking(X_train,y_train)
# top 15...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X_train,y_train,sorted_scores, 15,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 15,estimators)

# top 20...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X_train,y_train,sorted_scores, 20,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 20,estimators)

# top 30...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X_train,y_train,sorted_scores, 30,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 30,estimators)

# top 40...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X_train,y_train,sorted_scores, 40,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 40,estimators)

# top 50...
mean_rfrs, std_rfrs_upper, std_rfrs_lower = getModel(X_train,y_train,sorted_scores, 50,estimators)
plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 50,estimators)

