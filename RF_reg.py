import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
#===========#
# skew #
#===========#
ax1=fig.add_subplot(1,2,1)
# ax1.set_xlabel("features")
ax1.set_ylabel("Skewness")

bars1 = ax1.bar(QFact.columns,dataSk,color='darksalmon',edgecolor='darkred', linewidth=1)
ax1.tick_params(axis='both', which='major', labelsize=8)
plt.xticks(rotation=90)
#==============#
# Kurtosis #
#==============#
ax2 = fig.add_subplot(1,2,2) 
# ax2.set_xlabel("features")
ax2.set_ylabel("Kurtosis") 
bars2 = ax2.bar(QFact.columns,dataKu,color='darksalmon',edgecolor='darkred', linewidth=1)
plt.xticks(rotation=90)
ax2.tick_params(axis='both', which='major', labelsize=8)
#=============#
# Save Figure #
#=============#
fig.savefig('Housing-Price-Prediction/skew_kurt.png')
plt.show()
# ==============================================================================================
# to plot a histogram of SalePrice
print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
sns.distplot(train['SalePrice'])
plt.show()

target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()
sns.distplot(target)
plt.show()
# displays the correlation between the columns and examine the correlations between the features and the target.
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
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600)     # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()