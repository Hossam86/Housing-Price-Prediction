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
print ("\n Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()
sns.distplot(target)
plt.show()

#######################################################
#   Working with Numeric Features                   ###
#######################################################
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