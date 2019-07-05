# define a function to do the necessary model building....
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

import numpy as np
def getModel(X_train,y_train,sorted_scores, numFeatures,estimators):
    np.random.seed(42)
    included_features = np.array(sorted_scores)[:, 0][:numFeatures]  # ordered list of important features
    X=X_train[included_features]

    mean_rfrs = []
    std_rfrs_upper = []
    std_rfrs_lower = []
    # yt = [i for i in Y["SalePrice"]]

    # for each number of estimators, fit the model and find the results for 8-fold cross validation
    for i in estimators:
        model = rfr(n_estimators=i, max_depth=None)
        scores_rfr = cross_val_score(model, X, y_train, cv=10, scoring="explained_variance")
        mean_rfrs.append(scores_rfr.mean())
        std_rfrs_upper.append(scores_rfr.mean() + scores_rfr.std() * 2)  # for error plotting
        std_rfrs_lower.append(scores_rfr.mean() - scores_rfr.std() * 2)  # for error plotting
    return mean_rfrs, std_rfrs_upper, std_rfrs_lower

def plotResults(mean_rfrs, std_rfrs_upper, std_rfrs_lower, numFeatures,estimators):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(estimators, mean_rfrs, marker="o", linewidth=4, markersize=12)
    ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,facecolor="green",alpha=0.3,interpolate=True)
    ax.set_ylim([-0.2, 1])
    ax.set_xlim([0, 80])
    plt.title("Expected Variance of Random Forest Regressor: Top %d Features" % numFeatures)
    plt.ylabel("Expected Variance")
    plt.xlabel("Trees in Forest")
    plt.grid()
    plt.show()
    return