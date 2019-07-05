import sklearn.feature_selection as fs  # feature selection library in scikit-learn
import matplotlib.pyplot as plt
import numpy as np

def Feature_Ranking(X_train,y_train):
    '''Mutual Information Regression Metric for Feature Ranking'''
    mir_result = fs.mutual_info_regression(
        X_train,y_train
    )  # mutual information regression feature ordering
    feature_scores = []
    for i in np.arange(len(X_train.columns)):
        feature_scores.append([X_train.columns[i], mir_result[i]])
    sorted_scores = sorted(np.array(feature_scores), key=lambda s: float(s[1]), reverse=True)
    print(np.array(sorted_scores))
    # and plot...
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    ind = np.arange(len(X_train.columns))
    plt.bar(ind, [float(i) for i in np.array(sorted_scores)[:, 1]])
    ax.axes.set_xticks(ind)
    plt.title("Feature Importances (Mutual Information Regression)")
    plt.ylabel("Importance")
    # plt.xlabel('Trees in Forest')
    # plt.grid()
    plt.show()
    return sorted_scores