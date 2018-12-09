import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn.metrics import mean_squared_error
# models
from sklearn.linear_model import LogisticRegression
# prep
from sklearn.model_selection import train_test_split, cross_val_score

# validation libraries

PCT_CHANGE_THRESHOLD = 0.02

def train(X, y, lm):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    lm.fit(X_train, y_train)

    try:
        print("train score is {}".format(lm.score(X_train, y_train)))
        print("5-fold cross_val score is {}".format(np.mean(cross_val_score(lm, X_train, y_train, cv=5))))
        print("test score is {}".format(lm.score(X_valid, y_valid)))
    except():
        print("no support for score")

    # y_cls_pred = lm.predict(X_valid)
    #
    # cnf_matrix = metrics.confusion_matrix(y_valid, y_cls_pred)
    # np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plot_confusion_matrix(cnf_matrix, classes=['No', 'Yes'],
    #                       title='Confusion matrix, without normalization')


def train_gridient_boost(X,y,lm):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_valid, lm.predict(X_valid))
    print("MSE: %.4f" % mse)

    # #############################################################################
    # Plot training deviance

    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(X_valid)):
        test_score[i] = clf.loss_(y_valid, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    # #############################################################################
    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_corr(train_df):
    cor_column = train_df.columns.values
    corr = train_df.corr()
    corr = corr.round(3)
    corr.insert(0, "corr", cor_column)
    # corr.style.background_gradient()

    corr.to_csv("correlation_matrix.csv", encoding='utf-8', index=False)

    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv('all_reduced.csv')
    train_df = train_df.loc[np.abs(train_df['Brent_Spot_Price_pct_change']) >= PCT_CHANGE_THRESHOLD]

    feature_cols = [col for col in train_df.columns if 'sustained' not in col]
    X = train_df[feature_cols]
    y = train_df['sustained_average_1_to_130_days_later']


    print("Sample size is", train_df.shape[0], "with threshold", PCT_CHANGE_THRESHOLD)
    print("sustained_average_1_to_130_days_later baseline is {}".format(sum(y == 1) / len(y)))
    print()

    #-- SVM
    print("-- SVM")
    train(X, y, svm.SVC(gamma="scale"))

    #--gradient boosting
    print("-- SVM gradient boosting")
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    train_gridient_boost(X, y, clf)

    # -- logistic regression
    print("-- logistic regression")
    train(X, y, LogisticRegression(solver='lbfgs', max_iter=1000, C=3))

    #-- Decision tree
    print("-- Decision Tree")
    train(X,y,tree.DecisionTreeClassifier(max_depth = 8, min_samples_split = 24))

    #-- Bagging
    print("-- Decision Tree Bagging (Generalization of Random Forest)")
    train(X,y,ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth = 8, min_samples_split = 12), n_estimators = 100, max_samples = 0.9, max_features = 0.9))

    #-- Extremely Randomized Trees
    print("-- Extremely Randomized Trees")
    train(X,y,ensemble.ExtraTreesClassifier(n_estimators = 1000, max_depth = 12, min_samples_split = 16, max_features = 0.5))
