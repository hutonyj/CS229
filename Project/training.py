import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

# prep
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, QuantileTransformer

# models
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# validation libraries
from sklearn import metrics

from sklearn import svm, metrics
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
import pandas as pd
import seaborn as sns


def train(X, y, lm):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    lm.fit(X_train, y_train)

    try:
        print("score is {}".format(lm.score(X_valid, y_valid)))
    except():
        print("no support for score")

    y_cls_pred = lm.predict(X_valid)

    cnf_matrix = metrics.confusion_matrix(y_valid, y_cls_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    print(cnf_matrix)
    # plot_confusion_matrix(cnf_matrix, classes=['No', 'Yes'],
    #                       title='Confusion matrix, without normalization')


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
    train_df = train_df.drop('Brent_Spot_Price', axis=1)

    #get_corr(train_df)

    feature_cols = [col for col in train_df.columns if
                    'Brent_Spot_Price_pct_change' not in col and 'sustained' not in col and "Brent_Spot_Price" not in col]
    X = train_df[feature_cols]
    y1 = train_df['sustained_1_day_later']
    y2 = train_df['sustained_5_days_later']
    y3 = train_df['sustained_average_1_to_5_days_later']

    y4 = train_df['sustained_25_days_later']
    y5 = train_df['sustained_130_days_later']
    y6 = train_df['sustained_average_1_to_25_days_later']
    y7 = train_df['sustained_average_1_to_130_days_later']



    # get phi
    print("y1_phi_0 {}".format(sum(y1 == 0) / len(y1)))
    print("y1_phi_1 {}".format(sum(y1 == 1) / len(y1)))
    print("y2_phi_0 {}".format(sum(y2 == 0) / len(y2)))
    print("y2_phi_1 {}".format(sum(y2 == 1) / len(y2)))
    print("y3_phi_0 {}".format(sum(y3 == 0) / len(y3)))
    print("y3_phi_1 {}".format(sum(y3 == 1) / len(y3)))

    # -- logistic regression
    print("-- logistic regression")
    # print("sustained_1_day_later")
    # train(X, y1, LogisticRegression())
    # print("sustained_5_days_later")
    # train(X, y2, LogisticRegression())
    # print("sustained_average_1_to_5_days_later")
    # train(X, y3, LogisticRegression())

    # 2nd round
    print("sustained_25_days_later")
    train(X, y4, LogisticRegression())
    print()
    print("sustained_130_days_later")
    train(X, y5, LogisticRegression())
    print()
    print("sustained_average_1_to_25_days_later")
    train(X, y6, LogisticRegression())
    print()
    print("sustained_average_1_to_130_days_later")
    train(X, y7, LogisticRegression())
    print()


    # -- SVM
    print("-- SVM")
    # print("sustained_1_day_later")
    # train(X, y1, svm.SVC(gamma="scale"))
    # print("sustained_5_days_later")
    # train(X, y2, svm.SVC(gamma="scale"))
    # print("sustained_average_1_to_5_days_later")
    # train(X, y3, svm.SVC(gamma="scale"))

    # 2nd round
    print("sustained_25_days_later")
    train(X, y4, svm.SVC(gamma="scale"))
    print()
    print("sustained_130_days_later")
    train(X, y5, svm.SVC(gamma="scale"))
    print()
    print("sustained_average_1_to_25_days_later")
    train(X, y6, svm.SVC(gamma="scale"))
    print()
    print("sustained_average_1_to_130_days_later")
    train(X, y7, svm.SVC(gamma="scale"))
    print()

    print(" --5 nearest neighbors")
    # -- k - nearest neighbors
    print("sustained_1_day_later")
    # train(X, y1, KNeighborsClassifier(n_neighbors=5))
    # print("sustained_5_days_later")
    # train(X, y2, KNeighborsClassifier(n_neighbors=5))
    # print("sustained_average_1_to_5_days_later")
    # train(X, y3, KNeighborsClassifier(n_neighbors=5))

    # 2nd round
    print("sustained_25_days_later")
    train(X, y4, KNeighborsClassifier(n_neighbors=5))
    print()
    print("sustained_130_days_later")
    train(X, y5, KNeighborsClassifier(n_neighbors=5))
    print()
    print("sustained_average_1_to_25_days_later")
    train(X, y6, KNeighborsClassifier(n_neighbors=5))
    print()
    print("sustained_average_1_to_130_days_later")
    train(X, y7, KNeighborsClassifier(n_neighbors=5))
    print()
