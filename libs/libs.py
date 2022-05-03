import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


def check_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:

    # Logistic Regression
    log = LogisticRegression()
    log.fit(X, y)
    acc_log = round(log.score(X, y), 2)

    # Support Vector Machines
    svc = SVC()
    svc.fit(X, y)
    acc_svc = round(svc.score(X, y), 2)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    acc_knn = round(knn.score(X, y), 2)

    # Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(X, y)
    acc_gaussian = round(gaussian.score(X, y), 2)

    # Perceptron
    perceptron = Perceptron()
    perceptron.fit(X, y)
    acc_perceptron = round(perceptron.score(X, y), 2)

    # Linear SVC
    linear_svc = LinearSVC()
    linear_svc.fit(X, y)
    acc_linear_svc = round(linear_svc.score(X, y), 2)

    # Stochastic Gradient Descent
    sgd = SGDClassifier()
    sgd.fit(X, y)
    acc_sgd = round(sgd.score(X, y), 2)

    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X, y)
    acc_decision_tree = round(decision_tree.score(X, y), 2)

    # Random forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X, y)
    acc_random_forest = round(random_forest.score(X, y), 2)

    models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log,
                  acc_random_forest, acc_gaussian, acc_perceptron,
                  acc_sgd, acc_linear_svc, acc_decision_tree]})

    models.sort_values(by='Score', ascending=False, inplace=True)

    return models
