from sklearn.linear_model import LogisticRegression

def logistic_regression(X, y):
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf
