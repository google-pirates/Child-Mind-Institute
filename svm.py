from sklearn import svm

def svm(X, y):
    model = svm.SVC().fit(X, y)
    return model