from sklearn.linear_model import LinearRegression

def model(X,y):
    lr = LinearRegression
    ret = lr.fit(X,y)
    return ret