from sklearn.linear_model import LinearRegression

def model(x,y):
    lr = LinearRegression
    ret = lr.fit(x,y)
    return ret