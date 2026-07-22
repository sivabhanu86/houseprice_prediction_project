from sklearn.linear_model import LinearRegression as SklearnLinearRegression

def train_sklearn_model(X, y):
    model = SklearnLinearRegression()
    model.fit(X, y)
    return model
