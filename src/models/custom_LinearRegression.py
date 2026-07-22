import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.best_fit = None

    def fit(self, X, y):
        rows, _ = X.shape
        # Add bias term (column of ones)
        X_new = np.c_[np.ones((rows, 1)), X]
        # Normal equation: (XᵀX)⁻¹ Xᵀy
        self.best_fit = np.linalg.pinv(X_new.T.dot(X_new)).dot(X_new.T).dot(y)
        self.intercept_ = self.best_fit[0]
        self.coef_ = self.best_fit[1:]

    def predict(self, X):
        rows, _ = X.shape
        X_new = np.c_[np.ones((rows, 1)), X]
        return X_new.dot(self.best_fit)
    
    def print_predictions(self, y_pred):
        # Since target was log(price), exponentiate to get actual prices
        print(np.exp(y_pred))
