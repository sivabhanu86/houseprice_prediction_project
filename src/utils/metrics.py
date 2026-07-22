from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(y_true, y_pred):
    mse =mean_squared_error(y_true, y_pred)
    return{
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse, 
        "RMSE":np.sqrt(mse), # correct usage
        "R2": r2_score(y_true, y_pred)
    }

def print_actual_prices(y_pred):
    print(np.exp(y_pred))