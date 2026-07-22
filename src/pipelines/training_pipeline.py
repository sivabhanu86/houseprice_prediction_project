from src.models.custom_LinearRegression import LinearRegression
from src.models.sklearn_models import train_sklearn_model
from src.utils.data_loader import load_processed
from src.utils.metrics import evaluate,print_actual_prices
from src.utils.plotting import plot_predictions
from src.utils.model_saver import save_model

def run_training():
    data = load_processed()
    X = data.drop("price", axis=1)
    y = data["price"]

    # Custom model
    custom = LinearRegression()
    custom.fit(X, y)
    print("\n coefiecients : ",custom.coef_)
    print("\n intercept : ",custom.intercept_)
    preds_custom = custom.predict(X)
    print("\n\n custom linear regresion completed")
    custom.print_predictions(preds_custom)
    plot_predictions(y, preds_custom)
    print("Custom:", evaluate(y, preds_custom))

    print("\n\n sk linear regression started ")
    # Sklearn model
    sk_model = train_sklearn_model(X, y)
    preds_sk = sk_model.predict(X)
    print("\n\n sklearn l r completed")
    
    print_actual_prices(preds_sk)
    plot_predictions(y, preds_custom)
    print("Sklearn:", evaluate(y, preds_sk))

    save_model(custom, name="custom_linear_regression")
    save_model(sk_model, name="sklearn_linear_regression")
