import pickle
from pathlib import Path

RESULTS_DIR = Path("results/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model, name="linear_regression"):
    file_path = RESULTS_DIR / f"{name}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {file_path}")
    

def load_model(name="linear_regression"):
    
    file_path = RESULTS_DIR / f"{name}.pkl"
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model
