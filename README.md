This project predicts house prices using both a custom Linear Regression implementation and Scikit‑Learn’s LinearRegression.
It includes data preprocessing, exploratory analysis, model training, evaluation, and saving trained models as .pkl files. 
houseprice_prediction_project/

project structure
│
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── results/               # Model outputs, plots, and saved models
├── src/                   # Source code
│   ├── models/            # Custom and sklearn models
│   ├── pipelines/         # Training pipeline
│   └── utils/             # Utilities (data loader, metrics, plotting, saver)
├── run.py                 # Entry point to run the training pipeline
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

Installation
Clone the repo and install dependencies:

git clone https://github.com/sivabhanu86/houseprice_prediction_project.git
cd houseprice_prediction_project
pip install -r requirements.txt

Usage
Run the training pipeline:

python run.py

This will:
Train both custom and sklearn models
Print metrics (MAE, MSE, RMSE, R²)
Save trained models to results/models/
