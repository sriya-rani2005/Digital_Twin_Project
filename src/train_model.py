import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Import our data processing functions
import sys
sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, add_rul, normalize

def prepare_features(df):
    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' in df.columns]
    op_cols = ['op1', 'op2', 'op3']
    feature_cols = sensor_cols + op_cols
    X = df[feature_cols]
    y = df['RUL']
    return X, y

def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    print("Training XGBoost...")
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_anomaly_detector(X_train):
    print("Training Isolation Forest (Anomaly Detection)...")
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train)
    return model

def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"\n{name} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    return preds

def plot_results(y_test, preds_rf, preds_xgb):
    os.makedirs('../results', exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:200], preds_rf[:200], alpha=0.5, color='blue')
    plt.plot([0, 300], [0, 300], 'r--')
    plt.title('Random Forest: Actual vs Predicted RUL')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:200], preds_xgb[:200], alpha=0.5, color='green')
    plt.plot([0, 300], [0, 300], 'r--')
    plt.title('XGBoost: Actual vs Predicted RUL')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')

    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    print("\nGraph saved to results/model_comparison.png!")
    plt.show()

if __name__ == "__main__":
    # Load and prepare data
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    anomaly_model = train_anomaly_detector(X_train)

    # Evaluate
    preds_rf  = evaluate_model(rf_model,  X_test, y_test, "Random Forest")
    preds_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Plot
    plot_results(y_test.values, preds_rf, preds_xgb)