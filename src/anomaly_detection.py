import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, add_rul, normalize

def train_anomaly_detector(X_train):
    print("Training Isolation Forest for Anomaly Detection...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # 5% expected anomalies
        random_state=42
    )
    model.fit(X_scaled)
    return model, scaler

def detect_anomalies(model, scaler, df):
    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' in df.columns]
    X = df[sensor_cols].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_scaled)
    df = df.copy()
    df['anomaly'] = preds
    df['anomaly_score'] = scores
    df['is_anomaly'] = df['anomaly'] == -1
    return df

def plot_anomalies(df, unit_id=1):
    os.makedirs('results', exist_ok=True)
    unit_data = df[df['unit_id'] == unit_id].reset_index(drop=True)
    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' in df.columns]

    # Pick the most variable sensor for visualization
    variances = unit_data[sensor_cols].var()
    top_sensor = variances.idxmax()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Sensor reading with anomalies highlighted
    normal = unit_data[~unit_data['is_anomaly']]
    anomalous = unit_data[unit_data['is_anomaly']]

    axes[0].plot(unit_data['cycle'], unit_data[top_sensor],
                 color='blue', linewidth=1, label=f'Sensor {top_sensor}', alpha=0.7)
    axes[0].scatter(anomalous['cycle'], anomalous[top_sensor],
                    color='red', s=50, zorder=5, label='⚠️ Anomaly Detected')
    axes[0].set_title(f'Unit {unit_id} — Anomaly Detection on {top_sensor}')
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('Sensor Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Anomaly score over time
    axes[1].plot(unit_data['cycle'], unit_data['anomaly_score'],
                 color='green', linewidth=1.5, label='Anomaly Score')
    axes[1].axhline(y=0, color='red', linestyle='--', label='Anomaly Boundary')
    axes[1].fill_between(unit_data['cycle'], unit_data['anomaly_score'], 0,
                         where=unit_data['anomaly_score'] < 0,
                         color='red', alpha=0.2, label='Anomaly Zone')
    axes[1].set_title(f'Unit {unit_id} — Anomaly Score Over Time')
    axes[1].set_xlabel('Cycle')
    axes[1].set_ylabel('Anomaly Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/anomaly_detection.png')
    print("Anomaly detection graph saved to results/anomaly_detection.png!")
    plt.show()

def print_summary(df):
    total = len(df)
    anomalies = df['is_anomaly'].sum()
    print(f"\n📊 Anomaly Detection Summary:")
    print(f"   Total readings  : {total}")
    print(f"   Normal readings : {total - anomalies}")
    print(f"   Anomalies found : {anomalies} ({100*anomalies/total:.1f}%)")

    # Show which units have most anomalies
    print(f"\n🔍 Top 5 units with most anomalies:")
    top_units = df.groupby('unit_id')['is_anomaly'].sum().sort_values(ascending=False).head()
    for unit, count in top_units.items():
        print(f"   Unit {unit}: {count} anomalies ⚠️")

if __name__ == "__main__":
    print("Loading data...")
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' in df.columns]
    X_train = df[sensor_cols].values

    # Train anomaly detector
    model, scaler = train_anomaly_detector(X_train)

    # Detect anomalies across all data
    df = detect_anomalies(model, scaler, df)

    # Print summary
    print_summary(df)

    # Plot for unit 1
    plot_anomalies(df, unit_id=1)