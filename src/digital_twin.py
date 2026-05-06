import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, add_rul, normalize

class DigitalTwin:
    def __init__(self, model, sensor_cols):
        self.model = model
        self.sensor_cols = sensor_cols
        self.history = []
        self.rul_predictions = []
        self.alert_threshold = 30  # Alert if RUL drops below 30 cycles

    def sync(self, sensor_reading):
        """Receive live sensor data and predict RUL"""
        self.history.append(sensor_reading)
        features = sensor_reading[self.sensor_cols + ['op1', 'op2', 'op3']]
        rul_pred = self.model.predict([features])[0]
        self.rul_predictions.append(rul_pred)

        # Alert system
        if rul_pred < self.alert_threshold:
            print(f"⚠️  ALERT! Predicted RUL = {rul_pred:.1f} cycles — Schedule maintenance NOW!")
        else:
            print(f"✅  Machine OK — Predicted RUL = {rul_pred:.1f} cycles")

        return rul_pred

    def simulate(self, df, unit_id=1):
        """Simulate the digital twin on one machine's data"""
        print(f"\n🔄 Starting Digital Twin Simulation for Unit {unit_id}...\n")
        unit_data = df[df['unit_id'] == unit_id].reset_index(drop=True)

        for i, row in unit_data.iterrows():
            self.sync(row)
            if i > 50:  # Simulate only first 50 readings for demo
                break

        self.plot_rul()

    def plot_rul(self):
        """Plot the RUL predictions over time"""
        os.makedirs('../results', exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(self.rul_predictions, color='blue', linewidth=2, label='Predicted RUL')
        plt.axhline(y=self.alert_threshold, color='red', linestyle='--', label='Alert Threshold (30 cycles)')
        plt.fill_between(range(len(self.rul_predictions)),
                         self.rul_predictions,
                         self.alert_threshold,
                         where=[r < self.alert_threshold for r in self.rul_predictions],
                         color='red', alpha=0.2, label='Danger Zone')
        plt.title('Digital Twin — Real-Time RUL Prediction')
        plt.xlabel('Time (cycles)')
        plt.ylabel('Remaining Useful Life (cycles)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/digital_twin_simulation.png')
        print("\n📊 Simulation graph saved to results/digital_twin_simulation.png!")
        plt.show()

if __name__ == "__main__":
    # Load data
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' in df.columns]

    # Train a quick model for simulation
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from train_model import prepare_features

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Run Digital Twin simulation
    twin = DigitalTwin(model, sensor_cols)
    twin.simulate(df, unit_id=1)