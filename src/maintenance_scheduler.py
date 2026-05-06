import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, add_rul, normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from train_model import prepare_features

# Each cycle = 1 operating day (assumption for scheduling)
CYCLE_TO_DAYS = 1
CRITICAL_RUL  = 30   # schedule maintenance if RUL < 30 days
WARNING_RUL   = 60   # issue warning if RUL < 60 days

def classify_health(rul):
    if rul <= CRITICAL_RUL:
        return 'CRITICAL', 'red'
    elif rul <= WARNING_RUL:
        return 'WARNING', 'orange'
    else:
        return 'HEALTHY', 'green'

def generate_schedule(df, model, unit_ids=None):
    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' in df.columns]
    op_cols     = ['op1', 'op2', 'op3']
    feature_cols = [c for c in sensor_cols + op_cols if c in df.columns]

    if unit_ids is None:
        unit_ids = df['unit_id'].unique()[:10]  # show first 10 machines

    print("\n" + "="*65)
    print("       🏭  PREDICTIVE MAINTENANCE SCHEDULE REPORT")
    print("="*65)
    print(f"{'Machine':<10} {'Current Cycle':<15} {'Predicted RUL':<16} {'Status':<10} {'Action'}")
    print("-"*65)

    schedule = []
    for uid in unit_ids:
        unit_data  = df[df['unit_id'] == uid]
        last_row   = unit_data.iloc[int(len(unit_data)*0.80)]
        features   = last_row[feature_cols].values.reshape(1, -1)
        rul_pred   = model.predict(features)[0]
        status, _  = classify_health(rul_pred)
        cycle      = int(last_row['cycle'])

        if status == 'CRITICAL':
            action = f"⚠️  Schedule NOW — failure in ~{int(rul_pred)} days!"
        elif status == 'WARNING':
            action = f"🔔  Plan maintenance in {int(rul_pred - CRITICAL_RUL)} days"
        else:
            action = f"✅  No action needed"

        print(f"Unit {uid:<6} {cycle:<15} {rul_pred:<16.1f} {status:<10} {action}")
        schedule.append({
            'unit_id': uid,
            'cycle': cycle,
            'predicted_rul': rul_pred,
            'status': status
        })

    print("="*65)
    return pd.DataFrame(schedule)

def plot_schedule(schedule_df):
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: RUL bar chart per machine
    colors = [classify_health(r)[1] for r in schedule_df['predicted_rul']]
    bars = axes[0].bar(
        [f"Unit {i}" for i in schedule_df['unit_id']],
        schedule_df['predicted_rul'],
        color=colors, edgecolor='black', linewidth=0.5
    )
    axes[0].axhline(y=CRITICAL_RUL, color='red',    linestyle='--', linewidth=1.5, label=f'Critical ({CRITICAL_RUL} cycles)')
    axes[0].axhline(y=WARNING_RUL,  color='orange',  linestyle='--', linewidth=1.5, label=f'Warning ({WARNING_RUL} cycles)')
    axes[0].set_title('Predicted Remaining Useful Life per Machine')
    axes[0].set_xlabel('Machine Unit')
    axes[0].set_ylabel('Predicted RUL (cycles/days)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Health status pie chart
    status_counts = schedule_df['status'].value_counts()
    status_colors = {'HEALTHY': 'green', 'WARNING': 'orange', 'CRITICAL': 'red'}
    pie_colors = [status_colors[s] for s in status_counts.index]
    axes[1].pie(
        status_counts.values,
        labels=status_counts.index,
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12}
    )
    axes[1].set_title('Fleet Health Status Overview')

    plt.suptitle('🏭 Digital Twin — Predictive Maintenance Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/maintenance_schedule.png')
    print("\n📊 Maintenance schedule graph saved to results/maintenance_schedule.png!")
    plt.show()

if __name__ == "__main__":
    print("Loading data...")
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    # Train model
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained!")

    # Generate schedule for first 10 machines
    schedule_df = generate_schedule(df, model, unit_ids=range(1, 11))

    # Plot dashboard
    plot_schedule(schedule_df)