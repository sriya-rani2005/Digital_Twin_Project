import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, add_rul, normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from train_model import prepare_features

# ── Sensor descriptions ─────────────────────────────────────────
SENSOR_DESC = {
    's11': 'HPC Outlet Static Pressure',
    's9' : 'Physical Core Speed',
    's4' : 'LPT Outlet Temp',
    's12': 'Ratio of Fan Speed',
    's7' : 'HPC Outlet Pressure',
    's14': 'Corrected Core Speed',
    's15': 'Bypass Ratio',
    's2' : 'LPC Outlet Temp',
    's3' : 'HPC Outlet Temp',
}

TOP_SENSORS = ['s11', 's9', 's4', 's12', 's7']

C_CRITICAL = '#e74c3c'
C_WARNING  = '#f39c12'
C_HEALTHY  = '#2ecc71'
C_BLUE     = '#2980b9'
C_DARK     = '#2c3e50'

def get_status(rul):
    if rul <= 30:  return 'CRITICAL ⚠️',  C_CRITICAL
    if rul <= 60:  return 'WARNING 🔔',   C_WARNING
    return 'HEALTHY ✅', C_HEALTHY

def explain_prediction(row, rul, feat_imp, feature_cols):
    """Generate human readable explanation for a prediction"""
    explanations = []

    for sensor in TOP_SENSORS:
        if sensor not in feature_cols:
            continue
        val  = row[sensor]
        imp  = feat_imp.loc[feat_imp['Feature']==sensor,
                            'Importance'].values
        if len(imp) == 0:
            continue
        imp = imp[0]
        desc = SENSOR_DESC.get(sensor, sensor)

        # Interpret sensor value
        if val > 0.7:
            level = 'HIGH'
            effect = 'indicating stress on engine components'
            col = C_CRITICAL
        elif val > 0.4:
            level = 'MODERATE'
            effect = 'showing normal wear'
            col = C_WARNING
        else:
            level = 'LOW'
            effect = 'within safe operating range'
            col = C_HEALTHY

        explanations.append({
            'sensor' : sensor,
            'desc'   : desc,
            'value'  : val,
            'level'  : level,
            'effect' : effect,
            'imp'    : imp,
            'color'  : col
        })

    return explanations

def print_explanation(rul, explanations, cycle):
    status, _ = get_status(rul)
    print("\n" + "="*65)
    print(f"   🔮  PREDICTION EXPLANATION  —  Cycle {cycle}")
    print("="*65)
    print(f"   Predicted RUL : {rul:.1f} cycles")
    print(f"   Status        : {status}")
    print("-"*65)
    print("   WHY did the model predict this RUL?\n")
    for i, e in enumerate(explanations):
        print(f"   {i+1}. Sensor {e['sensor']} — {e['desc']}")
        print(f"      Value      : {e['value']:.4f} ({e['level']})")
        print(f"      Importance : {e['imp']*100:.1f}% contribution")
        print(f"      Reason     : {e['effect']}")
        print()
    print("="*65)

def plot_explanation(rul, explanations, cycle, unit_data,
                     feature_cols):
    os.makedirs('results', exist_ok=True)
    status, status_col = get_status(rul)

    fig = plt.figure(figsize=(18, 11), facecolor='white')
    fig.suptitle(
        f'🔮  Prediction Explanation  —  '
        f'Cycle {cycle}  |  RUL: {rul:.1f}  |  {status}',
        fontsize=14, fontweight='bold', color=C_DARK, y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.48, wspace=0.38)

    # ── 1. Why panel ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0,1); ax1.set_ylim(0,1)
    ax1.axis('off')
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.03,0.03), 0.94, 0.94,
                         boxstyle="round,pad=0.04",
                         linewidth=2,
                         edgecolor=status_col,
                         facecolor=status_col+'22')
    ax1.add_patch(box)
    ax1.text(0.5, 0.92, 'PREDICTION RESULT',
             ha='center', fontsize=9,
             color=C_DARK, fontweight='bold')
    ax1.text(0.5, 0.76, f'RUL = {rul:.1f} cycles',
             ha='center', fontsize=16,
             color=status_col, fontweight='bold')
    ax1.text(0.5, 0.60, status,
             ha='center', fontsize=13,
             color=status_col, fontweight='bold')
    ax1.text(0.5, 0.44, f'Cycle: {cycle}',
             ha='center', fontsize=10, color='grey')
    pct = max(0, int(rul/125*100))
    ax1.text(0.5, 0.30, f'Health: {pct}%',
             ha='center', fontsize=11,
             color=status_col, fontweight='bold')

    # Main reason
    top = explanations[0]
    ax1.text(0.5, 0.14,
             f'Main cause: {top["sensor"]} is {top["level"]}',
             ha='center', fontsize=8,
             color=C_CRITICAL, fontweight='bold')
    ax1.set_title('Prediction Result',
                  fontweight='bold', fontsize=10)

    # ── 2. Sensor contribution bar ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sensors = [e['sensor'] for e in explanations]
    values  = [e['value']  for e in explanations]
    colors  = [e['color']  for e in explanations]
    bars    = ax2.bar(sensors, values,
                      color=colors,
                      edgecolor='black', linewidth=0.5)
    ax2.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)
    ax2.axhline(y=0.7, color=C_CRITICAL, linestyle='--',
                linewidth=1.2, label='High threshold (0.7)')
    ax2.axhline(y=0.4, color=C_WARNING,  linestyle='--',
                linewidth=1.2, label='Moderate threshold (0.4)')
    ax2.set_title('Key Sensor Values at This Cycle',
                  fontweight='bold', fontsize=10)
    ax2.set_ylabel('Normalized Sensor Value')
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=7)
    ax2.grid(axis='y', alpha=0.3)

    # ── 3. Importance contribution ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    imps   = [e['imp']*100 for e in explanations]
    colors3= [e['color']   for e in explanations]
    bars3  = ax3.barh(sensors, imps,
                      color=colors3,
                      edgecolor='black', linewidth=0.5)
    ax3.bar_label(bars3, fmt='%.1f%%', padding=2, fontsize=8)
    ax3.set_title('Sensor Importance Contribution (%)',
                  fontweight='bold', fontsize=10)
    ax3.set_xlabel('Importance (%)')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    # ── 4-6. Sensor trends ───────────────────────────────────────
    plot_sensors = TOP_SENSORS[:3]
    for i, sensor in enumerate(plot_sensors):
        ax = fig.add_subplot(gs[1, i])
        if sensor in unit_data.columns:
            vals   = unit_data[sensor].values
            cycles = unit_data['cycle'].values
            color  = [e['color'] for e in explanations
                      if e['sensor']==sensor]
            color  = color[0] if color else C_BLUE

            ax.plot(cycles, vals,
                    color=C_BLUE, linewidth=1.5, alpha=0.7)
            ax.fill_between(cycles, vals,
                            alpha=0.12, color=C_BLUE)

            # Mark current cycle
            curr_val = unit_data[unit_data['cycle']==cycle][sensor]
            if len(curr_val) > 0:
                ax.scatter([cycle], [curr_val.values[0]],
                           color=color, s=100, zorder=5,
                           label=f'Current: {curr_val.values[0]:.3f}')
                ax.axvline(x=cycle, color='grey',
                           linestyle=':', linewidth=1)

            ax.axhline(y=0.7, color=C_CRITICAL,
                       linestyle='--', linewidth=1,
                       label='High threshold')
            ax.set_title(f'{sensor} — '
                         f'{SENSOR_DESC.get(sensor,sensor)}',
                         fontweight='bold', fontsize=9)
            ax.set_xlabel('Cycle', fontsize=8)
            ax.set_ylabel('Normalized Value', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.savefig('results/prediction_explanation.png',
                dpi=130, bbox_inches='tight')
    print('\n✅  Saved to results/prediction_explanation.png!')
    plt.show()

if __name__ == '__main__':
    print('Loading data...')
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print('Training model...')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)

    feature_cols = X.columns.tolist()

    # Feature importance
    feat_imp = pd.DataFrame({
        'Feature'   : feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # ── Show 3 scenarios ────────────────────────────────────────
    unit_data = df[df['unit_id']==1].reset_index(drop=True)
    scenarios = [
        ('🟢 HEALTHY  Engine', 0.20),
        ('🟠 WARNING  Engine', 0.75),
        ('🔴 CRITICAL Engine', 0.95),
    ]

    for label, pct in scenarios:
        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
        idx  = int(len(unit_data) * pct)
        row  = unit_data.iloc[idx]
        rul  = model.predict(
            row[feature_cols].values.reshape(1,-1))[0]
        cycle = int(row['cycle'])
        explanations = explain_prediction(
            row, rul, feat_imp, feature_cols)
        print_explanation(rul, explanations, cycle)

    # Plot for WARNING scenario
    print('\nGenerating explanation graph for WARNING scenario...')
    idx  = int(len(unit_data) * 0.75)
    row  = unit_data.iloc[idx]
    rul  = model.predict(
        row[feature_cols].values.reshape(1,-1))[0]
    cycle = int(row['cycle'])
    explanations = explain_prediction(
        row, rul, feat_imp, feature_cols)
    plot_explanation(rul, explanations, cycle,
                     unit_data, feature_cols)