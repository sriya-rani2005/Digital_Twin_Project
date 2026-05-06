import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, sys
sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, add_rul, normalize
from train_model import prepare_features

C_HEALTHY  = '#2ecc71'
C_WARNING  = '#f39c12'
C_CRITICAL = '#e74c3c'
C_BLUE     = '#2980b9'
C_DARK     = '#2c3e50'
C_PURPLE   = '#8e44ad'
C_BG       = '#f8f9fa'

SENSOR_DESC = {
    's11': 'HPC Outlet Static Pressure',
    's9' : 'Physical Core Speed',
    's4' : 'LPT Outlet Temp',
    's12': 'Ratio of Fan Speed',
    's2' : 'LPC Outlet Temp',
    's7' : 'HPC Outlet Pressure',
}

def get_status(rul):
    if rul <= 30:  return 'CRITICAL ⚠️', C_CRITICAL
    if rul <= 60:  return 'WARNING 🔔',  C_WARNING
    return 'HEALTHY ✅', C_HEALTHY

def get_failure_reason(row, feature_cols):
    """Find which sensor is causing degradation"""
    top_sensors = ['s11','s9','s4','s12','s7']
    reasons = []
    for s in top_sensors:
        if s not in feature_cols:
            continue
        val  = float(row[s])
        desc = SENSOR_DESC.get(s, s)
        if val > 0.7:
            reasons.append(
                f"⚠️ {s} ({desc}) is HIGH at {val:.3f}")
        elif val > 0.5:
            reasons.append(
                f"🔔 {s} ({desc}) is MODERATE at {val:.3f}")
    if not reasons:
        reasons.append("✅ All sensors within normal range")
    return reasons

class DigitalTwinReplica:
    """
    Full Digital Twin Virtual Replica:
    Inputs  → sensor values (21 sensors + 3 operational)
    Process → ML model predicts RUL
    Output  → RUL prediction
    Decision→ HEALTHY / WARNING / CRITICAL
    Behavior→ updates every cycle
    """
    def __init__(self, model, unit_data, unit_id, feature_cols):
        self.model        = model
        self.unit_data    = unit_data.reset_index(drop=True)
        self.unit_id      = unit_id
        self.feature_cols = feature_cols
        self.sensor_cols  = [f's{i}' for i in range(1,22)
                             if f's{i}' in unit_data.columns]

        # Histories
        self.rul_history      = []
        self.cycle_history    = []
        self.status_history   = []
        self.anomaly_history  = []
        self.reason_history   = []
        self.sensor_histories = {s:[] for s in self.sensor_cols}

    def run(self):
        """Run Digital Twin cycle by cycle"""
        print(f"\n🏭 Running Digital Twin for Unit {self.unit_id}...")
        print(f"{'Cycle':>6} {'RUL':>8} {'Status':<12} {'Main Cause'}")
        print("-"*65)

        for _, row in self.unit_data.iterrows():
            features = row[self.feature_cols].values.reshape(1,-1)
            rul      = self.model.predict(features)[0]
            status, _= get_status(rul)
            reasons  = get_failure_reason(row, self.feature_cols)
            cycle    = int(row['cycle'])

            self.rul_history.append(rul)
            self.cycle_history.append(cycle)
            self.status_history.append(status)
            self.reason_history.append(reasons[0])

            for s in self.sensor_cols:
                self.sensor_histories[s].append(float(row[s]))

            # Anomaly = sudden RUL drop
            if len(self.rul_history) > 1:
                drop = self.rul_history[-2] - self.rul_history[-1]
                self.anomaly_history.append(drop > 15)
            else:
                self.anomaly_history.append(False)

            # Print every 20 cycles
            if cycle % 20 == 0 or cycle == 1:
                cause = reasons[0][:35] if reasons else 'Normal'
                print(f"{cycle:>6} {rul:>8.1f} {status:<12} {cause}")

        print("-"*65)
        print(f"  Final RUL    : {self.rul_history[-1]:.1f} cycles")
        print(f"  Final Status : {self.status_history[-1]}")
        print(f"  Anomalies    : {sum(self.anomaly_history)}")

    def plot(self):
        os.makedirs('results', exist_ok=True)
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        fig.suptitle(
            f'🏭  Digital Twin Virtual Replica  —  Engine Unit {self.unit_id}\n'
            f'Inputs → Sensor Values  |  Process → ML Prediction  |  '
            f'Output → RUL  |  Decision → Health Status',
            fontsize=13, fontweight='bold', color=C_DARK, y=0.98
        )
        gs = gridspec.GridSpec(3, 4, figure=fig,
                               hspace=0.50, wspace=0.40)

        last_rul    = self.rul_history[-1]
        last_reason = self.reason_history[-1]
        status, scol= get_status(last_rul)
        cycles      = self.cycle_history
        pct         = max(0, int(last_rul/125*100))

        # ── 1. System flow diagram (top full width) ──────────────
        ax_flow = fig.add_subplot(gs[0, :])
        ax_flow.set_xlim(0,1); ax_flow.set_ylim(0,1)
        ax_flow.axis('off')
        ax_flow.set_title(
            'Digital Twin System Architecture — Input → Process → Output → Decision',
            fontweight='bold', fontsize=11)

        boxes = [
            (0.05, '📡 INPUTS\n21 Sensors\n3 Op Settings',  C_BLUE),
            (0.28, '⚙️ PROCESS\nRandom Forest\nML Model',   C_PURPLE),
            (0.51, '📊 OUTPUT\nRUL Prediction\n(cycles)',    C_WARNING),
            (0.74, '🚦 DECISION\nHEALTHY /\nWARNING / CRITICAL', scol),
        ]
        for x, text, color in boxes:
            box = FancyBboxPatch((x, 0.15), 0.18, 0.70,
                                 boxstyle="round,pad=0.03",
                                 facecolor=color+'33',
                                 edgecolor=color, linewidth=2)
            ax_flow.add_patch(box)
            ax_flow.text(x+0.09, 0.50, text,
                         ha='center', va='center',
                         fontsize=9, fontweight='bold',
                         color=C_DARK)
        # Arrows
        for x in [0.235, 0.465, 0.695]:
            ax_flow.annotate('',
                xy=(x+0.015, 0.50),
                xytext=(x, 0.50),
                arrowprops=dict(arrowstyle='->', color=C_DARK,
                                lw=2))

        # Current values
        ax_flow.text(0.14, 0.08,
                     f'Current: {len(self.cycle_history)} cycles processed',
                     ha='center', fontsize=8, color='grey')
        ax_flow.text(0.60, 0.08,
                     f'Predicted RUL: {last_rul:.1f} cycles',
                     ha='center', fontsize=8,
                     color=C_WARNING, fontweight='bold')
        ax_flow.text(0.83, 0.08,
                     f'Status: {status}',
                     ha='center', fontsize=8,
                     color=scol, fontweight='bold')

        # ── 2. RUL timeline ──────────────────────────────────────
        ax_rul = fig.add_subplot(gs[1, :3])
        ax_rul.fill_between(cycles, self.rul_history,
                            alpha=0.15, color=C_BLUE)
        ax_rul.plot(cycles, self.rul_history,
                    color=C_BLUE, linewidth=2,
                    label='Predicted RUL')
        ax_rul.axhline(30, color=C_CRITICAL, linestyle='--',
                       linewidth=1.2, label='Critical (30)')
        ax_rul.axhline(60, color=C_WARNING,  linestyle='--',
                       linewidth=1.2, label='Warning (60)')
        if any(r<30 for r in self.rul_history):
            ax_rul.fill_between(cycles, self.rul_history, 30,
                where=[r<30 for r in self.rul_history],
                color=C_CRITICAL, alpha=0.20,
                label='Danger Zone')
        # Anomaly dots
        anom_c = [c for c,a in
                  zip(cycles, self.anomaly_history) if a]
        anom_r = [self.rul_history[cycles.index(c)]
                  for c in anom_c]
        if anom_c:
            ax_rul.scatter(anom_c, anom_r,
                           color=C_CRITICAL, s=50, zorder=5,
                           label=f'Anomalies ({len(anom_c)})')
        ax_rul.set_title('Real-Time RUL Prediction '
                         '(Output of Digital Twin)',
                         fontweight='bold')
        ax_rul.set_xlabel('Operating Cycle')
        ax_rul.set_ylabel('Remaining Useful Life (cycles)')
        ax_rul.legend(loc='upper right', fontsize=8)
        ax_rul.grid(True, alpha=0.3)
        ax_rul.set_xlim(0, max(cycles))
        ax_rul.set_ylim(-5, 135)

        # ── 3. Status panel ──────────────────────────────────────
        ax_st = fig.add_subplot(gs[1, 3])
        ax_st.set_xlim(0,1); ax_st.set_ylim(0,1)
        ax_st.axis('off')
        box = FancyBboxPatch((0.03,0.03), 0.94, 0.94,
                             boxstyle="round,pad=0.04",
                             facecolor=scol+'22',
                             edgecolor=scol, linewidth=2)
        ax_st.add_patch(box)
        ax_st.text(0.5,0.88,'DIGITAL TWIN STATUS',
                   ha='center',fontsize=8,
                   color=C_DARK,fontweight='bold')
        ax_st.text(0.5,0.72,status,
                   ha='center',fontsize=13,
                   color=scol,fontweight='bold')
        ax_st.text(0.5,0.57,f'RUL: {last_rul:.1f} cycles',
                   ha='center',fontsize=11,color=C_DARK)
        ax_st.text(0.5,0.44,f'Health: {pct}%',
                   ha='center',fontsize=10,
                   color=scol,fontweight='bold')
        ax_st.text(0.5,0.32,f'Cycle: {cycles[-1]}',
                   ha='center',fontsize=9,color='grey')
        ax_st.text(0.5,0.20,f'Anomalies: {sum(self.anomaly_history)}',
                   ha='center',fontsize=9,color=C_CRITICAL)
        # Failure reason
        reason_short = last_reason[:30] if last_reason else 'Normal'
        ax_st.text(0.5,0.09,f'Cause: {reason_short}',
                   ha='center',fontsize=7,
                   color=C_CRITICAL,style='italic')
        ax_st.set_title('Decision Output',
                        fontweight='bold',fontsize=10)

        # ── 4. Top 4 sensor trends (bottom row) ──────────────────
        top_sensors = ['s11','s9','s4','s12']
        colors_s    = [C_CRITICAL, C_BLUE, C_WARNING, C_PURPLE]
        for i, (sensor, color) in enumerate(
                zip(top_sensors, colors_s)):
            ax = fig.add_subplot(gs[2, i])
            if sensor in self.sensor_histories:
                vals = self.sensor_histories[sensor]
                ax.plot(cycles, vals,
                        color=color, linewidth=1.2, alpha=0.8)
                ax.fill_between(cycles, vals,
                                alpha=0.12, color=color)
                ax.axhline(y=0.7, color=C_CRITICAL,
                           linestyle='--', linewidth=0.8,
                           label='High threshold')
                desc = SENSOR_DESC.get(sensor, sensor)
                ax.set_title(f'{sensor} — {desc}',
                             fontsize=8, fontweight='bold')
                ax.set_xlabel('Cycle', fontsize=7)
                ax.set_ylabel('Norm. Value', fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=6)

        plt.savefig('results/digital_twin_replica.png',
                    dpi=130, bbox_inches='tight')
        print('\n✅  Saved to results/digital_twin_replica.png!')
        plt.show()

if __name__ == '__main__':
    print('Loading data...')
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    print('Model ready!')

    feature_cols = X.columns.tolist()
    unit_data    = df[df['unit_id']==1]

    twin = DigitalTwinReplica(
        model, unit_data, 1, feature_cols)
    twin.run()
    twin.plot()