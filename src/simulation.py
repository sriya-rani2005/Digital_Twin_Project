import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import sys, os
sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, add_rul, normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from train_model import prepare_features
import time

# ── Colours ─────────────────────────────────────────────────────
C_HEALTHY  = '#2ecc71'
C_WARNING  = '#f39c12'
C_CRITICAL = '#e74c3c'
C_BLUE     = '#2980b9'
C_DARK     = '#2c3e50'
C_BG       = '#f8f9fa'

def get_status(rul):
    if rul <= 30:  return 'CRITICAL ⚠️',  C_CRITICAL
    if rul <= 60:  return 'WARNING 🔔',   C_WARNING
    return 'HEALTHY ✅', C_HEALTHY

def run_simulation(unit_id=1):
    # ── Load & train ────────────────────────────────────────────
    print("Loading data...")
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("Training model...")
    model.fit(X_tr, y_tr)
    print("Model ready! Starting simulation...\n")

    # ── Unit data ───────────────────────────────────────────────
    sensor_cols  = [f's{i}' for i in range(1,22) if f's{i}' in df.columns]
    op_cols      = ['op1','op2','op3']
    feature_cols = [c for c in sensor_cols+op_cols if c in df.columns]
    unit_data    = df[df['unit_id']==unit_id].reset_index(drop=True)

    # ── Pre-compute all predictions ─────────────────────────────
    cycles, ruls, temps, pressures = [], [], [], []
    for _, row in unit_data.iterrows():
        feats = row[feature_cols].values.reshape(1,-1)
        rul   = model.predict(feats)[0]
        cycles.append(int(row['cycle']))
        ruls.append(rul)
        temps.append(row['s2'])      # Fan Inlet Temp
        pressures.append(row['s7'])  # Fan Inlet Pressure

    # ── Setup figure ────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    fig.suptitle(
        f'🏭  Digital Twin Live Simulation  —  Engine Unit {unit_id}',
        fontsize=15, fontweight='bold', color=C_DARK
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    ax_rul  = fig.add_subplot(gs[0, :2])   # RUL timeline (top left)
    ax_stat = fig.add_subplot(gs[0, 2])    # Status panel (top right)
    ax_temp = fig.add_subplot(gs[1, 0])    # Temp sensor
    ax_pres = fig.add_subplot(gs[1, 1])    # Pressure sensor
    ax_bar  = fig.add_subplot(gs[1, 2])    # Health bar

    for ax in [ax_rul, ax_temp, ax_pres]:
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3)

    # ── Animation update ────────────────────────────────────────
    def update(frame):
        if frame >= len(cycles):
            return

        c_now  = cycles[:frame+1]
        r_now  = ruls[:frame+1]
        t_now  = temps[:frame+1]
        p_now  = pressures[:frame+1]
        rul    = ruls[frame]
        status, color = get_status(rul)

        # Print to terminal
        bar_len = int(rul / 125 * 20)
        bar     = '█' * bar_len + '░' * (20 - bar_len)
        print(f"Cycle {cycles[frame]:>3} | RUL: {rul:>6.1f} | "
              f"{bar} | {status}")

        # ── RUL plot ────────────────────────────────────────────
        ax_rul.clear()
        ax_rul.set_facecolor('white')
        ax_rul.grid(True, alpha=0.3)
        ax_rul.fill_between(c_now, r_now, alpha=0.15, color=C_BLUE)
        ax_rul.plot(c_now, r_now, color=C_BLUE,
                    linewidth=2, label='Predicted RUL')
        ax_rul.axhline(30, color=C_CRITICAL, linestyle='--',
                       linewidth=1.2, label='Critical (30)')
        ax_rul.axhline(60, color=C_WARNING,  linestyle='--',
                       linewidth=1.2, label='Warning (60)')
        if rul < 30:
            ax_rul.fill_between(c_now, r_now, 30,
                where=[r<30 for r in r_now],
                color=C_CRITICAL, alpha=0.2)
        ax_rul.set_xlim(0, max(cycles))
        ax_rul.set_ylim(-5, 135)
        ax_rul.set_title('Real-Time RUL Prediction',
                         fontweight='bold')
        ax_rul.set_xlabel('Cycle')
        ax_rul.set_ylabel('Remaining Useful Life')
        ax_rul.legend(loc='upper right', fontsize=8)
        # Dot at current position
        ax_rul.scatter([cycles[frame]], [rul],
                       color=color, s=80, zorder=5)

        # ── Status panel ────────────────────────────────────────
        ax_stat.clear()
        ax_stat.set_xlim(0,1); ax_stat.set_ylim(0,1)
        ax_stat.axis('off')
        ax_stat.set_facecolor(color+'22')
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((0.05,0.05), 0.90, 0.90,
                             boxstyle="round,pad=0.05",
                             linewidth=2, edgecolor=color,
                             facecolor=color+'22')
        ax_stat.add_patch(box)
        ax_stat.text(0.5, 0.80, 'ENGINE STATUS',
                     ha='center', fontsize=9,
                     color=C_DARK, fontweight='bold')
        ax_stat.text(0.5, 0.58, status,
                     ha='center', fontsize=14,
                     color=color, fontweight='bold')
        ax_stat.text(0.5, 0.40, f'RUL: {rul:.1f} cycles',
                     ha='center', fontsize=11, color=C_DARK)
        ax_stat.text(0.5, 0.25, f'Cycle: {cycles[frame]}',
                     ha='center', fontsize=10, color='grey')
        pct = int(rul/125*100)
        ax_stat.text(0.5, 0.12, f'Health: {pct}%',
                     ha='center', fontsize=10, color=color,
                     fontweight='bold')

        # ── Temp sensor ─────────────────────────────────────────
        ax_temp.clear()
        ax_temp.set_facecolor('white')
        ax_temp.grid(True, alpha=0.3)
        ax_temp.plot(c_now, t_now, color='#e74c3c', linewidth=1.5)
        ax_temp.fill_between(c_now, t_now, alpha=0.15, color='#e74c3c')
        ax_temp.set_title('Fan Inlet Temperature (s2)',
                          fontweight='bold', fontsize=9)
        ax_temp.set_xlabel('Cycle', fontsize=8)
        ax_temp.set_xlim(0, max(cycles))

        # ── Pressure sensor ─────────────────────────────────────
        ax_pres.clear()
        ax_pres.set_facecolor('white')
        ax_pres.grid(True, alpha=0.3)
        ax_pres.plot(c_now, p_now, color='#8e44ad', linewidth=1.5)
        ax_pres.fill_between(c_now, p_now, alpha=0.15, color='#8e44ad')
        ax_pres.set_title('Fan Inlet Pressure (s7)',
                          fontweight='bold', fontsize=9)
        ax_pres.set_xlabel('Cycle', fontsize=8)
        ax_pres.set_xlim(0, max(cycles))

        # ── Health bar ──────────────────────────────────────────
        ax_bar.clear()
        ax_bar.set_facecolor('white')
        ax_bar.set_xlim(0,1); ax_bar.set_ylim(0,1)
        ax_bar.axis('off')
        ax_bar.set_title('Engine Health %',
                         fontweight='bold', fontsize=9)
        # background bar
        ax_bar.barh(0.5, 1, height=0.15,
                    color='#ecf0f1', edgecolor='grey')
        # health bar
        ax_bar.barh(0.5, pct/100, height=0.15,
                    color=color, edgecolor='black', linewidth=0.5)
        ax_bar.text(0.5, 0.72, f'{pct}% Health Remaining',
                    ha='center', fontsize=11,
                    color=color, fontweight='bold')
        ax_bar.text(0.5, 0.30, f'RUL: {rul:.1f} / 125 cycles',
                    ha='center', fontsize=10, color=C_DARK)

    # ── Run animation ───────────────────────────────────────────
    print("="*55)
    print("  🏭  DIGITAL TWIN LIVE SIMULATION STARTING")
    print("="*55)
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(cycles),
        interval=100,    # 100ms per cycle
        repeat=False
    )
    plt.show()
    print("\n✅ Simulation complete!")

if __name__ == '__main__':
    run_simulation(unit_id=1)
