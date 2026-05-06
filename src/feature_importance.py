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
    's1' : 'Fan Inlet Temp',
    's2' : 'LPC Outlet Temp',
    's3' : 'HPC Outlet Temp',
    's4' : 'LPT Outlet Temp',
    's5' : 'Fan Inlet Pressure',
    's6' : 'Bypass-duct Pressure',
    's7' : 'HPC Outlet Pressure',
    's8' : 'Physical Fan Speed',
    's9' : 'Physical Core Speed',
    's10': 'Engine Pressure Ratio',
    's11': 'HPC Outlet Static Pressure',
    's12': 'Ratio of Fan Speed',
    's13': 'Corrected Fan Speed',
    's14': 'Corrected Core Speed',
    's15': 'Bypass Ratio',
    's16': 'Burner Fuel-Air Ratio',
    's17': 'Bleed Enthalpy',
    's18': 'Demanded Fan Speed',
    's19': 'Demanded Corrected Fan Speed',
    's20': 'HPT Coolant Bleed',
    's21': 'LPT Coolant Bleed',
    'op1': 'Operational Setting 1',
    'op2': 'Operational Setting 2',
    'op3': 'Operational Setting 3',
}

def get_feature_importance(model, feature_cols):
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature'    : feature_cols,
        'Importance' : importances,
        'Description': [SENSOR_DESC.get(f, f) for f in feature_cols]
    })
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    return feat_imp

def plot_feature_importance(feat_imp):
    os.makedirs('results', exist_ok=True)

    fig = plt.figure(figsize=(18, 13), facecolor='white')
    fig.suptitle('🔍  Feature Importance — Which Sensors Matter Most for RUL Prediction?',
                 fontsize=14, fontweight='bold', color='#2c3e50', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.50, wspace=0.35)

    # ── colours per rank ────────────────────────────────────────
    colors = []
    for i in range(len(feat_imp)):
        if i == 0:   colors.append('#e74c3c')   # red
        elif i == 1: colors.append('#e67e22')   # dark orange
        elif i == 2: colors.append('#f39c12')   # orange
        elif i < 6:  colors.append('#2980b9')   # blue
        else:        colors.append('#95a5a6')   # grey

    # ── Plot 1: horizontal bar — top 15 only ────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    top15     = feat_imp.head(15)
    colors15  = colors[:15]

    labels = [f"{r['Feature']:>4}  {r['Description']}"
              for _, r in top15.iterrows()]

    bars = ax1.barh(labels, top15['Importance'],
                    color=colors15,
                    edgecolor='black', linewidth=0.4)
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance Score', fontsize=10)
    ax1.set_title('Top 15 Sensor Importances  '
                  '(🔴 Critical   🟠 High   🔵 Medium   ⚫ Low)',
                  fontweight='bold', fontsize=11)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.grid(axis='x', alpha=0.3)

    # value labels on bars
    for bar, val in zip(bars, top15['Importance']):
        if val > 0.001:
            ax1.text(bar.get_width() + 0.003,
                     bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=8)

    # ── Plot 2: Top 5 pie ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    top5       = feat_imp.head(5)
    pie_colors = ['#e74c3c','#e67e22','#f39c12','#2980b9','#8e44ad']
    ax2.pie(
        top5['Importance'],
        labels=[f"{r['Feature']}\n{r['Description']}"
                for _, r in top5.iterrows()],
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 8},
        wedgeprops={'edgecolor':'white','linewidth':1.5}
    )
    ax2.set_title('Top 5 Most Important Sensors',
                  fontweight='bold', fontsize=11)

    # ── Plot 3: Cumulative importance ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    cumulative = feat_imp['Importance'].cumsum().values
    ax3.plot(range(1, len(cumulative)+1), cumulative,
             color='#2980b9', linewidth=2,
             marker='o', markersize=5)

    ax3.axhline(y=0.8, color='red',    linestyle='--',
                linewidth=1.5, label='80% threshold')
    ax3.axhline(y=0.9, color='orange', linestyle='--',
                linewidth=1.5, label='90% threshold')

    # mark 80% point
    n80 = next(i+1 for i, c in enumerate(cumulative) if c >= 0.8)
    ax3.axvline(x=n80, color='red', linestyle=':',
                linewidth=1.2)
    ax3.text(n80 + 0.4, 0.45,
             f'{n80} sensors\nexplain 80%',
             fontsize=9, color='red', fontweight='bold')

    ax3.set_title('Cumulative Feature Importance',
                  fontweight='bold', fontsize=11)
    ax3.set_xlabel('Number of Features', fontsize=10)
    ax3.set_ylabel('Cumulative Importance', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlim(0, len(cumulative)+1)

    plt.savefig('results/feature_importance.png',
                dpi=130, bbox_inches='tight')
    print('\n✅  Saved to results/feature_importance.png!')
    plt.show()

def print_summary(feat_imp):
    print("\n" + "="*65)
    print("        🔍  FEATURE IMPORTANCE SUMMARY")
    print("="*65)
    print(f"{'Rank':<5} {'Sensor':<6} {'Description':<32} {'Importance':>10}  Tag")
    print("-"*65)
    for i, (_, row) in enumerate(feat_imp.iterrows()):
        tag = '🔴 CRITICAL' if i < 3 else \
              '🟠 HIGH'     if i < 6 else \
              '🔵 NORMAL'
        print(f"{i+1:<5} {row['Feature']:<6} "
              f"{row['Description']:<32} "
              f"{row['Importance']:>10.4f}  {tag}")
    print("="*65)

    top3     = feat_imp.head(3)
    top3_imp = top3['Importance'].sum()
    names    = ', '.join(top3['Feature'].tolist())
    print(f"\n🏆  TOP 3 MOST IMPORTANT SENSORS:")
    for i, (_, row) in enumerate(top3.iterrows()):
        print(f"   {i+1}. {row['Feature']}  —  {row['Description']}")
        print(f"      Importance : {row['Importance']:.4f}  "
              f"({row['Importance']*100:.1f}% contribution)")
    print(f"\n💡  INSIGHT:")
    print(f"   Sensors {names} together explain "
          f"{top3_imp*100:.1f}% of all RUL predictions!")
    print(f"   👉  These 3 sensors are the PRIMARY reason")
    print(f"       the model detects engine degradation.\n")

if __name__ == '__main__':
    print('Loading data...')
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print('Training Random Forest...')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)

    feature_cols = X.columns.tolist()
    feat_imp     = get_feature_importance(model, feature_cols)

    print_summary(feat_imp)
    plot_feature_importance(feat_imp)