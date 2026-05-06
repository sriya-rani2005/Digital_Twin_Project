import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os, sys, time
sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, add_rul, normalize
from train_model import prepare_features

C_RF     = '#2980b9'
C_XGB    = '#27ae60'
C_LSTM   = '#8e44ad'
C_DARK   = '#2c3e50'
C_RED    = '#e74c3c'
C_ORANGE = '#f39c12'
C_GREEN  = '#2ecc71'

def score(y_true, y_pred, name, train_time):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:<14} RMSE={rmse:.2f}  MAE={mae:.2f}  "
          f"R²={r2:.3f}  Time={train_time:.1f}s")
    return dict(Model=name, RMSE=round(rmse,2),
                MAE=round(mae,2), R2=round(r2,3),
                TrainTime=round(train_time,1))

def run_comparison(df):
    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    results = []
    preds   = {}

    # Random Forest
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict(X_te)
    results.append(score(y_te, p_rf, 'Random Forest', time.time()-t0))
    preds['Random Forest'] = p_rf

    # XGBoost
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)
    p_xgb = xgb.predict(X_te)
    results.append(score(y_te, p_xgb, 'XGBoost', time.time()-t0))
    preds['XGBoost'] = p_xgb

    # LSTM
    try:
        import tensorflow as tf
        from lstm_model import create_sequences
        print("  Loading LSTM model...")
        lstm   = tf.keras.models.load_model('models/lstm_model.keras')
        X_seq, y_seq = create_sequences(df)
        split        = int(0.8*len(X_seq))
        X_te_seq     = X_seq[split:]
        y_te_seq     = y_seq[split:]
        t0           = time.time()
        p_lstm       = lstm.predict(X_te_seq, verbose=0).flatten()
        results.append(score(y_te_seq, p_lstm,
                             'LSTM', time.time()-t0))
        preds['LSTM'] = (y_te_seq, p_lstm)
    except Exception as e:
        print(f"  LSTM skipped: {e}")

    return pd.DataFrame(results), y_te, preds

def analyze_errors(y_true, y_pred, model_name):
    """Analyze where model is weak — early vs late cycles"""
    errors = np.abs(y_true - y_pred)

    # Split into early, mid, late based on RUL
    early = errors[y_true > 80]   # High RUL = early life
    mid   = errors[(y_true > 30) & (y_true <= 80)]
    late  = errors[y_true <= 30]  # Low RUL = near failure

    print(f"\n  📊 Error Analysis — {model_name}")
    print(f"     Early cycles (RUL>80)  : MAE={early.mean():.2f} "
          f"— {'Less accurate' if early.mean()>15 else 'Accurate'}")
    print(f"     Mid cycles (30<RUL≤80) : MAE={mid.mean():.2f} "
          f"— {'Less accurate' if mid.mean()>15 else 'Accurate'}")
    print(f"     Late cycles (RUL≤30)   : MAE={late.mean():.2f} "
          f"— {'Less accurate' if late.mean()>15 else 'Accurate'}")

    return {
        'early': early.mean(),
        'mid'  : mid.mean(),
        'late' : late.mean()
    }

def plot_comparison(results_df, y_te, preds):
    os.makedirs('results', exist_ok=True)

    fig = plt.figure(figsize=(20, 14), facecolor='white')
    fig.suptitle('📊  Complete Model Analysis — RF vs XGBoost vs LSTM',
                 fontsize=15, fontweight='bold', color=C_DARK)
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.50, wspace=0.38)

    colors  = [C_RF, C_XGB, C_LSTM]
    models  = results_df['Model'].tolist()

    # ── 1. RMSE bar ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(models, results_df['RMSE'],
                   color=colors[:len(models)],
                   edgecolor='black', linewidth=0.5)
    ax1.bar_label(bars, fmt='%.2f', fontsize=9, padding=2)
    ax1.set_title('RMSE  (lower = better)',
                  fontweight='bold', fontsize=10)
    ax1.set_ylabel('RMSE')
    ax1.grid(axis='y', alpha=0.3)

    # ── 2. MAE bar ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(models, results_df['MAE'],
                    color=colors[:len(models)],
                    edgecolor='black', linewidth=0.5)
    ax2.bar_label(bars2, fmt='%.2f', fontsize=9, padding=2)
    ax2.set_title('MAE  (lower = better)',
                  fontweight='bold', fontsize=10)
    ax2.set_ylabel('MAE')
    ax2.grid(axis='y', alpha=0.3)

    # ── 3. R² bar ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(models, results_df['R2'],
                    color=colors[:len(models)],
                    edgecolor='black', linewidth=0.5)
    ax3.bar_label(bars3, fmt='%.3f', fontsize=9, padding=2)
    ax3.set_title('R² Score  (higher = better)',
                  fontweight='bold', fontsize=10)
    ax3.set_ylabel('R²')
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y', alpha=0.3)

    # ── 4. RF scatter ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y_te[:300], preds['Random Forest'][:300],
                alpha=0.5, color=C_RF, s=20)
    ax4.plot([0,125],[0,125],'r--', linewidth=1.2)
    ax4.set_title('Random Forest: Actual vs Predicted',
                  fontweight='bold', fontsize=10)
    ax4.set_xlabel('Actual RUL')
    ax4.set_ylabel('Predicted RUL')
    ax4.grid(True, alpha=0.3)

    # ── 5. XGBoost scatter ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(y_te[:300], preds['XGBoost'][:300],
                alpha=0.5, color=C_XGB, s=20)
    ax5.plot([0,125],[0,125],'r--', linewidth=1.2)
    ax5.set_title('XGBoost: Actual vs Predicted',
                  fontweight='bold', fontsize=10)
    ax5.set_xlabel('Actual RUL')
    ax5.set_ylabel('Predicted RUL')
    ax5.grid(True, alpha=0.3)

    # ── 6. Summary table ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['Model'],
            str(row['RMSE']),
            str(row['MAE']),
            str(row['R2']),
            f"{row['TrainTime']}s"
        ])
    table = ax6.table(
        cellText=table_data,
        colLabels=['Model','RMSE','MAE','R²','Time'],
        cellLoc='center', loc='center',
        bbox=[0, 0.2, 1, 0.75]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for j in range(5):
        table[0,j].set_facecolor(C_DARK)
        table[0,j].set_text_props(color='white',
                                   fontweight='bold')
    row_cols = [C_RF+'33', C_XGB+'33', C_LSTM+'33']
    for i in range(1, len(table_data)+1):
        for j in range(5):
            table[i,j].set_facecolor(row_cols[i-1])
    ax6.set_title('Performance Summary Table',
                  fontweight='bold', fontsize=10, pad=12)

    # ── 7. Error by RUL range — RF ───────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    rf_errors = analyze_errors(
        y_te, preds['Random Forest'], 'Random Forest')
    xgb_errors = analyze_errors(
        y_te, preds['XGBoost'], 'XGBoost')

    stages     = ['Early\n(RUL>80)', 'Mid\n(30<RUL≤80)',
                  'Late\n(RUL≤30)']
    rf_vals    = [rf_errors['early'],
                  rf_errors['mid'],
                  rf_errors['late']]
    xgb_vals   = [xgb_errors['early'],
                  xgb_errors['mid'],
                  xgb_errors['late']]
    x = np.arange(len(stages))
    w = 0.35
    b1 = ax7.bar(x-w/2, rf_vals, w,
                 label='Random Forest',
                 color=C_RF, edgecolor='black', linewidth=0.5)
    b2 = ax7.bar(x+w/2, xgb_vals, w,
                 label='XGBoost',
                 color=C_XGB, edgecolor='black', linewidth=0.5)
    ax7.bar_label(b1, fmt='%.1f', padding=2, fontsize=8)
    ax7.bar_label(b2, fmt='%.1f', padding=2, fontsize=8)
    ax7.set_title('Error by Engine Life Stage\n'
                  '(Where is model weak?)',
                  fontweight='bold', fontsize=10)
    ax7.set_xticks(x)
    ax7.set_xticklabels(stages)
    ax7.set_ylabel('Mean Absolute Error')
    ax7.legend(fontsize=8)
    ax7.grid(axis='y', alpha=0.3)

    # ── 8. Error distribution RF ─────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    residuals_rf  = y_te - preds['Random Forest']
    residuals_xgb = y_te - preds['XGBoost']
    ax8.hist(residuals_rf, bins=40, alpha=0.6,
             color=C_RF, label='Random Forest',
             edgecolor='black', linewidth=0.3)
    ax8.hist(residuals_xgb, bins=40, alpha=0.6,
             color=C_XGB, label='XGBoost',
             edgecolor='black', linewidth=0.3)
    ax8.axvline(x=0, color='black', linestyle='--',
                linewidth=1.5, label='Zero error')
    ax8.set_title('Error Distribution\n'
                  '(centered at 0 = unbiased)',
                  fontweight='bold', fontsize=10)
    ax8.set_xlabel('Prediction Error')
    ax8.set_ylabel('Frequency')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # ── 9. Error insight panel ───────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_xlim(0,1); ax9.set_ylim(0,1)
    ax9.axis('off')
    ax9.set_title('Error Understanding Insights',
                  fontweight='bold', fontsize=10)

    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.03,0.03), 0.94, 0.94,
                         boxstyle="round,pad=0.04",
                         facecolor='#f8f9fa',
                         edgecolor=C_DARK, linewidth=1.5)
    ax9.add_patch(box)

    insights = [
        ('📊 Error Understanding', '',           C_DARK),
        ('',                       '',           C_DARK),
        ('Early Cycles',  'Less accurate',       C_ORANGE),
        ('(RUL > 80)',    'Engine healthy,',      'grey'),
        ('',              'patterns unclear',     'grey'),
        ('',              '',                    C_DARK),
        ('Mid Cycles',    'Moderate accuracy',   C_ORANGE),
        ('(30-80 RUL)',   'Degradation starts',  'grey'),
        ('',              '',                    C_DARK),
        ('Late Cycles',   'Most accurate ✅',    C_GREEN),
        ('(RUL < 30)',    'Clear failure',        'grey'),
        ('',              'pattern detected',    'grey'),
        ('',              '',                    C_DARK),
        ('💡 Insight:',   'Model improves',      C_DARK),
        ('',              'as engine ages!',     C_GREEN),
    ]

    y = 0.94
    for label, val, color in insights:
        if label == '' and val == '':
            y -= 0.02
            continue
        if label:
            ax9.text(0.05, y, label,
                     fontsize=8, color=C_DARK,
                     fontweight='bold')
        if val:
            ax9.text(0.55, y, val,
                     fontsize=8, color=color)
        y -= 0.065

    plt.savefig('results/model_comparison_full.png',
                dpi=130, bbox_inches='tight')
    print('\n✅  Saved to results/model_comparison_full.png!')
    plt.show()

if __name__ == '__main__':
    print('Loading data...')
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    print('\nTraining & evaluating all models:')
    results_df, y_te, preds = run_comparison(df)

    print('\n── Final Results ──')
    print(results_df.to_string(index=False))

    plot_comparison(results_df, y_te, preds)