import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
sys.path.append(os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_processing import load_data, add_rul, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEQUENCE_LENGTH = 30
C_BLUE   = '#2980b9'
C_PURPLE = '#8e44ad'
C_GREEN  = '#2ecc71'
C_RED    = '#e74c3c'
C_DARK   = '#2c3e50'

def create_sequences(df):
    sensor_cols = [f's{i}' for i in range(1,22)
                   if f's{i}' in df.columns]
    X, y = [], []
    for uid in df['unit_id'].unique():
        unit     = df[df['unit_id']==uid].reset_index(drop=True)
        sensors  = unit[sensor_cols].values
        ruls     = unit['RUL'].values
        for i in range(len(unit) - SEQUENCE_LENGTH):
            X.append(sensors[i:i+SEQUENCE_LENGTH])
            y.append(ruls[i+SEQUENCE_LENGTH])
    return np.array(X), np.array(y)

def build_improved_lstm(input_shape):
    """
    Improved LSTM:
    - 3 LSTM layers (deeper = captures more complex patterns)
    - BatchNormalization (stabilizes training)
    - Dropout after each layer (prevents overfitting)
    - ReduceLROnPlateau (auto reduces learning rate)
    """
    model = Sequential([
        # Layer 1 — learns basic time patterns
        LSTM(128, input_shape=input_shape,
             return_sequences=True,
             name='lstm_layer_1'),
        BatchNormalization(),
        Dropout(0.3, name='dropout_1'),

        # Layer 2 — learns deeper patterns
        LSTM(64, return_sequences=True,
             name='lstm_layer_2'),
        BatchNormalization(),
        Dropout(0.2, name='dropout_2'),

        # Layer 3 — final temporal understanding
        LSTM(32, return_sequences=False,
             name='lstm_layer_3'),
        Dropout(0.2, name='dropout_3'),

        # Dense layers for prediction
        Dense(32, activation='relu', name='dense_1'),
        Dropout(0.1, name='dropout_4'),
        Dense(16, activation='relu', name='dense_2'),
        Dense(1,  name='output')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def print_model_info(model):
    print("\n" + "="*60)
    print("   🧠  IMPROVED LSTM ARCHITECTURE")
    print("="*60)
    print(f"   {'Layer':<25} {'Output Shape':<20} {'Parameters'}")
    print("-"*60)
    total_params = 0
    for layer in model.layers:
        params = layer.count_params()
        total_params += params
        print(f"   {layer.name:<25} "
              f"{str(layer.output_shape):<20} "
              f"{params:,}")
    print("-"*60)
    print(f"   {'Total Parameters':<25} {'':20} {total_params:,}")
    print("="*60)
    print(f"""
   📋 IMPROVEMENTS OVER BASIC LSTM:
   ✅ 3 LSTM layers (was 2) — captures deeper patterns
   ✅ BatchNormalization — stabilizes & speeds training
   ✅ Dropout 0.3/0.2/0.2 — stronger overfitting prevention
   ✅ ReduceLROnPlateau — auto adjusts learning rate
   ✅ 128/64/32 units — more capacity to learn
""")

def train_improved_lstm(X_train, y_train):
    print("\n🧠 Training Improved LSTM...")
    model = build_improved_lstm(
        (X_train.shape[1], X_train.shape[2]))
    print_model_info(model)

    callbacks = [
        EarlyStopping(monitor='val_loss',
                      patience=7,
                      restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5,
                          patience=3,
                          min_lr=1e-6,
                          verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

def plot_results(y_test, preds_new, preds_old, history):
    os.makedirs('results', exist_ok=True)
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    fig.suptitle('🧠  Improved LSTM — Architecture & Results',
                 fontsize=14, fontweight='bold', color=C_DARK)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.38)

    # ── 1. Architecture diagram ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0,1); ax1.set_ylim(0,1)
    ax1.axis('off')
    ax1.set_title('Improved LSTM Architecture',
                  fontweight='bold', fontsize=10)

    layers = [
        ('Input\n(30 cycles × 21 sensors)', '#3498db'),
        ('LSTM Layer 1\n128 units + BatchNorm + Dropout 0.3', '#9b59b6'),
        ('LSTM Layer 2\n64 units + BatchNorm + Dropout 0.2',  '#8e44ad'),
        ('LSTM Layer 3\n32 units + Dropout 0.2',              '#6c3483'),
        ('Dense 32 + Dense 16',                               '#e67e22'),
        ('Output: RUL Prediction',                            '#e74c3c'),
    ]
    from matplotlib.patches import FancyBboxPatch
    y_pos = 0.92
    for text, color in layers:
        box = FancyBboxPatch((0.05, y_pos-0.10), 0.90, 0.09,
                             boxstyle="round,pad=0.02",
                             facecolor=color+'33',
                             edgecolor=color, linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(0.50, y_pos-0.055, text,
                 ha='center', va='center',
                 fontsize=7, color=C_DARK)
        if y_pos > 0.25:
            ax1.annotate('', xy=(0.50, y_pos-0.12),
                         xytext=(0.50, y_pos-0.10),
                         arrowprops=dict(arrowstyle='->',
                                         color='grey'))
        y_pos -= 0.155

    # ── 2. Training loss ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history['loss'],
             color=C_BLUE, linewidth=2, label='Train Loss')
    ax2.plot(history.history['val_loss'],
             color=C_RED, linewidth=2,
             linestyle='--', label='Val Loss')
    ax2.set_title('Training vs Validation Loss',
                  fontweight='bold', fontsize=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── 3. Actual vs Predicted ───────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_test[:300], preds_new[:300],
                alpha=0.5, color=C_PURPLE, s=20,
                label='Improved LSTM')
    ax3.plot([0,125],[0,125],'r--', linewidth=1.5,
             label='Perfect prediction')
    ax3.set_title('Improved LSTM: Actual vs Predicted',
                  fontweight='bold', fontsize=10)
    ax3.set_xlabel('Actual RUL')
    ax3.set_ylabel('Predicted RUL')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4. Old vs New comparison ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    rmse_old = np.sqrt(mean_squared_error(y_test, preds_old))
    rmse_new = np.sqrt(mean_squared_error(y_test, preds_new))
    mae_old  = mean_absolute_error(y_test, preds_old)
    mae_new  = mean_absolute_error(y_test, preds_new)
    r2_old   = r2_score(y_test, preds_old)
    r2_new   = r2_score(y_test, preds_new)

    metrics  = ['RMSE', 'MAE']
    old_vals = [rmse_old, mae_old]
    new_vals = [rmse_new, mae_new]
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax4.bar(x-w/2, old_vals, w, label='Basic LSTM',
                 color=C_BLUE+'99', edgecolor='black',
                 linewidth=0.5)
    b2 = ax4.bar(x+w/2, new_vals, w, label='Improved LSTM',
                 color=C_PURPLE, edgecolor='black',
                 linewidth=0.5)
    ax4.bar_label(b1, fmt='%.2f', padding=2, fontsize=8)
    ax4.bar_label(b2, fmt='%.2f', padding=2, fontsize=8)
    ax4.set_title('Basic vs Improved LSTM',
                  fontweight='bold', fontsize=10)
    ax4.set_xticks(x); ax4.set_xticklabels(metrics)
    ax4.set_ylabel('Error (lower = better)')
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    # ── 5. R² comparison ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    models = ['Basic\nLSTM', 'Improved\nLSTM']
    r2s    = [r2_old, r2_new]
    cols   = [C_BLUE, C_PURPLE]
    bars5  = ax5.bar(models, r2s, color=cols,
                     edgecolor='black', linewidth=0.5)
    ax5.bar_label(bars5, fmt='%.3f', padding=2, fontsize=10)
    ax5.set_title('R² Score Comparison\n(higher = better)',
                  fontweight='bold', fontsize=10)
    ax5.set_ylabel('R² Score')
    ax5.set_ylim(0, 1.1)
    ax5.grid(axis='y', alpha=0.3)

    # ── 6. Summary panel ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0,1); ax6.set_ylim(0,1)
    ax6.axis('off')
    ax6.set_title('Improvement Summary',
                  fontweight='bold', fontsize=10)

    rmse_imp = ((rmse_old - rmse_new)/rmse_old)*100
    mae_imp  = ((mae_old  - mae_new) /mae_old )*100
    r2_imp   = ((r2_new   - r2_old)  /r2_old  )*100

    summary = [
        ('Basic LSTM RMSE',    f'{rmse_old:.2f}', C_BLUE),
        ('Improved LSTM RMSE', f'{rmse_new:.2f}', C_PURPLE),
        ('RMSE Improvement',   f'{rmse_imp:.1f}%', C_GREEN),
        ('',                   '',                C_DARK),
        ('Basic LSTM R²',      f'{r2_old:.3f}',  C_BLUE),
        ('Improved LSTM R²',   f'{r2_new:.3f}',  C_PURPLE),
        ('R² Improvement',     f'+{r2_imp:.1f}%', C_GREEN),
        ('',                   '',                C_DARK),
        ('Key Improvements',   '',                C_DARK),
        ('3 LSTM layers',      '✅', C_GREEN),
        ('BatchNormalization',  '✅', C_GREEN),
        ('ReduceLROnPlateau',   '✅', C_GREEN),
        ('Stronger Dropout',    '✅', C_GREEN),
    ]

    y = 0.95
    for label, val, color in summary:
        if label == '':
            y -= 0.04
            continue
        ax6.text(0.05, y, label,
                 fontsize=8, color=C_DARK, fontweight='bold')
        ax6.text(0.75, y, val,
                 fontsize=8, color=color, fontweight='bold')
        y -= 0.07

    plt.savefig('results/improved_lstm.png',
                dpi=130, bbox_inches='tight')
    print('\n✅  Saved to results/improved_lstm.png!')
    plt.show()

if __name__ == '__main__':
    print('Loading data...')
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    print('Creating sequences...')
    X, y = create_sequences(df)
    split    = int(0.8*len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    print(f'Train: {X_tr.shape}  Test: {X_te.shape}')

    # Train improved model
    model, history = train_improved_lstm(X_tr, y_tr)
    preds_new = model.predict(X_te, verbose=0).flatten()

    # Load old model for comparison
    import tensorflow as tf
    try:
        old_model  = tf.keras.models.load_model(
            'models/lstm_model.keras')
        preds_old  = old_model.predict(X_te, verbose=0).flatten()
        print('\n✅  Loaded basic LSTM for comparison')
    except:
        preds_old = preds_new  # fallback

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_te, preds_new))
    mae  = mean_absolute_error(y_te, preds_new)
    r2   = r2_score(y_te, preds_new)
    print(f'\n🏆  Improved LSTM Results:')
    print(f'    RMSE : {rmse:.2f}')
    print(f'    MAE  : {mae:.2f}')
    print(f'    R²   : {r2:.3f} ({r2*100:.1f}% accuracy)')

    # Save improved model
    os.makedirs('models', exist_ok=True)
    model.save('models/improved_lstm.keras')
    print('✅  Model saved to models/improved_lstm.keras')

    plot_results(y_te, preds_new, preds_old, history)