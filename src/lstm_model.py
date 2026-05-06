import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, sys
sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, add_rul, normalize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEQUENCE_LENGTH = 30

def create_sequences(df):
    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' in df.columns]
    X, y = [], []
    for unit_id in df['unit_id'].unique():
        unit_data    = df[df['unit_id'] == unit_id].reset_index(drop=True)
        unit_sensors = unit_data[sensor_cols].values
        unit_rul     = unit_data['RUL'].values
        for i in range(len(unit_data) - SEQUENCE_LENGTH):
            X.append(unit_sensors[i:i + SEQUENCE_LENGTH])
            y.append(unit_rul[i + SEQUENCE_LENGTH])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Improved LSTM Architecture:
    - 3 LSTM layers instead of 2 (captures deeper temporal patterns)
    - BatchNormalization (stabilizes training)
    - Stronger Dropout (better overfitting prevention)
    - Adam optimizer with tuned learning rate
    - ReduceLROnPlateau (auto adjusts learning rate)
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

        # Dense layers
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
    model.summary()
    return model

def train_lstm(X_train, y_train):
    print("\n🧠 Training Improved LSTM model...")
    print("""
    Improvements over basic LSTM:
    ✅ 3 LSTM layers (was 2) — deeper pattern learning
    ✅ BatchNormalization    — faster stable training
    ✅ Dropout 0.3/0.2/0.2  — stronger regularization
    ✅ Adam lr=0.001         — tuned learning rate
    ✅ ReduceLROnPlateau     — auto learning rate decay
    ✅ EarlyStopping p=7     — more patience (was 5)
    """)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

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

def plot_lstm_results(y_test, preds, history):
    os.makedirs('results', exist_ok=True)

    fig = plt.figure(figsize=(16, 10), facecolor='white')
    fig.suptitle('🧠  Improved LSTM — Results & Analysis',
                 fontsize=14, fontweight='bold', color='#2c3e50')
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.38)

    # ── 1. Actual vs Predicted ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test[:300], preds[:300],
                alpha=0.5, color='#8e44ad', s=20)
    ax1.plot([0,125],[0,125],'r--', linewidth=1.5,
             label='Perfect prediction')
    ax1.set_title('Actual vs Predicted RUL',
                  fontweight='bold')
    ax1.set_xlabel('Actual RUL')
    ax1.set_ylabel('Predicted RUL')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. Training loss ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history['loss'],
             color='#2980b9', linewidth=2,
             label='Train Loss')
    ax2.plot(history.history['val_loss'],
             color='#e74c3c', linewidth=2,
             linestyle='--', label='Val Loss')
    ax2.set_title('Training vs Validation Loss',
                  fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── 3. Architecture panel ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0,1); ax3.set_ylim(0,1)
    ax3.axis('off')
    ax3.set_title('LSTM Architecture',
                  fontweight='bold')
    from matplotlib.patches import FancyBboxPatch
    layers = [
        ('Input (30 cycles × 21 sensors)', '#3498db'),
        ('LSTM 128 + BatchNorm + Dropout 0.3', '#9b59b6'),
        ('LSTM 64  + BatchNorm + Dropout 0.2', '#8e44ad'),
        ('LSTM 32  + Dropout 0.2',             '#6c3483'),
        ('Dense 32 + Dense 16',                '#e67e22'),
        ('Output → RUL',                       '#e74c3c'),
    ]
    y_pos = 0.92
    for text, color in layers:
        box = FancyBboxPatch(
            (0.03, y_pos-0.10), 0.94, 0.09,
            boxstyle="round,pad=0.02",
            facecolor=color+'33',
            edgecolor=color, linewidth=1.5)
        ax3.add_patch(box)
        ax3.text(0.50, y_pos-0.055, text,
                 ha='center', va='center',
                 fontsize=7.5, color='#2c3e50')
        if y_pos > 0.25:
            ax3.annotate('',
                xy=(0.50, y_pos-0.12),
                xytext=(0.50, y_pos-0.10),
                arrowprops=dict(arrowstyle='->',
                                color='grey'))
        y_pos -= 0.155

    # ── 4. Residual plot ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    residuals = y_test - preds
    ax4.scatter(preds[:300], residuals[:300],
                alpha=0.4, color='#2980b9', s=15)
    ax4.axhline(y=0, color='red', linestyle='--',
                linewidth=1.5)
    ax4.set_title('Residual Plot\n(errors around zero = good)',
                  fontweight='bold')
    ax4.set_xlabel('Predicted RUL')
    ax4.set_ylabel('Residual (Actual - Predicted)')
    ax4.grid(True, alpha=0.3)

    # ── 5. Error distribution ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(residuals, bins=40,
             color='#8e44ad', alpha=0.7,
             edgecolor='black', linewidth=0.4)
    ax5.axvline(x=0, color='red', linestyle='--',
                linewidth=1.5, label='Zero error')
    ax5.axvline(x=residuals.mean(),
                color='orange', linestyle='--',
                linewidth=1.5,
                label=f'Mean: {residuals.mean():.2f}')
    ax5.set_title('Error Distribution\n(centered at 0 = unbiased)',
                  fontweight='bold')
    ax5.set_xlabel('Prediction Error')
    ax5.set_ylabel('Frequency')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── 6. Metrics summary ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0,1); ax6.set_ylim(0,1)
    ax6.axis('off')
    ax6.set_title('Performance Summary',
                  fontweight='bold')

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    from matplotlib.patches import FancyBboxPatch
    bg = FancyBboxPatch((0.03,0.03), 0.94, 0.94,
                        boxstyle="round,pad=0.04",
                        facecolor='#8e44ad22',
                        edgecolor='#8e44ad',
                        linewidth=2)
    ax6.add_patch(bg)

    metrics = [
        ('Model',     'Improved LSTM'),
        ('Layers',    '3 LSTM + 2 Dense'),
        ('RMSE',      f'{rmse:.2f} cycles'),
        ('MAE',       f'{mae:.2f} cycles'),
        ('R² Score',  f'{r2:.3f}'),
        ('Accuracy',  f'{r2*100:.1f}%'),
        ('Dropout',   '0.3 / 0.2 / 0.2'),
        ('Optimizer', 'Adam lr=0.001'),
        ('BatchNorm', '✅ Added'),
        ('LR Decay',  '✅ Added'),
    ]
    y = 0.90
    for label, val in metrics:
        ax6.text(0.08, y, f'{label}:',
                 fontsize=9, fontweight='bold',
                 color='#2c3e50')
        ax6.text(0.55, y, val,
                 fontsize=9, color='#8e44ad',
                 fontweight='bold')
        y -= 0.09

    plt.savefig('results/lstm_results.png',
                dpi=130, bbox_inches='tight')
    print('\n✅  Saved to results/lstm_results.png!')
    plt.show()

if __name__ == "__main__":
    print("Loading data...")
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)

    print("Creating sequences...")
    X, y = create_sequences(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    split    = int(0.8 * len(X))
    X_train  = X[:split]; X_test = X[split:]
    y_train  = y[:split]; y_test = y[split:]

    model, history = train_lstm(X_train, y_train)

    preds = model.predict(X_test, verbose=0).flatten()
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"\n🏆  Improved LSTM Results:")
    print(f"    RMSE : {rmse:.2f}")
    print(f"    MAE  : {mae:.2f}")
    print(f"    R²   : {r2:.3f} ({r2*100:.1f}% accuracy)")

    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_model.keras')
    print("✅  Model saved to models/lstm_model.keras!")

    plot_lstm_results(y_test, preds, history)