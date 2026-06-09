"""
src/models.py
=============
Model Ridge (regresja liniowa L2, baseline)
Model XGBoost (gradient boosting, odporny na anomalie)
Reshape do 3D i architektura LSTM (Keras/TensorFlow)

Wszystkie modele obsługują prognozę wielokrokową (Multi-Output).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Liczba cech niebędących lagami – musi zgadzać się z src/features.py
N_CALENDAR = 7   # hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend
N_SENSOR   = 6   # reactive_power, voltage, intensity, sub1, sub2, sub3
N_OTHER    = N_CALENDAR + N_SENSOR   # = 13


# ══════════════════════════════════════════════════════════════════════════════
# Krok 3: Ridge (baseline)
# ══════════════════════════════════════════════════════════════════════════════

def build_ridge_model(alpha: float = 1.0) -> MultiOutputRegressor:
    """
    Regresja Ridge (L2) opakowana w MultiOutputRegressor.

    MultiOutputRegressor trenuje osobny model Ridge dla każdego
    z forecast_hours kroków naprzód – daje solidny punkt odniesienia.

    Parameters
    ----------
    alpha : float
        Siła regularyzacji L2 (domyślnie 1.0).

    Returns
    -------
    MultiOutputRegressor(Ridge)
    """
    return MultiOutputRegressor(
        Ridge(alpha=alpha, max_iter=2000),
        n_jobs=-1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Krok 4: XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def build_xgboost_model(
    n_estimators:    int   = 200,
    max_depth:       int   = 6,
    learning_rate:   float = 0.05,
    subsample:       float = 0.8,
    colsample_bytree: float = 0.8,
    random_state:    int   = 42,
) -> MultiOutputRegressor:

    base = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        tree_method='hist',
        verbosity=0,
        n_jobs=-1,
    )
    # n_jobs=1 w MultiOutputRegressor – XGB sam jest wielowątkowy (n_jobs=-1 w base)
    return MultiOutputRegressor(base, n_jobs=1)


# ══════════════════════════════════════════════════════════════════════════════
# Krok 5a: Reshape do formatu 3D dla LSTM
# ══════════════════════════════════════════════════════════════════════════════

def reshape_for_lstm(X: np.ndarray, lookback_hours: int) -> np.ndarray:

    lag_cols = X[:, N_OTHER: N_OTHER + lookback_hours]   # [samples, lookback]
    X_3d     = lag_cols[:, :, np.newaxis]                # [samples, lookback, 1]
    return X_3d.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Krok 5b: Architektura sieci LSTM
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_model(input_shape: tuple, output_size: int):

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=True, name='lstm_1'),
            Dropout(0.2, name='drop_1'),
            LSTM(32, return_sequences=False, name='lstm_2'),
            Dropout(0.2, name='drop_2'),
            Dense(output_size, name='output'),
        ],
        name='LSTM_energy_forecast',
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae'],
    )
    model.summary()
    return model
