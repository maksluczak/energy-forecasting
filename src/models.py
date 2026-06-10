import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

from src.config import N_OTHER

def build_ridge_model(alpha: float = 1.0) -> MultiOutputRegressor:
    return MultiOutputRegressor(
        Ridge(alpha=alpha, max_iter=2000),
        n_jobs=-1,
    )

def build_xgboost_model(n_estimators: int = 200, max_depth: int = 6, learning_rate: float = 0.05, subsample: float = 0.8, colsample_bytree: float = 0.8, random_state: int = 42) -> MultiOutputRegressor:
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
    return MultiOutputRegressor(base, n_jobs=-1)

def reshape_for_lstm(X: np.ndarray, lookback_hours: int) -> np.ndarray:
    num_samples = X.shape[0]
    num_features_per_step = N_OTHER + 1

    X_3d = np.zeros((num_samples, lookback_hours, num_features_per_step), dtype=np.float32)
    static_features = X[:, :N_OTHER]

    for t in range(lookback_hours):
        lag_val = X[:, N_OTHER + (lookback_hours - 1 - t): N_OTHER + (lookback_hours - t)]
        X_3d[:, t, :] = np.hstack([static_features, lag_val])

    return X_3d


def build_lstm_model(input_shape: tuple, output_size: int) -> Sequential:
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
    return model