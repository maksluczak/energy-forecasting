import numpy as np
import pandas as pd

def prepare_features_and_targets(df: pd.DataFrame, lookback_hours: int = 24, forecast_hours: int = 24) -> tuple[np.ndarray, np.ndarray]:
    d = df.copy()
    target_col = 'Global_active_power'

    hour = d.index.hour
    dow = d.index.dayofweek
    month = d.index.month

    # cyclic coding for hours, days of week and months (converting e.g. 0-23 to circle coordinates)
    d['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    d['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    d['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    d['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    d['month_sin'] = np.sin(2 * np.pi * month / 12)
    d['month_cos'] = np.cos(2 * np.pi * month / 12)
    d['is_weekend'] = (dow >= 5).astype(np.float32)

    # generate lags (past history features)
    lag_dict = {f'lag_{lag}': d[target_col].shift(lag) for lag in range(1, lookback_hours + 1)}
    # generate targets (future forecast horizon)
    target_dict = {f'target_h{h}': d[target_col].shift(-h) for h in range(1, forecast_hours + 1)}

    d = pd.concat([d, pd.DataFrame(lag_dict, index=d.index), pd.DataFrame(target_dict, index=d.index)], axis=1)
    d.dropna(inplace=True)

    calendar_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'is_weekend']
    sensor_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    lag_cols = [f'lag_{i}' for i in range(1, lookback_hours + 1)]
    target_cols = [f'target_h{h}' for h in range(1, forecast_hours + 1)]

    # extract final sliding window matrices for ML models
    X_raw = d[calendar_cols + sensor_cols + lag_cols].values.astype(np.float32)
    y_raw = d[target_cols].values.astype(np.float32)

    return X_raw, y_raw