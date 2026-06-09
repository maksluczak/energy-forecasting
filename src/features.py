"""
Inżynieria Cech (Feature Engineering)
"""
import numpy as np
import pandas as pd


def load_and_resample(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=',',
        na_values=['?', 'NA', '', ' '],
        low_memory=False,
    )
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], dayfirst=True
    )
    df = df.drop(columns=['Date', 'Time'])
    df = df.set_index('datetime')
    df = df.apply(pd.to_numeric, errors='coerce')

    df_hourly = df.resample('h').mean()
    df_hourly = df_hourly.dropna(subset=['Global_active_power'])

    print(f"  Załadowano {len(df_hourly):,} godzinnych próbek  "
          f"({df_hourly.index[0].date()} → {df_hourly.index[-1].date()})")
    return df_hourly


def prepare_features_and_targets(
    df: pd.DataFrame,
    lookback_hours: int = 24,
    forecast_hours: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    d = df.copy()
    target_col = 'Global_active_power'

    hour  = d.index.hour
    dow   = d.index.dayofweek
    month = d.index.month

    d['hour_sin']   = np.sin(2 * np.pi * hour  / 24)
    d['hour_cos']   = np.cos(2 * np.pi * hour  / 24)
    d['dow_sin']    = np.sin(2 * np.pi * dow   / 7)
    d['dow_cos']    = np.cos(2 * np.pi * dow   / 7)
    d['month_sin']  = np.sin(2 * np.pi * month / 12)
    d['month_cos']  = np.cos(2 * np.pi * month / 12)
    d['is_weekend'] = (dow >= 5).astype(np.float32)

    for lag in range(1, lookback_hours + 1):
        d[f'lag_{lag}'] = d[target_col].shift(lag)

    for h in range(1, forecast_hours + 1):
        d[f'target_h{h}'] = d[target_col].shift(-h)

    d = d.dropna()

    calendar_cols = ['hour_sin','hour_cos','dow_sin','dow_cos',
                     'month_sin','month_cos','is_weekend']
    sensor_cols   = ['Global_reactive_power','Voltage','Global_intensity',
                     'Sub_metering_1','Sub_metering_2','Sub_metering_3']
    lag_cols      = [f'lag_{i}'     for i in range(1, lookback_hours + 1)]
    target_cols   = [f'target_h{h}' for h in range(1, forecast_hours + 1)]

    X = d[calendar_cols + sensor_cols + lag_cols].values.astype(np.float32)
    y = d[target_cols].values.astype(np.float32)

    print(f"  Macierz cech X: {X.shape}  |  Target y: {y.shape}")
    return X, y