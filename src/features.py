import numpy as np
import pandas as pd


def prepare_features(df, lookback_hours):
    df_features = df.copy()

    df_features['day_of_week'] = df_features.index.dayofweek

    df_features['hour'] = df_features.index.hour
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24.0)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24.0)

    for l_hour in lookback_hours:
        df_features[f'lag_{l_hour}h'] = df_features['Global_active_power'].shift(l_hour)

    df_features.dropna(inplace=True)
    return df_features


def prepare_forecast(df, forecast_hours):
    df_forecast = pd.DataFrame(index=df.index)

    for f_hour in forecast_hours:
        df_forecast[f'fcast_+{f_hour}h'] = df['Global_active_power'].shift(-f_hour)

    df_forecast.dropna(inplace=True)
    return df_forecast