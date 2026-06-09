import numpy as np

def prepare_features(df, lookback_hours):
    df_features = df.copy()

    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.day.dayofweek

    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24.0)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24.0)

    for l_hour in lookback_hours:
        df_features[f'lag_{l_hour}'] = df_features['Global_active_power'].shift(l_hour)

    df_features.dropna(inplace=True)

    return df_features