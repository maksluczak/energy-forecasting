from src.data_loader import load_clean_data
from src.features import prepare_features, prepare_forecast

df = load_clean_data()

X_raw = prepare_features(df, lookback_hours = [1, 2, 3, 24])
y_raw = prepare_forecast(df, forecast_hours = [1, 2, 3])

common_index = X_raw.index.intersection(y_raw.index)
X = X_raw.loc[common_index].drop(columns = ['Global_active_power', 'hour'])
y = y_raw.loc[common_index]