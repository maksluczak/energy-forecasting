from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = str(BASE_DIR / "data" / "raw" / "household_power_consumption.txt")
PROCESSED_DATA_PATH = str(BASE_DIR / "data" / "processed" / "hourly_clean.csv")
SAVE_DIR = str(BASE_DIR / "reports")
LOOKBACK_HOURS = 24
FORECAST_HOURS = 24
TRAIN_RATIO = 0.80
LSTM_EPOCHS = 30
BATCH_SIZE = 128