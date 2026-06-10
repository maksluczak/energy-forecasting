import os
import sys
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, LOOKBACK_HOURS, FORECAST_HOURS, TRAIN_RATIO, LSTM_EPOCHS, BATCH_SIZE
from src.data_loader import load_clean_data
from src.features import prepare_features_and_targets
from src.models import (build_ridge_model, build_xgboost_model, reshape_for_lstm, build_lstm_model)
from src.evaluation import evaluate_models, plot_predictions

def main() -> None:
    if not os.path.exists(RAW_DATA_PATH) and not os.path.exists(PROCESSED_DATA_PATH):
        print(f"\nData not found...")
        sys.exit(1)

    df = load_clean_data()

    X, y = prepare_features_and_targets(df, LOOKBACK_HOURS, FORECAST_HOURS)

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    ridge = build_ridge_model()
    ridge.fit(X_train_sc, y_train)
    y_pred_ridge = ridge.predict(X_test_sc)

    xgb_model = build_xgboost_model()
    xgb_model.fit(X_train_sc, y_train)
    y_pred_xgb = xgb_model.predict(X_test_sc)

    X_train_3d = reshape_for_lstm(X_train_sc, LOOKBACK_HOURS)
    X_test_3d = reshape_for_lstm(X_test_sc, LOOKBACK_HOURS)

    input_shape = (X_train_3d.shape[1], X_train_3d.shape[2])
    lstm_model = build_lstm_model(input_shape, FORECAST_HOURS)

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-5, verbose=1,
        ),
    ]

    lstm_model.fit(
        X_train_3d, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.10,
        callbacks=callbacks,
        verbose=1
    )
    y_pred_lstm = lstm_model.predict(X_test_3d, verbose=0)

    predictions = {
        'Ridge': y_pred_ridge,
        'XGBoost': y_pred_xgb,
        'LSTM': y_pred_lstm,
    }

    evaluate_models(y_test, predictions)
    plot_predictions(y_test, predictions, n_days=7)

if __name__ == '__main__':
    main()