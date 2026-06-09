import os
import sys
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.features   import load_and_resample, prepare_features_and_targets
from src.models     import (
    build_ridge_model,
    build_xgboost_model,
    reshape_for_lstm,
    build_lstm_model,
)
from src.evaluation import evaluate_models, plot_predictions


# ─── Konfiguracja ─────────────────────────────────────────────────────────────
DATA_PATH      = 'data/household_power_consumption.csv'
LOOKBACK_HOURS = 24    # okno historyczne – ile godzin wstecz jako lagi
FORECAST_HOURS = 24    # horyzont prognozy – ile godzin naprzód
TRAIN_RATIO    = 0.80  # 80% train, 20% test (chronologicznie)
LSTM_EPOCHS    = 30
BATCH_SIZE     = 128


def main() -> None:
    if not os.path.isfile(DATA_PATH):
        print(f"\n[BŁĄD] Nie znaleziono pliku danych: '{DATA_PATH}'")
        print("  Umieść plik CSV w katalogu data/ i uruchom ponownie.")
        sys.exit(1)

    # ─── Krok 1+2: Cechy i przygotowanie potoku danych ────────────────────
    print("\n" + "─" * 60)
    print(" KROK 1+2 – Wczytanie danych, inżynieria cech, skalowanie")
    print("─" * 60)

    df = load_and_resample(DATA_PATH)
    X, y = prepare_features_and_targets(df, LOOKBACK_HOURS, FORECAST_HOURS)

    # Chronologiczny podział – BEZ mieszania kolejności!
    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Podział: Train {X_train.shape[0]:,} próbek  |  Test {X_test.shape[0]:,} próbek")

    # RobustScaler – neutralizuje wpływ outlierów (anomalie zużycia energii)
    scaler      = RobustScaler()
    X_train_sc  = scaler.fit_transform(X_train)   # fit tylko na train!
    X_test_sc   = scaler.transform(X_test)
    print("  RobustScaler zastosowany.")

    # ─── Krok 3: Ridge (baseline) ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" KROK 3 – Ridge (baseline, MultiOutput L2)")
    print("─" * 60)

    ridge = build_ridge_model()
    ridge.fit(X_train_sc, y_train)
    y_pred_ridge = ridge.predict(X_test_sc)
    print(f"  Trenowanie Ridge – zakończone. "
          f"Output shape: {y_pred_ridge.shape}")

    # ─── Krok 4: XGBoost ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" KROK 4 – XGBoost (MultiOutput Gradient Boosting)")
    print("─" * 60)

    xgb_model    = build_xgboost_model()
    xgb_model.fit(X_train_sc, y_train)
    y_pred_xgb   = xgb_model.predict(X_test_sc)
    print(f"  Trenowanie XGBoost – zakończone. "
          f"Output shape: {y_pred_xgb.shape}")

    # ─── Krok 5: LSTM ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" KROK 5 – LSTM (reshape 3D + trening sieci neuronowej)")
    print("─" * 60)

    X_train_3d = reshape_for_lstm(X_train_sc, LOOKBACK_HOURS)
    X_test_3d  = reshape_for_lstm(X_test_sc,  LOOKBACK_HOURS)
    print(f"  Reshape: {X_train_sc.shape} → {X_train_3d.shape}  "
          f"[samples, timesteps={LOOKBACK_HOURS}h, features=1]")

    input_shape = (X_train_3d.shape[1], X_train_3d.shape[2])
    lstm_model  = build_lstm_model(input_shape, FORECAST_HOURS)

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
        verbose=1,
    )
    y_pred_lstm = lstm_model.predict(X_test_3d, verbose=0)
    print(f"  Trenowanie LSTM – zakończone. "
          f"Output shape: {y_pred_lstm.shape}")

    # ─── Krok 6: Ewaluacja i wykresy ───────────────────────────────────────
    print("\n" + "─" * 60)
    print(" KROK 6 – Ewaluacja i porównanie modeli")
    print("─" * 60)

    predictions = {
        'Ridge':   y_pred_ridge,
        'XGBoost': y_pred_xgb,
        'LSTM':    y_pred_lstm,
    }
    evaluate_models(y_test, predictions)
    plot_predictions(y_test, predictions, save_dir='reports', n_days=7)

    print("\n✓ Pipeline zakończony pomyślnie.\n")


if __name__ == '__main__':
    main()
