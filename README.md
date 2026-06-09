# Energy Consumption Anomaly Prediction
### Python in Electrical Engineering – projekt ML

## Struktura projektu

```
energy_project/
├── data/
│   └── household_power_consumption.csv   ← tu wrzuć dataset
├── src/
│   ├── __init__.py
│   ├── features.py      # Krok 1: inżynieria cech
│   ├── models.py        # Kroki 3, 4, 5: Ridge, XGBoost, LSTM
│   └── evaluation.py    # Krok 6: metryki i wykresy
├── reports/             # tu lądują wykresy (auto-tworzony)
├── main.py              # Kroki 2–6: główny pipeline
├── requirements.txt
└── README.md
```

## Dataset

UCI Household Power Consumption (minutowy):
- `Global_active_power` [kW] – główny target prognozy
- `Global_reactive_power`, `Voltage`, `Global_intensity`
- `Sub_metering_1/2/3`

## Instalacja zależności

```bash
pip install -r requirements.txt
```

## Uruchomienie

```bash
python main.py
```

## Modele

| Model   | Typ         | Rola                          |
|---------|-------------|-------------------------------|
| Ridge   | Liniowy L2  | Baseline / punkt odniesienia  |
| XGBoost | Drzewiasty  | Główny model produkcyjny      |
| LSTM    | Sieć RNN    | Model sekwencyjny (deep)      |

## Podział danych

- **80% train / 20% test** – chronologicznie (bez shuffle)
- Skalowanie: `RobustScaler` (odporny na anomalie/piki)

## Wyniki

- Metryki MAE / RMSE / MAPE drukowane w konsoli
- Wykresy zapisywane w `reports/`:
  - `forecast_comparison.png` – szereg czasowy prognoz vs rzeczywistości
  - `metrics_comparison.png`  – porównanie metryk słupkami
