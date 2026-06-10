# Energy Consumption Multi-Step Forecasting
### Python in Electrical Engineering – Machine Learning Project

This project focuses on predicting multi-step household electricity consumption using machine learning and deep learning approaches. It aggregates raw minute-by-minute consumption data into hourly intervals, engineers cyclic calendar features and autoregressive lags, and compares three distinct modeling paradigms to forecast energy loads up to 24 hours ahead.

---

## Project Goal

The primary objective is to build a data pipeline capable of forecasting **Global Active Power** over a 24-hour horizon. Accurate load forecasting is essential in Electrical Engineering for optimizing smart grid operations, peak-shaving strategies, energy storage management, and identifying demand-side anomalies.

---

## Repository Structure

```
├── README.md           # project documentation and setup guide
├── data                # data directory 
│   ├── processed            # aggregated hourly clean CSV
│   │   └── hourly_clean.csv
│   └── raw                  # raw txt file
│       └── household_power_consumption.txt
├── main.py             # main pipeline execution script
├── notebooks           # research
│   └── 01_eda_and_anomalies.ipynb
├── reports             # plots and artifacts
│   ├── forecast_comparison.png
│   └── metrics_comparison.png
├── requirements.txt    # python environment dependencies
└── src
    ├── config.py            # global constants and paths
    ├── data_loader.py       # raw data consumtion and hourly resampling
    ├── evaluation.py        # custom metric math
    ├── features.py          # calendar encoding and time-series lags
    └── models.py            # models definition
```

---

## Dataset & Reference Links

The pipeline is pre-configured to use **Individual Household Power Consumption Dataset**. 
* **Official Kaggle Link:** [UCI Household Power Consumption Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

### Target & Features:
* **`Global_active_power` [kW]** – The primary forecasting target.
* **Sensors & Electrical Metrics:** `Global_reactive_power`, `Voltage`, `Global_intensity`
* **Sub-meterings:** `Sub_metering_1` (Kitchen), `Sub_metering_2` (Laundry), `Sub_metering_3` (Climate Control)

---

## How to Use & Get Started

### 1. Installation
Clone the repository, set up a virtual environment, and install the required dependencies:
```bash
# activate your virtual environment (.venv) first, then run:
pip install -r requirements.txt
```

### 2. Place the Data
Create the directory structure data/raw/ if it doesn't exist yet, download the dataset, and place the unzipped raw file inside it:
```bash
data/raw/household_power_consumption.txt
```

### 3. Execution

```bash
# activate your virtual environment (.venv) first, then run:
python main.py
```

## Models

| Model   | Typ                      | Rola                 |
|---------|--------------------------|----------------------|
| Ridge   | Linear L2 Regression     | Baseline / benchmark |
| XGBoost | Gradient Boosted Trees   | Main tabular model   |
| LSTM    | Recurrent Neural Network | 3D sequenced model   |

## Results

- MAE / RMSE / MAPE metrics printed in console
- Visual Reports saved inside `reports/`:
  - `forecast_comparison.png` – time series of forecasts vs reality
  - `metrics_comparison.png`  – comparison of metrics
