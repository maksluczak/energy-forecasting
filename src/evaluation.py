from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import SAVE_DIR

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # standard mean absolute error - average of absolute differences
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # root mean squared error - penalizes bigger mistakes harder
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    # percentage error - using mask to skip zero values and avoid division by zero crashes
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def _compute_metrics(y_true: np.ndarray, predictions_dict: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    # helper function to loop through all models and calculate MAE, RMSE, MAPE at once
    results: dict[str, dict[str, float]] = {}
    for name, y_pred in predictions_dict.items():
        results[name] = {
            'MAE':  mae(y_true, y_pred),
            'RMSE': rmse(y_true, y_pred),
            'MAPE': mape(y_true, y_pred),
        }
    return results

def evaluate_models(y_true: np.ndarray, predictions_dict: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    # get scores and print a terminal table to compare our models
    results = _compute_metrics(y_true, predictions_dict)

    print("\n" + "═" * 62)
    print("            EVALUATION METRICS – test set")
    print("═" * 62)
    print(f"  {'Model':<12}  {'MAE':>10}  {'RMSE':>10}  {'MAPE (%)':>10}")
    print("  " + "─" * 58)
    for name, m in results.items():
        print(f"  {name:<12}  {m['MAE']:>10.4f}  {m['RMSE']:>10.4f}  {m['MAPE']:>10.2f}")
    print("═" * 62)

    return results

def plot_predictions(y_true: np.ndarray, predictions_dict: dict[str, np.ndarray], n_days: int = 7) -> None:
    # creates plots folder and prepares data for the charts
    os.makedirs(SAVE_DIR, exist_ok=True)

    # limit data to show only a specific number of days on the plot (default: 7 days)
    n_show = min(n_days * 24, len(y_true))
    hours = np.arange(n_show)
    true_h1 = y_true[:n_show, 0]   # target for the next 1 hour ahead

    palette = {
        'Ridge':   '#e74c3c',
        'XGBoost': '#3498db',
        'LSTM':    '#2ecc71',
    }
    default_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    # chart 1: time series for 1-hour ahead prediction
    fig1, ax1 = plt.subplots(figsize=(16, 5))

    # plot the actual real energy consumption
    ax1.plot(hours, true_h1, color='black', linewidth=1.8, label='Actual', zorder=5)

    # loop and draw prediction lines for each model
    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        color = palette.get(name, default_colors[i % len(default_colors)])
        pred_h1 = y_pred[:n_show, 0]
        ax1.plot(hours, pred_h1, color=color, linewidth=1.0, alpha=0.85, linestyle='--', label=name)

    # add vertical grid lines every 24 hours to separate days visually
    for d in range(0, n_show, 24):
        ax1.axvline(d, color='gray', linewidth=0.4, alpha=0.4)

    ax1.set_title(f'Energy Consumption Forecast – Horizon h=1h ({n_days} Test Days)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Hour (Relative to Test Set Start)')
    ax1.set_ylabel('Global Active Power [kW]')
    ax1.legend(framealpha=0.9, loc='upper right')
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax1.xaxis.set_minor_locator(mticker.MultipleLocator(6))
    plt.tight_layout()

    # save the time series comparison plot to reports folder
    path1 = os.path.join(SAVE_DIR, 'forecast_comparison.png')
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    # chart 2: bar plots comparing the metrics
    results = _compute_metrics(y_true, predictions_dict)
    names = list(results.keys())
    mae_vals = [results[n]['MAE']  for n in names]
    rmse_vals = [results[n]['RMSE'] for n in names]
    mape_vals = [results[n]['MAPE'] for n in names]

    x = np.arange(len(names))
    width = 0.28

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle('Model Comparison – Error Metrics', fontsize=14, fontweight='bold')

    # left plot: MAE and RMSE bars
    b1 = ax2a.bar(x - width/2, mae_vals,  width, label='MAE',  color='#3498db', alpha=0.85)
    b2 = ax2a.bar(x + width/2, rmse_vals, width, label='RMSE', color='#e74c3c', alpha=0.85)
    ax2a.bar_label(b1, fmt='%.3f', padding=3, fontsize=9)
    ax2a.bar_label(b2, fmt='%.3f', padding=3, fontsize=9)
    ax2a.set_xticks(x)
    ax2a.set_xticklabels(names, fontsize=11)
    ax2a.set_ylabel('Error [kW]')
    ax2a.set_title('MAE & RMSE (Lower is Better)')
    ax2a.legend()
    ax2a.grid(axis='y', alpha=0.3)

    # right plot: MAPE percentage bar chart using matching model colors
    bar_colors = [palette.get(n, default_colors[i]) for i, n in enumerate(names)]
    b3 = ax2b.bar(x, mape_vals, width * 2, color=bar_colors, alpha=0.85)
    ax2b.bar_label(b3, fmt='%.1f%%', padding=3, fontsize=9)
    ax2b.set_xticks(x)
    ax2b.set_xticklabels(names, fontsize=11)
    ax2b.set_ylabel('MAPE [%]')
    ax2b.set_title('MAPE (Percentage Error)')
    ax2b.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    # save the metrics comparison plot to reports folder
    path2 = os.path.join(SAVE_DIR, 'metrics_comparison.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)