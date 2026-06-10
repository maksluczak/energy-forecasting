from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import SAVE_DIR

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # mean absolute error
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # root mean squared error
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    # mean absolute percentage error
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def _compute_metrics(y_true: np.ndarray, predictions_dict: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, y_pred in predictions_dict.items():
        results[name] = {
            'MAE':  mae(y_true, y_pred),
            'RMSE': rmse(y_true, y_pred),
            'MAPE': mape(y_true, y_pred),
        }
    return results

def evaluate_models(y_true: np.ndarray, predictions_dict: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    results = _compute_metrics(y_true, predictions_dict)

    print("\n" + "═" * 62)
    print("   METRYKI EWALUACJI – zbiór testowy (wszystkie horyzonty)")
    print("═" * 62)
    print(f"  {'Model':<12}  {'MAE':>10}  {'RMSE':>10}  {'MAPE (%)':>10}")
    print("  " + "─" * 58)
    for name, m in results.items():
        print(f"  {name:<12}  {m['MAE']:>10.4f}  {m['RMSE']:>10.4f}  {m['MAPE']:>10.2f}")
    print("═" * 62)

    return results

def plot_predictions(y_true: np.ndarray, predictions_dict: dict[str, np.ndarray], n_days: int = 7) -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Ograniczamy do n_days pełnych dni
    n_show    = min(n_days * 24, len(y_true))
    hours     = np.arange(n_show)
    true_h1   = y_true[:n_show, 0]   # wartości h=1 (następna godzina)

    palette = {
        'Ridge':   '#e74c3c',
        'XGBoost': '#3498db',
        'LSTM':    '#2ecc71',
    }
    default_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    # ── Wykres 1: szereg czasowy h=1 ──────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(16, 5))

    ax1.plot(hours, true_h1, color='black', linewidth=1.8,
             label='Rzeczywiste', zorder=5)

    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        color    = palette.get(name, default_colors[i % len(default_colors)])
        pred_h1  = y_pred[:n_show, 0]
        ax1.plot(hours, pred_h1, color=color, linewidth=1.0,
                 alpha=0.85, linestyle='--', label=name)

    # Zaznacz doby pionowymi liniami
    for d in range(0, n_show, 24):
        ax1.axvline(d, color='gray', linewidth=0.4, alpha=0.4)

    ax1.set_title(
        f'Prognoza zużycia energii – horyzont h=1 h  ({n_days} dni testowych)',
        fontsize=13, fontweight='bold',
    )
    ax1.set_xlabel('Godzina (względna od początku zbioru testowego)')
    ax1.set_ylabel('Global Active Power [kW]')
    ax1.legend(framealpha=0.9, loc='upper right')
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax1.xaxis.set_minor_locator(mticker.MultipleLocator(6))
    plt.tight_layout()

    path1 = os.path.join(SAVE_DIR, 'forecast_comparison.png')
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    # ── Wykres 2: porównanie metryk (MAE / RMSE) ───────────────────────────
    results = _compute_metrics(y_true, predictions_dict)
    names     = list(results.keys())
    mae_vals  = [results[n]['MAE']  for n in names]
    rmse_vals = [results[n]['RMSE'] for n in names]
    mape_vals = [results[n]['MAPE'] for n in names]

    x     = np.arange(len(names))
    width = 0.28

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle('Porównanie modeli – metryki błędu', fontsize=14, fontweight='bold')

    # MAE i RMSE
    b1 = ax2a.bar(x - width/2, mae_vals,  width, label='MAE',  color='#3498db', alpha=0.85)
    b2 = ax2a.bar(x + width/2, rmse_vals, width, label='RMSE', color='#e74c3c', alpha=0.85)
    ax2a.bar_label(b1, fmt='%.3f', padding=3, fontsize=9)
    ax2a.bar_label(b2, fmt='%.3f', padding=3, fontsize=9)
    ax2a.set_xticks(x)
    ax2a.set_xticklabels(names, fontsize=11)
    ax2a.set_ylabel('Błąd [kW]')
    ax2a.set_title('MAE i RMSE')
    ax2a.legend()
    ax2a.grid(axis='y', alpha=0.3)

    # MAPE
    bar_colors = [palette.get(n, default_colors[i]) for i, n in enumerate(names)]
    b3 = ax2b.bar(x, mape_vals, width * 2, color=bar_colors, alpha=0.85)
    ax2b.bar_label(b3, fmt='%.1f%%', padding=3, fontsize=9)
    ax2b.set_xticks(x)
    ax2b.set_xticklabels(names, fontsize=11)
    ax2b.set_ylabel('MAPE [%]')
    ax2b.set_title('MAPE')
    ax2b.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(SAVE_DIR, 'metrics_comparison.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)