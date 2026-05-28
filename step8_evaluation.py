import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ── Style ────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.grid": True,
    "grid.color": "#e0e0e0", "grid.linestyle": "--", "grid.linewidth": 0.5,
    "font.family": "sans-serif", "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
}
plt.rcParams.update(STYLE)

# ── Load data and training model ─────────────────────────────────────────────
df    = pd.read_csv("cpi_data.csv", index_col="date", parse_dates=True)
pi    = df["pi"].dropna()
train = pi[pi.index <= "2022-12-31"]
test  = pi[pi.index >  "2022-12-31"]      # 24 months: Jan 2023 – Dec 2024

with open("best_model.pkl", "rb") as f:
    sarima_res = pickle.load(f)

print(f"Training obs   : {len(train)}  ({train.index.min().date()} – {train.index.max().date()})")
print(f"Holdout obs    : {len(test)}  ({test.index.min().date()} – {test.index.max().date()})")

# ══════════════════════════════════════════════════════════════════════════════
# 1. SARIMA holdout predictions (genuine out-of-sample, no refit)
# ══════════════════════════════════════════════════════════════════════════════
sarima_fc   = sarima_res.get_forecast(steps=len(test))
sarima_pred = sarima_fc.predicted_mean.values   # shape (24,)

# ══════════════════════════════════════════════════════════════════════════════
# 2. ETS (Holt-Winters additive) benchmark
# ══════════════════════════════════════════════════════════════════════════════
ets_mod = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated",
)
ets_res  = ets_mod.fit(optimized=True, use_brute=True)
ets_pred = ets_res.forecast(len(test)).values   # shape (24,)

print(f"\nETS smoothing params: alpha={ets_res.params['smoothing_level']:.4f}  "
      f"beta={ets_res.params['smoothing_trend']:.4f}  "
      f"gamma={ets_res.params['smoothing_seasonal']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Performance metrics
# ══════════════════════════════════════════════════════════════════════════════
actual = test.values

sarima_mae  = np.mean(np.abs(actual - sarima_pred))
sarima_rmse = np.sqrt(np.mean((actual - sarima_pred) ** 2))
ets_mae     = np.mean(np.abs(actual - ets_pred))
ets_rmse    = np.sqrt(np.mean((actual - ets_pred) ** 2))

mae_pct_improvement  = (ets_mae  - sarima_mae)  / ets_mae  * 100
rmse_pct_improvement = (ets_rmse - sarima_rmse) / ets_rmse * 100

print()
print("=" * 75)
print("PERFORMANCE EVALUATION  (holdout: Jan 2023 – Dec 2024, 24 months)")
print("=" * 75)
hdr = f"{'Model':<20} | {'MAE':>8} | {'RMSE':>8} | {'MAE vs ETS':>12} | {'RMSE vs ETS':>13}"
print(hdr)
print("-" * 75)
print(f"{'SARIMA (selected)':<20} | {sarima_mae*100:>8.4f}% | {sarima_rmse*100:>8.4f}% | "
      f"{mae_pct_improvement:>+10.2f}% | {rmse_pct_improvement:>+11.2f}%")
print(f"{'ETS (Holt-Winters)':<20} | {ets_mae*100:>8.4f}% | {ets_rmse*100:>8.4f}% | "
      f"{'baseline':>12} | {'baseline':>13}")
print("=" * 75)

if sarima_mae < ets_mae:
    print(f"\nConclusion: SARIMA outperforms ETS by {abs(mae_pct_improvement):.2f}% on MAE "
          f"and {abs(rmse_pct_improvement):.2f}% on RMSE over the holdout.")
else:
    print(f"\nConclusion: ETS outperforms SARIMA by {abs(mae_pct_improvement):.2f}% on MAE "
          f"and {abs(rmse_pct_improvement):.2f}% on RMSE over the holdout.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOT 12 — Holdout comparison
# ══════════════════════════════════════════════════════════════════════════════
# Extend plot window: show some training tail for context
context_start = "2021-01-01"
obs_context   = pi[pi.index >= context_start]

fig, ax = plt.subplots(figsize=(13, 5))

# Training tail (context)
train_tail = train[train.index >= context_start]
ax.plot(train_tail.index, train_tail.values * 100,
        color="#2c3e50", linewidth=1.8, label="Observed $\\pi_t$ (training tail)")

# Actual holdout
ax.plot(test.index, actual * 100,
        color="#2c3e50", linewidth=1.8, linestyle="-",
        marker="o", markersize=5, label="Observed $\\pi_t$ (holdout)")

# SARIMA predictions
ax.plot(test.index, sarima_pred * 100,
        color="#2980b9", linewidth=1.6, linestyle="--",
        marker="s", markersize=4,
        label=f"SARIMA(1,0,1)(1,0,1)₁₂  [MAE={sarima_mae*100:.3f}%]")

# ETS predictions
ax.plot(test.index, ets_pred * 100,
        color="#e74c3c", linewidth=1.6, linestyle="--",
        marker="^", markersize=4,
        label=f"ETS Holt-Winters         [MAE={ets_mae*100:.3f}%]")

# Reference lines
ax.axvline(pd.Timestamp("2023-01-01"), color="#555555", linestyle=":",
           linewidth=1.2, label="Start of holdout (Jan 2023)")
ax.axhline(2.0, color="#27ae60", linestyle="--", linewidth=1.1, alpha=0.8,
           label="Fed 2% target")

ax.set_title(
    "Holdout Period Forecast Comparison — SARIMA vs ETS (Holt-Winters)\n"
    f"Jan 2023 – Dec 2024  |  SARIMA MAE={sarima_mae*100:.3f}%  |  ETS MAE={ets_mae*100:.3f}%",
    pad=8
)
ax.set_xlabel("Date")
ax.set_ylabel(r"Year-over-Year Inflation Rate $\pi_t$ (%)")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.set_xlim(pd.Timestamp("2021-01-01"), test.index.max())
ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
fig.tight_layout()
fig.savefig("plot_12_holdout_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: plot_12_holdout_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Monthly error breakdown + top-3 absolute errors
# ══════════════════════════════════════════════════════════════════════════════
errors = actual - sarima_pred
abs_errors = np.abs(errors)

print()
print("=" * 72)
print("SARIMA FORECAST ERRORS — Monthly Breakdown (Jan 2023 – Dec 2024)")
print("=" * 72)
print(f"{'Date':<10} | {'Actual (%)':>11} | {'Forecast (%)':>13} | {'Error (pp)':>11} | {'|Error|':>8}")
print("-" * 72)
for d, obs, pred, err, ae in zip(test.index, actual*100, sarima_pred*100, errors*100, abs_errors*100):
    flag = " <--" if ae >= sorted(abs_errors*100, reverse=True)[2] else ""
    print(f"{d.strftime('%Y-%m'):<10} | {obs:>11.4f} | {pred:>13.4f} | {err:>+11.4f} | {ae:>8.4f}{flag}")
print("=" * 72)

# Top-3
top3_idx = np.argsort(abs_errors)[::-1][:3]
top3_dates  = test.index[top3_idx]
top3_errors = errors[top3_idx]
top3_abs    = abs_errors[top3_idx]

economic_notes = {
    "2023-09": "Fed funds rate at 5.25-5.50% (22-year high); sticky services inflation",
    "2023-10": "Persistent shelter/services inflation; market re-pricing of 'higher for longer'",
    "2023-11": "Continued shelter inflation lag; OER contributes heavily to CPI",
    "2023-12": "Year-end shelter and services stickiness; disinflation slower than expected",
    "2024-03": "Surprise CPI beat; rent and shelter components re-accelerated",
}

print()
print("TOP-3 MONTHS BY ABSOLUTE FORECAST ERROR (SARIMA):")
print("-" * 72)
for rank, (i, d, err, ae) in enumerate(zip(top3_idx, top3_dates, top3_errors, top3_abs), 1):
    key = d.strftime("%Y-%m")
    note = economic_notes.get(key, "Late-2023 disinflation slower than model anticipated")
    print(f"  {rank}. {key}  |  Error = {err*100:+.4f} pp  |  |Error| = {ae*100:.4f} pp")
    print(f"     Context: {note}")

# Save key metrics for Step 9
results = {
    "sarima_mae":  sarima_mae * 100,
    "sarima_rmse": sarima_rmse * 100,
    "ets_mae":     ets_mae * 100,
    "ets_rmse":    ets_rmse * 100,
    "mae_improvement":  mae_pct_improvement,
    "rmse_improvement": rmse_pct_improvement,
    "top3": [(d.strftime("%Y-%m"), f"{err*100:+.4f}") for d, err in zip(top3_dates, top3_errors)],
    "better_model": "SARIMA" if sarima_mae < ets_mae else "ETS",
}
import json
with open("step8_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print()
print("STEP 8 COMPLETE.")
