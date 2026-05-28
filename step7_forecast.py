import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
test  = pi[pi.index >  "2022-12-31"]       # Jan 2023 – Dec 2024 (24 obs)

with open("best_model.pkl", "rb") as f:
    train_res = pickle.load(f)

ORDER      = train_res.model.order           # (1, 0, 1)
SEAS_ORDER = train_res.model.seasonal_order  # (1, 0, 1, 12)

print(f"Selected specification : SARIMA{ORDER}x{SEAS_ORDER}")
print(f"Training model         : {len(train)} obs  ({train.index.min().date()} – {train.index.max().date()})")
print(f"Test (holdout)         : {len(test)}  obs  ({test.index.min().date()} – {test.index.max().date()})")
print(f"Full series            : {len(pi)} obs  ({pi.index.min().date()} – {pi.index.max().date()})")

# ══════════════════════════════════════════════════════════════════════════════
# 1. REFIT ON FULL SERIES  (Jan 1991 – Dec 2024)
# ══════════════════════════════════════════════════════════════════════════════
print("\nRefitting on full series ...")
final_mod = SARIMAX(
    pi,
    order=ORDER,
    seasonal_order=SEAS_ORDER,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
final_res = final_mod.fit(method="lbfgs", disp=False)
print(f"  AIC  = {final_res.aic:.3f}")
print(f"  BIC  = {final_res.bic:.3f}")

with open("final_model.pkl", "wb") as f:
    pickle.dump(final_res, f)
print("  Saved: final_model.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 12-MONTH OUT-OF-SAMPLE FORECAST  (Jan – Dec 2025)
# ══════════════════════════════════════════════════════════════════════════════
HORIZON = 12
fc = final_res.get_forecast(steps=HORIZON)
fc_mean = fc.predicted_mean
fc_ci80 = fc.conf_int(alpha=0.20)   # 80% PI
fc_ci95 = fc.conf_int(alpha=0.05)   # 95% PI

# Build forecast date index: monthly from 2025-01
fc_dates = pd.date_range(start="2025-01-01", periods=HORIZON, freq="MS")
fc_pct   = fc_mean.values * 100          # convert to %
lo80     = fc_ci80.iloc[:, 0].values * 100
hi80     = fc_ci80.iloc[:, 1].values * 100
lo95     = fc_ci95.iloc[:, 0].values * 100
hi95     = fc_ci95.iloc[:, 1].values * 100

# ── Forecast table ────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("12-MONTH FORECAST  Jan 2025 – Dec 2025  (year-over-year log-diff, %)")
print("=" * 80)
hdr = f"{'Date':<10} | {'Forecast (%)':>13} | {'Lower 80%':>10} | {'Upper 80%':>10} | {'Lower 95%':>10} | {'Upper 95%':>10}"
print(hdr)
print("-" * 80)
for i, d in enumerate(fc_dates):
    print(f"{d.strftime('%Y-%m'):<10} | {fc_pct[i]:>13.4f} | {lo80[i]:>10.4f} | {hi80[i]:>10.4f} | {lo95[i]:>10.4f} | {hi95[i]:>10.4f}")
print("=" * 80)
print(f"\nForecast range: [{fc_pct.min():.3f}%, {fc_pct.max():.3f}%]")
print(f"Mean forecast : {fc_pct.mean():.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOT 10 — Forecast with PI bands
# ══════════════════════════════════════════════════════════════════════════════
obs_window = pi[pi.index >= "2018-01-01"]

fig, ax = plt.subplots(figsize=(13, 5))

# Observed (2018 onward)
ax.plot(obs_window.index, obs_window.values * 100,
        color="#2c3e50", linewidth=1.5, label="Observed $\\pi_t$", zorder=4)

# 95% PI shading
ax.fill_between(fc_dates, lo95, hi95,
                color="#3498db", alpha=0.18, label="95% Prediction Interval")
# 80% PI shading
ax.fill_between(fc_dates, lo80, hi80,
                color="#3498db", alpha=0.35, label="80% Prediction Interval")
# Point forecast
ax.plot(fc_dates, fc_pct,
        color="#e74c3c", linewidth=2.0, marker="o", markersize=5,
        label="Point Forecast", zorder=5)

# Reference lines
ax.axvline(pd.Timestamp("2024-12-01"), color="#555555", linestyle="--",
           linewidth=1.1, label="End of observed data (Dec 2024)")
ax.axhline(2.0, color="#27ae60", linestyle="--", linewidth=1.2,
           label="Fed 2% target", zorder=3)

ax.set_title(
    "SARIMA(1,0,1)(1,0,1)₁₂ — 12-Month Out-of-Sample Forecast\n"
    "Observed: Jan 2018 – Dec 2024  |  Forecast: Jan – Dec 2025",
    pad=8
)
ax.set_xlabel("Date")
ax.set_ylabel(r"Year-over-Year Inflation Rate $\pi_t$ (%)")
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.set_xlim(obs_window.index.min(), fc_dates[-1])
ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

fig.tight_layout()
fig.savefig("plot_10_forecast.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: plot_10_forecast.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOT 11 — Holdout period fit  (training model vs actual 2023–2024)
# ══════════════════════════════════════════════════════════════════════════════
# Use the training model (best_model.pkl) to forecast the 24-month holdout
holdout_fc  = train_res.get_forecast(steps=len(test))
holdout_mean = holdout_fc.predicted_mean.values
holdout_ci95 = holdout_fc.conf_int(alpha=0.05)
holdout_lo95 = holdout_ci95.iloc[:, 0].values
holdout_hi95 = holdout_ci95.iloc[:, 1].values
holdout_ci80 = holdout_fc.conf_int(alpha=0.20)
holdout_lo80 = holdout_ci80.iloc[:, 0].values
holdout_hi80 = holdout_ci80.iloc[:, 1].values

# Align to test index
holdout_idx = test.index

mae  = np.mean(np.abs(test.values - holdout_mean))
rmse = np.sqrt(np.mean((test.values - holdout_mean) ** 2))

print(f"\nHoldout evaluation (Jan 2023 – Dec 2024):")
print(f"  MAE  = {mae*100:.4f}%")
print(f"  RMSE = {rmse*100:.4f}%")

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.fill_between(holdout_idx, holdout_lo95 * 100, holdout_hi95 * 100,
                color="#3498db", alpha=0.18, label="95% PI")
ax.fill_between(holdout_idx, holdout_lo80 * 100, holdout_hi80 * 100,
                color="#3498db", alpha=0.35, label="80% PI")
ax.plot(holdout_idx, test.values * 100,
        color="#2c3e50", linewidth=1.8, marker="o", markersize=5,
        label="Observed $\\pi_t$", zorder=5)
ax.plot(holdout_idx, holdout_mean * 100,
        color="#e74c3c", linewidth=1.8, linestyle="--", marker="s", markersize=4,
        label="Model Forecast (dynamic)", zorder=4)
ax.axhline(2.0, color="#27ae60", linestyle=":", linewidth=1.1, alpha=0.8,
           label="Fed 2% target")

# Error bars (absolute errors)
for i, (d, obs, pred) in enumerate(zip(holdout_idx, test.values * 100, holdout_mean * 100)):
    ax.plot([d, d], [obs, pred], color="#aaaaaa", linewidth=0.8, zorder=3)

ax.set_title(
    "Holdout Period Forecast vs Actual — SARIMA(1,0,1)(1,0,1)₁₂\n"
    f"Jan 2023 – Dec 2024  |  MAE = {mae*100:.3f}%  |  RMSE = {rmse*100:.3f}%",
    pad=8
)
ax.set_xlabel("Date")
ax.set_ylabel(r"Year-over-Year Inflation Rate $\pi_t$ (%)")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.set_xlim(holdout_idx.min(), holdout_idx.max())
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.text(0.01, 0.04,
        f"MAE = {mae*100:.3f}%\nRMSE = {rmse*100:.3f}%",
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="#aaaaaa", alpha=0.85))

fig.tight_layout()
fig.savefig("plot_11_holdout_fit.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot_11_holdout_fit.png")

# ── Monthly error breakdown ───────────────────────────────────────────────────
print()
print("=" * 70)
print("HOLDOUT DETAIL TABLE  (Jan 2023 – Dec 2024)")
print("=" * 70)
print(f"{'Date':<10} | {'Observed (%)':>13} | {'Forecast (%)':>13} | {'Error (pp)':>11}")
print("-" * 70)
for d, obs, pred in zip(holdout_idx, test.values * 100, holdout_mean * 100):
    err = obs - pred
    print(f"{d.strftime('%Y-%m'):<10} | {obs:>13.4f} | {pred:>13.4f} | {err:>+11.4f}")
print("=" * 70)
print(f"\n  MAE  = {mae*100:.4f} percentage points")
print(f"  RMSE = {rmse*100:.4f} percentage points")

print()
print("STEP 7 COMPLETE.")
