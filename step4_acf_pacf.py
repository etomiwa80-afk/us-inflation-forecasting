import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.stattools import acf, pacf

# ── Load series ──────────────────────────────────────────────────────────────
df = pd.read_csv("cpi_data.csv", index_col="date", parse_dates=True)
pi = df["pi"].dropna()
n  = len(pi)

ALPHA = 0.05
CONF  = 1.96 / np.sqrt(n)     # approximate 95% bound (Bartlett)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "font.family":      "sans-serif",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
}
plt.rcParams.update(STYLE)

NLAGS = 48

# ── Compute ACF/PACF with confidence bands ───────────────────────────────────
acf_vals,  acf_ci  = acf(pi,  nlags=NLAGS, alpha=ALPHA, fft=True)
pacf_vals, pacf_ci = pacf(pi, nlags=NLAGS, alpha=ALPHA, method="ywmle")

lags = np.arange(0, NLAGS + 1)

acf_lower  = acf_ci[:, 0]  - acf_vals
acf_upper  = acf_ci[:, 1]  - acf_vals
pacf_lower = pacf_ci[:, 0] - pacf_vals
pacf_upper = pacf_ci[:, 1] - pacf_vals

# ── Shared stem-plot helper ──────────────────────────────────────────────────
def stem_corr(ax, lags, vals, lower, upper, color="#2c3e50", title=""):
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.fill_between(lags, lower, upper,
                    color="#3498db", alpha=0.15, label="95% CI", zorder=2)
    ax.axhline( CONF, color="#3498db", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(-CONF, color="#3498db", linestyle="--", linewidth=0.9, alpha=0.7)

    for lag, val in zip(lags, vals):
        c = "#e74c3c" if abs(val) > CONF and lag > 0 else color
        ax.vlines(lag, 0, val, colors=c, linewidth=1.2, zorder=3)
        ax.plot(lag, val, "o", color=c, markersize=4, zorder=4)

    # Mark seasonal lags
    for sl in [12, 24, 36, 48]:
        if sl <= NLAGS:
            ax.axvline(sl, color="#e67e22", linestyle=":", linewidth=0.9, alpha=0.6)

    ax.set_title(title, pad=6)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Correlation")
    ax.set_xlim(-0.5, NLAGS + 0.5)
    ax.set_ylim(-0.55, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Full ACF + PACF side-by-side (lags 0–48)
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
fig.suptitle(
    r"ACF and PACF — Stationary Inflation Rate ($\pi_t$), Lags 0–48"
    "\nOrange dotted verticals mark seasonal lags (12, 24, 36, 48)",
    fontsize=12, y=1.02
)

stem_corr(ax1, lags, acf_vals,  acf_lower,  acf_upper,
          title=r"Autocorrelation Function (ACF) — $\pi_t$")
stem_corr(ax2, lags, pacf_vals, pacf_lower, pacf_upper,
          title=r"Partial Autocorrelation Function (PACF) — $\pi_t$")

fig.tight_layout()
fig.savefig("plot_05_acf_pacf_stationary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot_05_acf_pacf_stationary.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Seasonal lags only (12, 24, 36, 48)
# ══════════════════════════════════════════════════════════════════════════════
seasonal_lags = [12, 24, 36, 48]
sl_idx        = seasonal_lags  # same as lag number since lags = 0..48

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle(
    r"Seasonal-Lag ACF and PACF — $\pi_t$ at Lags 12, 24, 36, 48",
    fontsize=12, y=1.02
)

for ax, vals, lower, upper, ylabel, title in [
    (ax1, acf_vals,  acf_lower,  acf_upper,
     "Autocorrelation",        "ACF — Seasonal Lags"),
    (ax2, pacf_vals, pacf_lower, pacf_upper,
     "Partial Autocorrelation", "PACF — Seasonal Lags"),
]:
    ax.axhline(0,     color="black",   linewidth=0.8, zorder=1)
    ax.axhline( CONF, color="#3498db", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(-CONF, color="#3498db", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.fill_between([10, 50], -CONF, CONF,
                    color="#3498db", alpha=0.10, label="95% CI")

    for sl in seasonal_lags:
        val  = vals[sl]
        c    = "#e74c3c" if abs(val) > CONF else "#2c3e50"
        ax.vlines(sl, 0, val, colors=c, linewidth=2.5, zorder=3)
        ax.plot(sl, val, "D", color=c, markersize=8, zorder=4)
        ax.annotate(
            f"Lag {sl}\n{val:.3f}",
            xy=(sl, val),
            xytext=(sl, val + (0.06 if val >= 0 else -0.06)),
            fontsize=8.5, ha="center",
            color=c, fontweight="bold",
        )

    ax.set_title(title, pad=6)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(8, 52)
    ax.set_ylim(-0.55, 0.75)
    ax.set_xticks(seasonal_lags)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

fig.tight_layout()
fig.savefig("plot_06_acf_pacf_seasonal.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot_06_acf_pacf_seasonal.png")


# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL ANALYSIS — identify significant lags
# ══════════════════════════════════════════════════════════════════════════════

# Significant lags in ACF (skip lag 0)
acf_sig  = [k for k in range(1, NLAGS + 1) if abs(acf_vals[k])  > CONF]
pacf_sig = [k for k in range(1, NLAGS + 1) if abs(pacf_vals[k]) > CONF]

# First lag where ACF drops inside bounds permanently (consecutive 5)
def cutoff_lag(sig_list, total=NLAGS):
    in_bounds = set(range(1, total + 1)) - set(sig_list)
    for k in range(1, total - 3):
        if all(j in in_bounds for j in range(k, k + 5)):
            return k
    return total

acf_cutoff  = cutoff_lag(acf_sig)
pacf_cutoff = cutoff_lag(pacf_sig)

# Values at seasonal lags
acf_s12,  acf_s24  = acf_vals[12],  acf_vals[24]
pacf_s12, pacf_s24 = pacf_vals[12], pacf_vals[24]
acf_s12_sig  = abs(acf_s12)  > CONF
acf_s24_sig  = abs(acf_s24)  > CONF
pacf_s12_sig = abs(pacf_s12) > CONF
pacf_s24_sig = abs(pacf_s24) > CONF

# Detect decay pattern: compare first ~6 significant ACF lags
# Geometric decay: successive ratio ~ constant and < 1
early_acf = [acf_vals[k] for k in range(1, min(10, NLAGS + 1))]
ratios     = [abs(early_acf[i+1] / early_acf[i])
              for i in range(len(early_acf)-1) if early_acf[i] != 0]
geo_ratio  = np.mean(ratios[:5]) if ratios else 0
acf_pattern  = "Geometric decay (AR signature)" if geo_ratio > 0.5 else "Sharp cutoff (MA signature)"

# PACF: significant lags 1..p then cuts off  => AR(p)
pacf_early_sig = [k for k in range(1, 10) if abs(pacf_vals[k]) > CONF]
pacf_pattern   = (f"Sharp cutoff after lag {max(pacf_early_sig)} (AR signature)"
                  if len(pacf_early_sig) <= 6
                  else "Geometric decay (MA signature)")

print()
print("=" * 65)
print("ACF ANALYSIS")
print("=" * 65)
print(f"  95% confidence threshold    : +/-{CONF:.4f}")
print(f"  Significant lags (ACF)      : {acf_sig[:20]} ...")
print(f"  ACF first enters/stays CI   : lag ~ {acf_cutoff}")
print(f"  Decay pattern               : {acf_pattern}")
print(f"  ACF at lag 12               : {acf_s12:.4f}  "
      f"({'SIGNIFICANT' if acf_s12_sig else 'not significant'})")
print(f"  ACF at lag 24               : {acf_s24:.4f}  "
      f"({'SIGNIFICANT' if acf_s24_sig else 'not significant'})")

print()
print("=" * 65)
print("PACF ANALYSIS")
print("=" * 65)
print(f"  Significant lags (PACF)     : {pacf_sig[:20]} ...")
print(f"  PACF first enters/stays CI  : lag ~ {pacf_cutoff}")
print(f"  Decay pattern               : {pacf_pattern}")
print(f"  PACF at lag 12              : {pacf_s12:.4f}  "
      f"({'SIGNIFICANT' if pacf_s12_sig else 'not significant'})")
print(f"  PACF at lag 24              : {pacf_s24:.4f}  "
      f"({'SIGNIFICANT' if pacf_s24_sig else 'not significant'})")

# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE MODEL TABLE
# ══════════════════════════════════════════════════════════════════════════════
# d=0 confirmed by Step 3.
# ACF: slow geometric decay => AR component present
# PACF: significant spikes at lags 1,2,(possibly 3) then drops => AR(2) or AR(3)
# Seasonal lags: check significance at 12, 24

print()
print("=" * 75)
print("CANDIDATE SARIMA(p,d,q)(P,D,Q)_12 MODELS")
print("=" * 75)
print()

# Build rationale strings dynamically
s12_str = f"ACF lag 12 = {acf_s12:.3f} ({'sig' if acf_s12_sig else 'insig'}), PACF lag 12 = {pacf_s12:.3f}"

header = f"{'Model':<6} | {'p':>2} | {'d':>2} | {'q':>2} | {'P':>2} | {'D':>2} | {'Q':>2} | Rationale"
sep    = "-" * 95
print(header)
print(sep)

models = [
    ("A", 2, 0, 0, 1, 0, 1,
     "PACF cuts off ~lag 2 (AR); ACF decays geometrically; seasonal AR+MA for lag-12 spike"),
    ("B", 3, 0, 0, 1, 0, 0,
     "PACF significant through lag 3; AR(3) non-seasonal; seasonal AR captures lag-12 pattern"),
    ("C", 2, 0, 1, 0, 0, 1,
     "Mixed ARMA non-seasonal (parsimonious); seasonal MA corrects residual lag-12 autocorr"),
    ("D", 1, 0, 1, 1, 0, 1,
     "Classic SARIMA benchmark; ARMA(1,1) non-seasonal + seasonal ARMA(1,1)_12"),
    ("E", 3, 0, 1, 1, 0, 0,
     "Richer non-seasonal AR+MA; seasonal AR only; balances fit vs parsimony"),
    ("F", 2, 0, 0, 0, 0, 0,
     "Purely non-seasonal AR(2) baseline; tests whether seasonality term is needed"),
]

for m in models:
    label, p, d, q, P, D, Q, rat = m
    print(f"{label:<6} | {p:>2} | {d:>2} | {q:>2} | {P:>2} | {D:>2} | {Q:>2} | {rat}")

print(sep)
print()
print("Notes:")
print("  d = 0  (confirmed I(0) in Step 3 — all three unit-root tests)")
print("  D = 0  (no seasonal differencing; seasonal pattern is stationary at level)")
print("  s = 12 (monthly data, annual seasonal period)")
print()
print("STEP 4 COMPLETE.")
