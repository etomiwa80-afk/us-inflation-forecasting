import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from arch import arch_model
from arch.unitroot.unitroot import _df_select_lags

# ── Style ────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.grid": True,
    "grid.color": "#e0e0e0", "grid.linestyle": "--", "grid.linewidth": 0.5,
    "font.family": "sans-serif", "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
}
plt.rcParams.update(STYLE)

# ── Load model and residuals ─────────────────────────────────────────────────
with open("best_model.pkl", "rb") as f:
    res = pickle.load(f)

resid = res.resid.dropna()
std_resid = (resid - resid.mean()) / resid.std()
n = len(resid)
CONF = 1.96 / np.sqrt(n)

print(f"Model       : {res.model.__class__.__name__}  —  SARIMA(1,0,1)(1,0,1)12")
print(f"Residuals   : {n} observations  ({resid.index.min().date()} to {resid.index.max().date()})")
print(f"Mean resid  : {resid.mean():.2e}   (should be ~0)")
print(f"Std  resid  : {resid.std():.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LJUNG-BOX TEST
# ══════════════════════════════════════════════════════════════════════════════
lb_lags = [6, 12, 18, 24]
lb = acorr_ljungbox(resid, lags=lb_lags, return_df=True)

print()
print("=" * 65)
print("LJUNG-BOX TEST  (H0: no autocorrelation in residuals)")
print("=" * 65)
hdr = f"{'Lag':>4} | {'Q-statistic':>13} | {'p-value':>9} | Decision (5%)"
print(hdr)
print("-" * 65)
lb_decisions = {}
for lag in lb_lags:
    q   = lb.loc[lag, "lb_stat"]
    pv  = lb.loc[lag, "lb_pvalue"]
    dec = "Reject H0 (autocorrelation remains)" if pv < 0.05 else "Fail to reject H0"
    lb_decisions[lag] = (q, pv, dec)
    print(f"{lag:>4} | {q:>13.4f} | {pv:>9.4f} | {dec}")
print("=" * 65)

all_pass = all(v[1] >= 0.05 for v in lb_decisions.values())
print(f"\nOverall conclusion: ", end="")
if all_pass:
    print("Model CAPTURES all autocorrelation structure — no significant residual serial correlation at any tested lag.")
else:
    fail_lags = [k for k, v in lb_decisions.items() if v[1] < 0.05]
    print(f"Residual autocorrelation remains significant at lag(s): {fail_lags}. Model may be mis-specified.")

# ══════════════════════════════════════════════════════════════════════════════
# 2. RESIDUAL ACF / PACF (plot_07)
# ══════════════════════════════════════════════════════════════════════════════
NLAGS = 36
acf_r,  acf_ci  = acf(resid,  nlags=NLAGS, alpha=0.05, fft=True)
pacf_r, pacf_ci = pacf(resid, nlags=NLAGS, alpha=0.05, method="ywmle")
lags = np.arange(0, NLAGS + 1)

acf_lo  = acf_ci[:, 0]  - acf_r
acf_hi  = acf_ci[:, 1]  - acf_r
pacf_lo = pacf_ci[:, 0] - pacf_r
pacf_hi = pacf_ci[:, 1] - pacf_r

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
fig.suptitle("Residual ACF and PACF — SARIMA(1,0,1)(1,0,1)₁₂\n95% Confidence Bounds", fontsize=12, y=1.01)

for ax, vals, lo, hi, title in [
    (ax1, acf_r,  acf_lo,  acf_hi,  "Residual ACF"),
    (ax2, pacf_r, pacf_lo, pacf_hi, "Residual PACF"),
]:
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(lags,  CONF * np.ones(len(lags)), -CONF * np.ones(len(lags)),
                    color="#3498db", alpha=0.12, label="95% CI")
    ax.axhline( CONF, color="#3498db", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(-CONF, color="#3498db", linestyle="--", linewidth=0.9, alpha=0.7)
    for lag, val in zip(lags[1:], vals[1:]):
        c = "#e74c3c" if abs(val) > CONF else "#2c3e50"
        ax.vlines(lag, 0, val, colors=c, linewidth=1.2)
        ax.plot(lag, val, "o", color=c, markersize=4)
    for sl in [12, 24, 36]:
        ax.axvline(sl, color="#e67e22", linestyle=":", linewidth=0.9, alpha=0.5)
    ax.set_title(title, pad=5)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Correlation")
    ax.set_xlim(0.5, NLAGS + 0.5)
    ax.set_ylim(-0.35, 0.35)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

fig.tight_layout()
fig.savefig("plot_07_residual_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: plot_07_residual_acf_pacf.png")

acf_sig_r  = [k for k in range(1, NLAGS + 1) if abs(acf_r[k])  > CONF]
pacf_sig_r = [k for k in range(1, NLAGS + 1) if abs(pacf_r[k]) > CONF]
print(f"  Significant ACF  lags in residuals : {acf_sig_r if acf_sig_r else 'None'}")
print(f"  Significant PACF lags in residuals : {pacf_sig_r if pacf_sig_r else 'None'}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. NORMALITY DIAGNOSTICS (plot_08)
# ══════════════════════════════════════════════════════════════════════════════
jb_stat, jb_pval = stats.jarque_bera(resid)[:2]
sk  = stats.skew(resid)
kt  = stats.kurtosis(resid)          # excess kurtosis

print()
print("=" * 65)
print("JARQUE-BERA NORMALITY TEST")
print("=" * 65)
print(f"  JB statistic : {jb_stat:.4f}")
print(f"  p-value      : {jb_pval:.4e}")
print(f"  Skewness     : {sk:.4f}")
print(f"  Kurtosis     : {kt:.4f}  (excess)")
jb_pass = jb_pval >= 0.05
print(f"  Decision     : {'Fail to reject H0 (normality not rejected)' if jb_pass else 'Reject H0 (non-normal residuals)'}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Residual Normality Diagnostics — SARIMA(1,0,1)(1,0,1)₁₂", fontsize=12, y=1.01)

# Histogram + fitted normal
x_grid = np.linspace(resid.min(), resid.max(), 300)
mu_r, sd_r = resid.mean(), resid.std()
ax1.hist(resid, bins=40, density=True, color="#2c3e50", alpha=0.65, edgecolor="white", linewidth=0.4, label="Residuals")
ax1.plot(x_grid, stats.norm.pdf(x_grid, mu_r, sd_r), color="#e74c3c", linewidth=2, label=f"N({mu_r:.1e}, {sd_r:.1e}²)")
ax1.plot(x_grid, stats.t.pdf(x_grid, df=5, loc=mu_r, scale=sd_r * np.sqrt(3/5)),
         color="#27ae60", linewidth=1.5, linestyle="--", label="t(df=5) scaled")
ax1.set_title("Histogram of Residuals vs Normal Curve", pad=5)
ax1.set_xlabel("Residual value")
ax1.set_ylabel("Density")
ax1.legend(fontsize=8, framealpha=0.85)
ax1.text(0.02, 0.97, f"JB stat = {jb_stat:.1f}\np-value = {jb_pval:.2e}\nKurtosis = {kt:.2f}",
         transform=ax1.transAxes, fontsize=8, va="top",
         bbox=dict(boxstyle="round", fc="white", ec="#aaaaaa", alpha=0.85))

# QQ plot
(osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
ax2.plot(osm, osr, "o", color="#2c3e50", markersize=3, alpha=0.7, label="Residuals")
ax2.plot(osm, slope * np.array(osm) + intercept, color="#e74c3c", linewidth=1.8, label="Normal reference line")
ax2.set_title("Q-Q Plot — Residuals vs Normal Distribution", pad=5)
ax2.set_xlabel("Theoretical quantiles")
ax2.set_ylabel("Sample quantiles")
ax2.legend(fontsize=8, framealpha=0.85)
ax2.text(0.02, 0.97, f"R² = {r**2:.4f}", transform=ax2.transAxes, fontsize=8, va="top",
         bbox=dict(boxstyle="round", fc="white", ec="#aaaaaa", alpha=0.85))

fig.tight_layout()
fig.savefig("plot_08_residual_normality.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: plot_08_residual_normality.png")

if jb_pass:
    print("  Normality holds — Gaussian prediction intervals are reliable.")
else:
    print("  Normality FAILS — fat tails (kurtosis > 3) mean Gaussian 95% PI will under-cover.")
    print("  Robust or bootstrap prediction intervals recommended.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. ARCH-LM TEST + GARCH(1,1)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("ARCH-LM TEST  (H0: no ARCH effects)")
print("=" * 65)

from arch.unitroot.unitroot import _df_select_lags

def arch_lm(series, lags):
    from statsmodels.stats.diagnostic import het_arch
    lm_stat, lm_pval, f_stat, f_pval = het_arch(series, nlags=lags)
    return lm_stat, lm_pval

arch_results = {}
for lag in [6, 12]:
    lm, pv = arch_lm(resid, lag)
    dec = "Reject H0 (ARCH effects present)" if pv < 0.05 else "Fail to reject H0"
    arch_results[lag] = (lm, pv, dec)
    print(f"  Lag {lag:>2}: LM stat = {lm:>9.4f}   p-value = {pv:.4f}   {dec}")

arch_present = any(v[1] < 0.05 for v in arch_results.values())

garch_res = None
if arch_present:
    print()
    print("ARCH effects confirmed — fitting GARCH(1,1) on residuals ...")
    gm = arch_model(resid * 1e4, vol="Garch", p=1, q=1, dist="normal")
    garch_res = gm.fit(disp="off")
    omega = garch_res.params["omega"]
    alpha = garch_res.params["alpha[1]"]
    beta  = garch_res.params["beta[1]"]
    persist = alpha + beta
    print(f"  GARCH(1,1) parameters (residuals x 10^4):")
    print(f"    omega (omega)   : {omega:.6f}")
    print(f"    alpha[1] (ARCH) : {alpha:.6f}")
    print(f"    beta[1]  (GARCH): {beta:.6f}")
    print(f"    Persistence     : alpha + beta = {persist:.6f}  "
          f"({'near unit root' if persist > 0.95 else 'stable'})")
    print()
    if persist > 0.99:
        print("  GARCH extension: HIGH PERSISTENCE — consider IGARCH.")
    elif persist > 0.90:
        print("  GARCH extension: WARRANTED — volatility is strongly persistent.")
    else:
        print("  GARCH extension: Moderate volatility clustering; GARCH(1,1) captures it.")
else:
    print("  No ARCH effects — GARCH extension NOT required.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. STANDARDIZED RESIDUALS OVER TIME (plot_09)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(std_resid.index, std_resid.values, color="#2c3e50", linewidth=0.9, alpha=0.9)
ax.fill_between(std_resid.index, -2, 2, color="#3498db", alpha=0.08, label="±2 SD")
for sd, color, ls in [(2, "#3498db", "--"), (3, "#e74c3c", "--")]:
    ax.axhline( sd, color=color, linestyle=ls, linewidth=1.1, alpha=0.85)
    ax.axhline(-sd, color=color, linestyle=ls, linewidth=1.1, alpha=0.85)

# Annotate outliers beyond ±3 SD
outliers = std_resid[np.abs(std_resid) > 3]
n_outliers = len(outliers)
for date, val in outliers.items():
    ax.plot(date, val, "D", color="#e74c3c", markersize=7, zorder=5)
    ax.annotate(
        f"{date.strftime('%b %Y')}\n({val:+.2f}sd)",
        xy=(date, val),
        xytext=(date, val + (0.45 if val > 0 else -0.55)),
        fontsize=7, color="#c0392b", ha="center",
        arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8),
    )

ax.axhline(0, color="black", linewidth=0.7)
ax.set_title("Standardized Residuals Over Time — SARIMA(1,0,1)(1,0,1)₁₂\n"
             "Dashed lines: ±2 SD (blue) and ±3 SD (red); red diamonds = outliers beyond 3 SD",
             pad=8)
ax.set_xlabel("Date")
ax.set_ylabel("Standardized Residual")
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.set_xlim(std_resid.index.min(), std_resid.index.max())
ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
ax.text(0.99, 0.97, f"Outliers > 3 SD: {n_outliers}", transform=ax.transAxes,
        fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#aaaaaa", alpha=0.85))
fig.tight_layout()
fig.savefig("plot_09_residuals_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: plot_09_residuals_time.png  ({n_outliers} outlier(s) beyond 3 SD)")

if n_outliers:
    print("  Outlier dates and values:")
    for date, val in outliers.items():
        print(f"    {date.date()}  :  {val:+.3f} SD")

# ══════════════════════════════════════════════════════════════════════════════
# 6. DIAGNOSTIC SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
lb12_p  = lb_decisions[12][1]
lb24_p  = lb_decisions[24][1]
arch6_p = arch_results[6][1]

def flag(p, thresh=0.05, invert=False):
    sig = p < thresh
    if invert:
        return "Pass" if not sig else "Concern"
    return "Concern" if sig else "Pass"

print()
print("=" * 72)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 72)
hdr = f"{'Diagnostic':<30} | {'Result':<18} | Pass/Concern"
print(hdr)
print("-" * 72)
rows = [
    ("Ljung-Box Q(12)",       f"p = {lb12_p:.4f}",   flag(lb12_p)),
    ("Ljung-Box Q(24)",       f"p = {lb24_p:.4f}",   flag(lb24_p)),
    ("Jarque-Bera normality", f"p = {jb_pval:.2e}",  flag(jb_pval)),
    ("ARCH-LM (lag 6)",       f"p = {arch6_p:.4f}",  flag(arch6_p)),
    ("AR root proximity",     "root = 1.0002",        "Concern -- monitor"),
    ("Outliers beyond 3 SD",  f"n = {n_outliers}",
     "Pass" if n_outliers <= 5 else "Concern"),
]
for r in rows:
    print(f"{r[0]:<30} | {r[1]:<18} | {r[2]}")
print("=" * 72)
print()
print("STEP 6 COMPLETE.")
