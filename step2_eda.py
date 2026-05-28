import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from statsmodels.graphics.tsaplots import plot_acf

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("cpi_data.csv", index_col="date", parse_dates=True)
cpi = df["CPI"].dropna()
pi  = df["pi"].dropna()

# Annotation events
events = [
    ("2008-10-01", "Oct 2008\nFinancial Crisis"),
    ("2020-04-01", "Apr 2020\nCOVID Shock"),
    ("2022-06-01", "Jun 2022\nInflation Peak"),
]

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.grid":        True,
    "grid.color":       "#dddddd",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
}
plt.rcParams.update(STYLE)

EVENT_COLORS = ["#c0392b", "#2980b9", "#27ae60"]

def add_events(ax, ymin_frac=0.02, ymax_frac=0.92):
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for (date, label), color in zip(events, EVENT_COLORS):
        x = pd.Timestamp(date)
        ax.axvline(x, color=color, linestyle="--", linewidth=1.2, alpha=0.85)
        ax.text(
            x, ylim[0] + ymax_frac * span, label,
            fontsize=7.5, color=color, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.75, lw=0.8),
        )

# ── Plot 1: Raw CPI level ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(cpi.index, cpi.values, color="#2c3e50", linewidth=1.4)
ax.set_title("U.S. Consumer Price Index — All Urban Consumers (CPIAUCSL)\nJanuary 1990 – December 2024", pad=10)
ax.set_xlabel("Date")
ax.set_ylabel("CPI Level (1982–84 = 100)")
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.set_xlim(cpi.index.min(), cpi.index.max())
add_events(ax, ymax_frac=0.97)
fig.tight_layout()
fig.savefig("plot_01_raw_cpi.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot_01_raw_cpi.png")

# ── Plot 2: Inflation rate (pi_t) ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(pi.index, pi.values * 100, color="#2c3e50", linewidth=1.2)
ax.fill_between(pi.index, pi.values * 100, 0,
                where=(pi.values >= 0), color="#e74c3c", alpha=0.15, label="Above zero")
ax.fill_between(pi.index, pi.values * 100, 0,
                where=(pi.values < 0),  color="#3498db", alpha=0.20, label="Below zero")
ax.axhline(2.0, color="#27ae60", linestyle="--", linewidth=1.3, label="Fed 2% target")
ax.set_title("U.S. Inflation Rate — Year-over-Year Log Difference of CPI\nJanuary 1991 – December 2024", pad=10)
ax.set_xlabel("Date")
ax.set_ylabel(r"$\pi_t = \ln(CPI_t) - \ln(CPI_{t-12})$ (%, annualized)")
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.set_xlim(pi.index.min(), pi.index.max())
add_events(ax, ymax_frac=0.96)
ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
fig.tight_layout()
fig.savefig("plot_02_inflation_rate.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot_02_inflation_rate.png")

# ── Plot 3: Seasonal subseries ────────────────────────────────────────────────
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

pi_df = pi.to_frame(name="pi")
pi_df["year"]  = pi_df.index.year
pi_df["month"] = pi_df.index.month

fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharey=True)
fig.suptitle("Seasonal Subseries Plot — Year-over-Year Inflation Rate (π_t)\nEach panel: one calendar month across 1991–2024",
             fontsize=13, y=1.01)

for m, ax in zip(range(1, 13), axes.flat):
    sub = pi_df[pi_df["month"] == m].sort_values("year")
    ax.plot(sub["year"], sub["pi"] * 100, color="#2c3e50", linewidth=0.9, marker="o",
            markersize=2.5, markerfacecolor="#e74c3c", markeredgewidth=0)
    month_mean = sub["pi"].mean() * 100
    ax.axhline(month_mean, color="#e74c3c", linestyle="--", linewidth=1.1,
               label=f"Mean: {month_mean:.2f}%")
    ax.axhline(2.0, color="#27ae60", linestyle=":", linewidth=0.9, alpha=0.8)
    ax.set_title(month_names[m - 1], fontsize=10, pad=3)
    ax.set_xlabel("Year", fontsize=7)
    ax.xaxis.set_major_locator(MultipleLocator(8))
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.7)

axes.flat[0].set_ylabel("π_t (%)", fontsize=8)
axes.flat[4].set_ylabel("π_t (%)", fontsize=8)
axes.flat[8].set_ylabel("π_t (%)", fontsize=8)

fig.tight_layout()
fig.savefig("plot_03_seasonal_subseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot_03_seasonal_subseries.png")

# ── Plot 4: ACF (raw pi_t, 48 lags) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
plot_acf(pi, lags=48, alpha=0.05, ax=ax,
         color="#2c3e50", vlines_kwargs={"colors": "#2c3e50", "linewidths": 1.2})
ax.set_title(r"Autocorrelation Function — Inflation Rate ($\pi_t$), Lags 0–48", pad=10)
ax.set_xlabel("Lag (months)")
ax.set_ylabel("Autocorrelation")
ax.xaxis.set_major_locator(MultipleLocator(6))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlim(-0.5, 48.5)
fig.tight_layout()
fig.savefig("plot_04_acf_raw.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot_04_acf_raw.png")

# ── Descriptive statistics ────────────────────────────────────────────────────
print()
print("=" * 60)
print("DESCRIPTIVE STATISTICS — pi_t (year-over-year log-diff)")
print("=" * 60)

pct = pi * 100   # express as percentage for readability

print(f"  Observations : {len(pi)}")
print(f"  Mean         : {pct.mean():.4f}%")
print(f"  Median       : {pct.median():.4f}%")
print(f"  Std dev      : {pct.std():.4f}%")
print(f"  Skewness     : {pct.skew():.4f}")
print(f"  Kurtosis     : {pct.kurt():.4f}  (excess, vs. normal = 0)")
print(f"  Min          : {pct.min():.4f}%  on  {pct.idxmin().date()}")
print(f"  Max          : {pct.max():.4f}%  on  {pct.idxmax().date()}")
print(f"  First obs    : {pct.iloc[0]:.4f}%  ({pi.index[0].date()})")
print(f"  Last obs     : {pct.iloc[-1]:.4f}%  ({pi.index[-1].date()})")

print()
print("STEP 2 COMPLETE.")
