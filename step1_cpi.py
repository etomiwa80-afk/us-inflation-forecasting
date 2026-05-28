import numpy as np
import pandas as pd
import pandas_datareader.data as web
import warnings
warnings.filterwarnings("ignore")

# ── 1. Fetch CPIAUCSL from FRED ──────────────────────────────────────────────
start, end = "1990-01-01", "2024-12-31"

try:
    raw = web.DataReader("CPIAUCSL", "fred", start, end)
except Exception as e:
    # Fallback: fredapi
    from fredapi import Fred
    fred = Fred()          # uses FRED_API_KEY env var if set; else public endpoint
    raw = fred.get_series("CPIAUCSL", observation_start=start, observation_end=end)
    raw = raw.to_frame(name="CPIAUCSL")

raw.index = pd.to_datetime(raw.index)
raw.index.name = "date"
raw.columns = ["CPI"]

# ── 2. Missing-value report ──────────────────────────────────────────────────
print("=" * 60)
print("RAW SERIES — CPIAUCSL (monthly, Jan 1990 – Dec 2024)")
print("=" * 60)
n_missing = raw["CPI"].isna().sum()
print(f"Missing values : {n_missing}")

# ── 3. Summary statistics for raw series ────────────────────────────────────
print(f"Shape          : {raw.shape}")
print(f"Date range     : {raw.index.min().date()}  to  {raw.index.max().date()}")
print(f"Min            : {raw['CPI'].min():.4f}")
print(f"Max            : {raw['CPI'].max():.4f}")
print(f"Mean           : {raw['CPI'].mean():.4f}")
print(f"Std dev        : {raw['CPI'].std():.4f}")

# ── 4. Year-over-year log-difference (inflation rate) ───────────────────────
raw["log_CPI"] = np.log(raw["CPI"])
raw["pi"] = raw["log_CPI"] - raw["log_CPI"].shift(12)   # π_t = log(CPI_t) − log(CPI_{t−12})

pi = raw["pi"].dropna()

print()
print("=" * 60)
print("TRANSFORMED SERIES — Year-over-year log-difference (π_t)")
print("=" * 60)
print(f"Missing values : {raw['pi'].isna().sum()} (first 12 obs, expected)")
print(f"Shape          : ({len(pi)},)")
print(f"Date range     : {pi.index.min().date()}  to  {pi.index.max().date()}")
print(f"Min            : {pi.min():.6f}  ({pi.idxmin().date()})")
print(f"Max            : {pi.max():.6f}  ({pi.idxmax().date()})")
print(f"Mean           : {pi.mean():.6f}")
print(f"Std dev        : {pi.std():.6f}")
print(f"Skewness       : {pi.skew():.6f}")
print(f"Kurtosis       : {pi.kurt():.6f}")

# ── 5. Save to CSV ───────────────────────────────────────────────────────────
out = raw[["CPI", "pi"]].copy()
out.to_csv("cpi_data.csv", float_format="%.6f")
print()
print("Saved → cpi_data.csv  (columns: date, CPI, pi)")

print()
print("STEP 1 COMPLETE.")
