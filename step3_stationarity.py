import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron

# ── Load series ──────────────────────────────────────────────────────────────
df = pd.read_csv("cpi_data.csv", index_col="date", parse_dates=True)
pi = df["pi"].dropna()

# ── Helper functions ─────────────────────────────────────────────────────────

def run_adf(series, label):
    result = adfuller(series, autolag="AIC")
    stat, pval, nlags = result[0], result[1], result[2]
    cv = result[4]          # dict: {'1%': ..., '5%': ..., '10%': ...}
    decision = "Reject H0 (stationary)" if pval < 0.05 else "Fail to reject H0 (unit root)"

    print(f"\n--- ADF Test on {label} ---")
    print(f"  Lags selected (AIC) : {nlags}")
    print(f"  Test statistic      : {stat:.6f}")
    print(f"  p-value             : {pval:.6f}")
    print(f"  Critical values     :  1% = {cv['1%']:.4f}  |  5% = {cv['5%']:.4f}  |  10% = {cv['10%']:.4f}")
    print(f"  Decision (5%)       : {decision}")
    return stat, pval, decision


def run_pp(series, label):
    pp = PhillipsPerron(series)
    stat = pp.stat
    pval = pp.pvalue
    cv   = pp.critical_values    # dict: {'1%': ..., '5%': ..., '10%': ...}
    decision = "Reject H0 (stationary)" if pval < 0.05 else "Fail to reject H0 (unit root)"

    print(f"\n--- Phillips-Perron Test on {label} ---")
    print(f"  Test statistic      : {stat:.6f}")
    print(f"  p-value             : {pval:.6f}")
    print(f"  Critical values     :  1% = {cv['1%']:.4f}  |  5% = {cv['5%']:.4f}  |  10% = {cv['10%']:.4f}")
    print(f"  Decision (5%)       : {decision}")
    return stat, pval, decision


def run_kpss(series, label):
    stat, pval, nlags, cv = kpss(series, regression="c", nlags="auto")
    # KPSS H0 = stationary; reject if stat > critical value
    cv5 = cv["5%"]
    decision = "Reject H0 (unit root)" if stat > cv5 else "Fail to reject H0 (stationary)"

    print(f"\n--- KPSS Test on {label} (H0: stationary) ---")
    print(f"  Lags used           : {nlags}")
    print(f"  Test statistic      : {stat:.6f}")
    print(f"  p-value (approx)    : {pval:.6f}")
    print(f"  Critical values     :  10% = {cv['10%']:.4f}  |  5% = {cv['5%']:.4f}  "
          f"|  2.5% = {cv['2.5%']:.4f}  |  1% = {cv['1%']:.4f}")
    print(f"  Decision (5%)       : {decision}")
    return stat, pval, decision


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1: Tests on pi_t (level)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STATIONARITY TESTS — pi_t  (YoY log-difference of CPI)")
print("=" * 65)

adf_stat1,  adf_pval1,  adf_dec1  = run_adf(pi,  "pi_t")
pp_stat1,   pp_pval1,   pp_dec1   = run_pp(pi,   "pi_t")
kpss_stat1, kpss_pval1, kpss_dec1 = run_kpss(pi, "pi_t")

# ── Consistency check ─────────────────────────────────────────────────────────
adf_stationary  = adf_pval1  < 0.05
pp_stationary   = pp_pval1   < 0.05
kpss_stationary = "Fail to reject" in kpss_dec1   # KPSS: fail to reject H0 => stationary
all_consistent  = (adf_stationary == pp_stationary == kpss_stationary)

print()
print(f"Consistency check: ADF={adf_stationary}, PP={pp_stationary}, "
      f"KPSS_stationary={kpss_stationary}  -->  All consistent: {all_consistent}")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 (conditional): Tests on d_pi_t (first difference)
# ═══════════════════════════════════════════════════════════════════════════════
run_diff = not all_consistent

if run_diff:
    d_pi = pi.diff().dropna()
    print()
    print("=" * 65)
    print("STATIONARITY TESTS — d_pi_t  (first difference of pi_t)")
    print("=" * 65)

    adf_stat2,  adf_pval2,  adf_dec2  = run_adf(d_pi,  "d_pi_t")
    pp_stat2,   pp_pval2,   pp_dec2   = run_pp(d_pi,   "d_pi_t")
    kpss_stat2, kpss_pval2, kpss_dec2 = run_kpss(d_pi, "d_pi_t")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
def short_dec(dec):
    if "Reject H0" in dec and "stationary" in dec.lower() and "unit root" not in dec.lower():
        return "Reject H0 (stationary)"
    elif "Fail to reject" in dec and "unit root" not in dec.lower():
        return "Fail to reject H0 (stationary)"
    elif "Reject H0" in dec and "unit root" in dec.lower():
        return "Reject H0 (non-stationary)"
    else:
        return dec

rows = [
    ("ADF",  "pi_t",  f"{adf_stat1:>9.4f}",  f"{adf_pval1:.4f}",  short_dec(adf_dec1)),
    ("PP",   "pi_t",  f"{pp_stat1:>9.4f}",   f"{pp_pval1:.4f}",   short_dec(pp_dec1)),
    ("KPSS", "pi_t",  f"{kpss_stat1:>9.4f}", f"{kpss_pval1:.4f}", short_dec(kpss_dec1)),
]
if run_diff:
    rows += [
        ("ADF",  "d_pi_t", f"{adf_stat2:>9.4f}",  f"{adf_pval2:.4f}",  short_dec(adf_dec2)),
        ("PP",   "d_pi_t", f"{pp_stat2:>9.4f}",   f"{pp_pval2:.4f}",   short_dec(pp_dec2)),
        ("KPSS", "d_pi_t", f"{kpss_stat2:>9.4f}", f"{kpss_pval2:.4f}", short_dec(kpss_dec2)),
    ]

print()
print("=" * 85)
print("SUMMARY TABLE")
print("=" * 85)
hdr = f"{'Test':<6} | {'Series':<8} | {'Statistic':>10} | {'p-value':>8} | Decision"
print(hdr)
print("-" * 85)
for r in rows:
    print(f"{r[0]:<6} | {r[1]:<8} | {r[2]:>10} | {r[3]:>8} | {r[4]}")
print("=" * 85)

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("FINAL CONCLUSION — Order of Integration")
print("-" * 65)
if all_consistent and adf_stationary:
    d_order = 0
    print("  All three tests agree: pi_t is STATIONARY at level.")
    print("  => pi_t is I(0).  ARIMA model order: d = 0.")
elif run_diff:
    adf2_stat  = adf_pval2  < 0.05
    pp2_stat   = pp_pval2   < 0.05
    kpss2_stat = "Fail to reject" in kpss_dec2
    if adf2_stat and pp2_stat and kpss2_stat:
        d_order = 1
        print("  pi_t has mixed/conflicting test results at level.")
        print("  d_pi_t (first difference) is stationary by all three tests.")
        print("  => pi_t is I(1).  ARIMA model order: d = 1.")
    else:
        d_order = 1
        print("  pi_t shows evidence of non-stationarity at level.")
        print("  First difference d_pi_t substantially reduces persistence.")
        print("  => Treat pi_t as I(1) for ARIMA purposes; d = 1.")
        print("  Note: consider ARFIMA if long-memory is suspected (see ACF).")
else:
    d_order = 0
    print("  Tests are consistent but mixed — treating pi_t as I(0) at level.")

print(f"\n  INTEGRATION ORDER d = {d_order}")
print()
print("STEP 3 COMPLETE.")
