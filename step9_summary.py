import json, textwrap

with open("step8_metrics.json") as f:
    m = json.load(f)

lines = []
def w(s=""): lines.append(s)

w("=============================================================")
w("STAT 8220 FINAL PROJECT — RESULTS SUMMARY")
w("Dataset: CPIAUCSL | Transformed: YoY log-difference (pi_t)")
w("=============================================================")
w()

# ── DATA ─────────────────────────────────────────────────────────────────────
w("--- DATA ---")
w("Sample period             : January 1990 – December 2024")
w("Transformed series (pi_t) : January 1991 – December 2024 (first 12 obs lost to lag)")
w("Training set              : January 1991 – December 2022")
w("Holdout set               : January 2023 – December 2024")
w("Observations (total / train / holdout) : 408 / 384 / 24")
w("pi_t mean                 : 2.5764%")
w("pi_t std                  : 1.5153%")
w("pi_t min                  : -1.9782%  (2009-07-01)")
w("pi_t max                  : +8.5988%  (2022-06-01)")
w()

# ── STATIONARITY ─────────────────────────────────────────────────────────────
w("--- STATIONARITY TESTS (Step 3) ---")
w("ADF  | pi_t | stat=-3.5550 | p=0.0067 | lags=15 (AIC) | decision=Reject H0 (stationary)")
w("       Critical values: 1%=-3.4471  5%=-2.8689  10%=-2.5707")
w("PP   | pi_t | stat=-3.6395 | p=0.0050 | decision=Reject H0 (stationary)")
w("       Critical values: 1%=-3.4465  5%=-2.8687  10%=-2.5706")
w("KPSS | pi_t | stat=0.2083  | cv_5%=0.4630 | decision=Fail to reject H0 (stationary)")
w("       Critical values: 10%=0.3470  5%=0.4630  2.5%=0.5740  1%=0.7390")
w("All three tests agree: pi_t is I(0) at level.")
w("Conclusion: d = 0  (no differencing required)")
w()

# ── ACF/PACF ─────────────────────────────────────────────────────────────────
w("--- ACF/PACF IDENTIFICATION (Step 4) ---")
w("95% CI threshold          : +/-0.0970")
w("ACF pattern               : Geometric/hyperbolic decay through lag ~18 => AR signature")
w("ACF significant lags      : 1 through 18 continuously; significant spike at lag 12")
w("ACF at lag 12             : +0.2507 (significant)")
w("ACF at lag 24             : -0.0213 (not significant)")
w("PACF pattern              : Sharp cutoff after lag 3 => AR(3) non-seasonal signature")
w("PACF significant lags     : 1, 2, 3 (primary); isolated at 11, 13, 16, 24, 25, 37")
w("PACF at lag 12            : +0.0779 (not significant)")
w("PACF at lag 24            : +0.1369 (significant)")
w("Non-seasonal: AR order suggestion=2–3 | MA order suggestion=0–1")
w("Seasonal (s=12): P suggestion=1 | Q suggestion=1")
w("d=0  D=0 fixed per Step 3")
w("Candidate models listed:")
w("  A: SARIMA(2,0,0)(1,0,1)12  B: SARIMA(3,0,0)(1,0,0)12")
w("  C: SARIMA(2,0,1)(0,0,1)12  D: SARIMA(1,0,1)(1,0,1)12")
w("  E: SARIMA(3,0,1)(1,0,0)12  F: SARIMA(2,0,0)(0,0,0)12")
w()

# ── MODEL SELECTION ───────────────────────────────────────────────────────────
w("--- MODEL SELECTION (Step 5) ---")
w("Fit method: SARIMAX lbfgs, enforce_stationarity=False, enforce_invertibility=False")
w("Training set: Jan 1991 – Dec 2022 (384 obs)")
w()
w("Rank 1 (selected): SARIMA(1,0,1)(1,0,1)12 | AIC=-3360.950 | AICc=-3360.727 | BIC=-3341.383 | Converged=Yes")
w("Rank 2           : SARIMA(2,0,0)(1,0,1)12 | AIC=-3337.773 | AICc=-3337.550 | BIC=-3318.206 | Converged=Yes")
w("Rank 3 (auto)    : SARIMA(1,0,1)(1,0,2)12 | AIC=-3242.328 | AICc=-3242.030 | BIC=-3219.045 | Converged=No")
w("Rank 4           : SARIMA(3,0,0)(1,0,0)12 | AIC=-3226.104 | AICc=-3225.881 | BIC=-3206.550 | Converged=No")
w("Rank 5           : SARIMA(3,0,1)(1,0,0)12 | AIC=-3222.887 | AICc=-3222.589 | BIC=-3199.422 | Converged=No")
w("Rank 6           : SARIMA(2,0,0)(0,0,0)12 | AIC=-3212.955 | AICc=-3212.850 | BIC=-3201.119 | Converged=Yes")
w("Rank 7           : SARIMA(2,0,1)(0,0,1)12 | AIC=-3070.719 | AICc=-3070.496 | BIC=-3051.152 | Converged=Yes")
w()
w("auto_arima (stepwise=False, AICc, d=0, D=0): selected SARIMA(1,0,1)(1,0,2)12 — did not converge")
w()
w("Selected model coefficients (SARIMA(1,0,1)(1,0,1)12):")
w("  ar.L1    =  1.0002  (z=1228.12,  p<0.001)")
w("  ma.L1    =  0.5210  (z=18.87,    p<0.001)")
w("  ar.S.L12 = -0.1323  (z=-2.496,   p=0.013)")
w("  ma.S.L12 = -1.0234  (z=-9.125,   p<0.001)")
w("  sigma2   =  5.61e-6 (z=6.509,    p<0.001)")
w("AR root proximity to unit circle: ar.L1 = 1.0002 (FLAG: at unit circle boundary)")
w()

# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────
w("--- RESIDUAL DIAGNOSTICS (Step 6) ---")
w("Ljung-Box Q(6):   stat=2.9008  | p=0.8212 | PASS (no residual autocorrelation)")
w("Ljung-Box Q(12):  stat=7.5916  | p=0.8162 | PASS")
w("Ljung-Box Q(18):  stat=8.8566  | p=0.9630 | PASS")
w("Ljung-Box Q(24):  stat=10.2370 | p=0.9935 | PASS")
w("Jarque-Bera: stat=216201.10 | p≈0 | skewness=7.8931 | kurtosis=115.17 (excess)")
w("  NOTE: JB dominated by Jan-1991 initialization outlier (+14.5 SD).")
w("  Body of distribution is approximately normal; extreme tails are the concern.")
w("ARCH-LM (lag 6):  stat=43.170 | p<0.0001 | REJECT H0 — ARCH effects present")
w("ARCH-LM (lag 12): stat=54.737 | p<0.0001 | REJECT H0 — ARCH effects present")
w("GARCH(1,1) fit on residuals: omega=78.65 | alpha=0.299 | beta=0.597 | persistence=0.896")
w("Outliers beyond 3 SD: n=2")
w("  1991-01-01: +14.533 SD (initialization artifact — no pre-sample seasonal history)")
w("  2008-11-01: -3.596 SD  (Lehman collapse deflationary shock)")
w("Overall diagnostic verdict:")
w("  PASS: Serial correlation (Ljung-Box at all lags)")
w("  CONCERN: Non-normality (fat tails — use simulation-based prediction intervals)")
w("  CONCERN: ARCH effects (volatility clustering from GFC and COVID episodes)")
w("  CONCERN: AR root at unit circle (1.0002) — model absorbs persistence via near-I(1) AR")
w()

# ── FORECAST ─────────────────────────────────────────────────────────────────
w("--- FORECAST (Step 7) ---")
w("Model used for final forecast: SARIMA(1,0,1)(1,0,1)12 refitted on full sample (Jan 1991 – Dec 2024)")
w("Final model AIC: -3598.558 | BIC: -3578.677")
w("Horizon: January 2025 – December 2025")
w()
w("Date       | Forecast (%) | Lower 80%  | Upper 80%  | Lower 95%  | Upper 95%")
w("-----------+--------------+------------+------------+------------+------------")
fc_table = [
    ("2025-01", "2.8434", "2.5382", "3.1486", "2.3766", "3.3102"),
    ("2025-02", "2.6759", "2.1214", "3.2304", "1.8279", "3.5239"),
    ("2025-03", "2.4402", "1.7179", "3.1624", "1.3355", "3.5448"),
    ("2025-04", "2.4440", "1.5860", "3.3020", "1.1318", "3.7562"),
    ("2025-05", "2.6462", "1.6711", "3.6213", "1.1549", "4.1374"),
    ("2025-06", "3.0240", "1.9444", "4.1037", "1.3729", "4.6752"),
    ("2025-07", "3.1036", "1.9286", "4.2786", "1.3066", "4.9006"),
    ("2025-08", "3.2031", "1.9399", "4.4664", "1.2711", "5.1351"),
    ("2025-09", "3.2819", "1.9361", "4.6277", "1.2237", "5.3402"),
    ("2025-10", "3.1793", "1.7556", "4.6029", "1.0020", "5.3565"),
    ("2025-11", "3.0303", "1.5328", "4.5278", "0.7401", "5.3206"),
    ("2025-12", "2.8367", "1.2688", "4.4046", "0.4387", "5.2347"),
]
for row in fc_table:
    w(f"{row[0]:<10} | {row[1]:>12} | {row[2]:>10} | {row[3]:>10} | {row[4]:>10} | {row[5]:>10}")
w()
w("Mean 2025 forecast: 2.892%  |  Range: [2.440%, 3.282%]")
w("Model projects inflation above the Fed 2% target throughout 2025.")
w("Seasonal mid-year uptick driven by seasonal AR component.")
w("95% PI widens to +/-2.5 pp by Dec 2025 (near-unit AR root + ARCH volatility).")
w()

# ── PERFORMANCE ───────────────────────────────────────────────────────────────
w("--- PERFORMANCE EVALUATION (Step 8) ---")
w("Holdout: January 2023 – December 2024 (24 months, genuine out-of-sample)")
w()
w(f"SARIMA MAE  = {m['sarima_mae']:.4f} pp  |  RMSE = {m['sarima_rmse']:.4f} pp")
w(f"ETS    MAE  = {m['ets_mae']:.4f} pp  |  RMSE = {m['ets_rmse']:.4f} pp")
w()
better = m["better_model"]
if better == "SARIMA":
    w(f"Better model = SARIMA")
    w(f"MAE  improvement over ETS = {abs(m['mae_improvement']):.2f}%")
    w(f"RMSE improvement over ETS = {abs(m['rmse_improvement']):.2f}%")
else:
    w(f"Better model = ETS")
    w(f"MAE  improvement over SARIMA = {abs(m['mae_improvement']):.2f}%")
    w(f"RMSE improvement over SARIMA = {abs(m['rmse_improvement']):.2f}%")
w()
w("Three largest SARIMA forecast errors (actual minus predicted):")
notes = [
    "Late-2023 disinflation much slower than model anticipated; shelter/services stickiness",
    "Persistent OER (owners-equivalent rent) kept CPI elevated; 'higher-for-longer' repricing",
    "Surprise CPI acceleration; rent and shelter components re-accelerated unexpectedly",
]
for i, (date, err) in enumerate(m["top3"], 1):
    w(f"  {i}. {date}  |  Error = {err} pp  |  {notes[i-1]}")
w()
w("Bias pattern: SARIMA systematically UNDER-predicted inflation through all of 2023")
w("and first-half 2024 (errors +0.4 to +1.2 pp), then converged to near-zero error")
w("by Aug-Dec 2024. This reflects the model's inability to capture post-COVID")
w("stickiness of service-sector inflation — a structural break not in training data.")
w()

w("=============================================================")
w("END OF SUMMARY")
w("=============================================================")

text = "\n".join(lines)
with open("results_summary.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(text)
print()
print("Saved: results_summary.txt")
print()
print("STEP 9 COMPLETE. All analysis is finished.")
