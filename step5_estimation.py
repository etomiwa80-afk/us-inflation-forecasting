import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# ── Load & split data ────────────────────────────────────────────────────────
df    = pd.read_csv("cpi_data.csv", index_col="date", parse_dates=True)
pi    = df["pi"].dropna()
train = pi[pi.index <= "2022-12-31"]
test  = pi[pi.index >  "2022-12-31"]

print(f"Training set : {train.index.min().date()} to {train.index.max().date()}  ({len(train)} obs)")
print(f"Test set     : {test.index.min().date()}  to {test.index.max().date()}  ({len(test)} obs)")

# ── Candidate models from Step 4 ─────────────────────────────────────────────
# Each tuple: (label, (p,d,q), (P,D,Q,s))
candidates = [
    ("A - SARIMA(2,0,0)(1,0,1)12", (2,0,0), (1,0,1,12)),
    ("B - SARIMA(3,0,0)(1,0,0)12", (3,0,0), (1,0,0,12)),
    ("C - SARIMA(2,0,1)(0,0,1)12", (2,0,1), (0,0,1,12)),
    ("D - SARIMA(1,0,1)(1,0,1)12", (1,0,1), (1,0,1,12)),
    ("E - SARIMA(3,0,1)(1,0,0)12", (3,0,1), (1,0,0,12)),
    ("F - SARIMA(2,0,0)(0,0,0)12", (2,0,0), (0,0,0,12)),
]

# ── AICc helper (statsmodels doesn't expose directly) ───────────────────────
def aicc(res):
    k = res.df_model + 1          # number of free parameters (incl. sigma2)
    n = res.nobs
    aic = res.aic
    correction = (2 * k * (k + 1)) / max(n - k - 1, 1)
    return aic + correction

# ── Fit all candidate models ─────────────────────────────────────────────────
results_store = []   # list of dicts

print()
print("=" * 70)
print("FITTING CANDIDATE MODELS  (training set: Jan 1991 – Dec 2022)")
print("=" * 70)

for label, order, seas_order in candidates:
    try:
        mod = SARIMAX(
            train,
            order=order,
            seasonal_order=seas_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(method="lbfgs", disp=False)

        aic_val  = res.aic
        aicc_val = aicc(res)
        bic_val  = res.bic

        # Convergence: check mle_retvals
        retvals   = getattr(res, "mle_retvals", {}) or {}
        converged = bool(retvals.get("converged", True))
        warn_flag = bool(retvals.get("warnflag", 0))
        conv_str  = "Yes" if (converged and not warn_flag) else "No"

        results_store.append({
            "label":     label,
            "order":     order,
            "seas":      seas_order,
            "aic":       aic_val,
            "aicc":      aicc_val,
            "bic":       bic_val,
            "converged": conv_str,
            "result":    res,
        })
        print(f"  {label:<38}  AIC={aic_val:>9.3f}  AICc={aicc_val:>9.3f}  BIC={bic_val:>9.3f}  Conv={conv_str}")

    except Exception as exc:
        print(f"  {label:<38}  FAILED: {exc}")
        results_store.append({
            "label": label, "order": order, "seas": seas_order,
            "aic": np.inf, "aicc": np.inf, "bic": np.inf,
            "converged": "No", "result": None,
        })

# ── auto_arima ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("AUTO-ARIMA  (stepwise=False, criterion=AICc, seasonal m=12)")
print("=" * 70)

auto_mod = auto_arima(
    train,
    seasonal=True, m=12,
    stepwise=False,
    information_criterion="aicc",
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    max_p=4, max_q=4, max_P=2, max_Q=2,
    d=0, D=0,                  # hold fixed per Step 3
)

auto_label = (f"Auto - SARIMA{auto_mod.order}"
              f"({auto_mod.seasonal_order[0]},{auto_mod.seasonal_order[1]},"
              f"{auto_mod.seasonal_order[2]}){auto_mod.seasonal_order[3]}")
print(f"  auto_arima selected : {auto_label}")

# Re-fit via SARIMAX so we have a statsmodels object for pickle
auto_sm = SARIMAX(
    train,
    order=auto_mod.order,
    seasonal_order=auto_mod.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(method="lbfgs", disp=False)

auto_aicc = aicc(auto_sm)
auto_retv = getattr(auto_sm, "mle_retvals", {}) or {}
auto_conv = "Yes" if (auto_retv.get("converged", True) and not auto_retv.get("warnflag", 0)) else "No"

print(f"  AIC={auto_sm.aic:>9.3f}  AICc={auto_aicc:>9.3f}  BIC={auto_sm.bic:>9.3f}  Conv={auto_conv}")

results_store.append({
    "label":     auto_label,
    "order":     auto_mod.order,
    "seas":      auto_mod.seasonal_order,
    "aic":       auto_sm.aic,
    "aicc":      auto_aicc,
    "bic":       auto_sm.bic,
    "converged": auto_conv,
    "result":    auto_sm,
})

# ── Ranked comparison table ───────────────────────────────────────────────────
ranked = sorted(
    [r for r in results_store if r["result"] is not None],
    key=lambda x: x["aicc"]
)

print()
print("=" * 90)
print("RANKED COMPARISON TABLE  (sorted by AICc, ascending)")
print("=" * 90)
hdr = f"{'Rank':<5} | {'Model':<42} | {'AIC':>10} | {'AICc':>10} | {'BIC':>10} | {'Conv':>9}"
print(hdr)
print("-" * 90)
for i, r in enumerate(ranked, 1):
    print(f"{i:<5} | {r['label']:<42} | {r['aic']:>10.3f} | {r['aicc']:>10.3f} | {r['bic']:>10.3f} | {r['converged']:>9}")
print("=" * 90)

# ── Select best converged model ───────────────────────────────────────────────
best = next((r for r in ranked if r["converged"] == "Yes"), ranked[0])

print()
print("=" * 70)
print(f"SELECTED MODEL: {best['label']}")
print(f"  Criterion: lowest AICc among converged models")
print(f"  AICc = {best['aicc']:.3f}")
print("=" * 70)

# ── Full statsmodels summary ──────────────────────────────────────────────────
print()
print(best["result"].summary())

# ── Save best model ───────────────────────────────────────────────────────────
with open("best_model.pkl", "wb") as f:
    pickle.dump(best["result"], f)
print()
print("Saved: best_model.pkl")

print()
print("STEP 5 COMPLETE.")
print()
print(f"SELECTED MODEL SPECIFICATION: {best['label']}")
