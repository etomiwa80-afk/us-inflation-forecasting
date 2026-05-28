# U.S. Inflation Forecasting - SARIMA Analysis

## Overview
Modeling U.S. inflation dynamics using monthly CPI data (1990-2024) from the Federal Reserve (FRED).

## Results
- Model: SARIMA(1,0,1)(1,0,1)12
- MAE: 0.646 percentage points on 24-month holdout
- 76.7% improvement over Holt-Winters ETS benchmark
- 2025 inflation projected between 2.4-3.3% above Fed 2% target

## Tools
Python, statsmodels, pandas, NumPy, matplotlib

## Structure
- step1_cpi.py - data loading
- step2_eda.py - exploratory analysis
- step3_stationarity.py - unit root tests
- step4_acf_pacf.py - model identification
- step5_estimation.py - model fitting
- step6_diagnostics.py - residual diagnostics
- step7_forecast.py - forecasting
- step8_evaluation.py - performance evaluation
- step9_summary.py - results summary
