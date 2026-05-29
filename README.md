# U.S. Inflation Forecasting

SARIMA model on monthly CPI data from the Federal Reserve (FRED), 1990-2024.

## Results

- Model: SARIMA(1,0,1)(1,0,1)12
- MAE: 0.646 percentage points on a 24-month holdout
- Benchmark: 76.7% improvement over Holt-Winters ETS
- 2025 projection: 2.4-3.3% (above the Fed's 2% target)

## Method

CPI data was pulled from FRED and converted to year-over-year inflation rates. ADF and KPSS tests confirmed stationarity. ACF/PACF plots guided model order selection. Ljung-Box, residual ACF/PACF, and normality checks confirmed no remaining structure in residuals.

## Tools

Python, statsmodels, pandas, NumPy, matplotlib

## Data

Monthly CPI data from FRED (https://fred.stlouisfed.org/series/CPIAUCSL), series CPIAUCSL.
