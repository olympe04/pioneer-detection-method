#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECB–SSSU Inflation Panel — HICP (ECB) + Ukraine CPI (SSSU) with ADF, Granger, and VAR
===================================================================================

Overview
--------
Single-file, reproducible script that builds a monthly inflation panel from:

1) ECB Data Portal (SDMX 2.1 REST) — HICP inflation (y/y, %), multiple countries.
2) State Statistics Service of Ukraine (SSSU) SDMX v3 — CPI index (prev. month = 100),
   converted to y/y inflation by chaining 12 monthly factors.

It then runs a compact time-series workflow suitable for teaching and quick diagnostics:
- ADF unit-root tests on inflation levels
- Bivariate Granger causality screening (predictors → target)
- Small VAR in levels with lag order selected by BIC

Key features (for readers + LLMs)
---------------------------------
- Uses official APIs (no HTML scraping).
- Explicit SDMX keys and dimensions documented in code.
- Robust handling of SSSU SDMX-CSV metadata rows (keeps only TIME_PERIOD = 'YYYY-Mmm').
- Month indexing standardized to month-start timestamps for safe merges.

Data sources
------------
ECB:  ECB Data Portal, dataset "ICP" (HICP).
      SDMX 2.1 REST pattern:
      https://data-api.ecb.europa.eu/service/data/ICP/{key}?format=csvdata&startPeriod=...&endPeriod=...

SSSU: SSSU SDMX v3 endpoint (Ukraine CPI, prev. month = 100), dataflow:
      SSSU / DF_PRICE_CHANGE_CONSUMER_GOODS_SERVICE / version "~"
      key: INDEX_CONSUMPRICE.PREV_MONTH.UA00000000000000000.0.M

Econometric workflow (teaching level)
-------------------------------------
- ADF test (H0: unit root) on each inflation series (levels).
- Granger causality tests (bivariate): does X help predict the target series?
  Ranking uses the minimum p-value across lags 1..maxlag.
- VAR: target + top 2 Granger predictors; lag order chosen by BIC; VAR in levels.

Outputs
-------
- Multi-line plot of the panel (incl. 0-line).
- Console tables:
  * ADF stats/p-values
  * Granger ranking
  * VAR lag selection (BIC) and estimation summary

Dependencies
------------
requests, pandas, numpy, matplotlib, statsmodels

Author / License
----------------
Eric Vansteenberghe (Banque de France)
Created: 2026-01-24
License: MIT (recommended for teaching code)


Notes
-----
- This script uses revised (latest) data, not real-time vintages.
- Missing values are handled with complete-case deletion prior to estimation.
"""



import requests
import pandas as pd
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

def fetch_ecb_hicp_inflation_panel(
    countries,
    start="1997-01-01",
    end=None,
    item="000000",   # headline all-items HICP
    sa="N",          # neither seasonally nor working-day adjusted
    measure="4",     # percentage change (as used in ICP keys)
    variation="ANR", # annual rate of change
    freq="M",
    timeout=60
):
    """
    Fetch a monthly cross-country panel of HICP inflation (annual rate of change)
    from the ECB Data Portal (ICP dataflow).

    Returns
    -------
    panel_wide : pd.DataFrame
        Index: pandas datetime (monthly)
        Columns: country codes (e.g., DE, FR, IT)
        Values: inflation rate (float)
    raw_long : pd.DataFrame
        Long format with series dimensions, TIME_PERIOD and OBS_VALUE.
    """
    # ECB Data Portal SDMX REST endpoint
    base = "https://data-api.ecb.europa.eu/service/data"

    # Build SDMX series key with OR operator (+) over countries
    # Dimension order for ICP: FREQ.REF_AREA.ADJ.ITEM.UNIT/MEASURE.VARIATION
    # Example keys are shown in the ECB portal for ICP datasets.
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"

    params = {"format": "csvdata", "startPeriod": start}
    if end is not None:
        params["endPeriod"] = end

    url = f"{base}/ICP/{key}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))

    # Keep standard SDMX columns
    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError(f"Unexpected response format. Columns: {list(raw.columns)}")

    # Identify the country dimension column (typically REF_AREA)
    # If REF_AREA is missing, fall back to any column that looks like a geo dimension.
    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        for cand in ["GEO", "LOCATION", "COUNTRY", "REF_AREA"]:
            if cand in raw.columns:
                country_col = cand
                break
    if country_col is None:
        # Last resort: infer as the first non-standard column
        standard = {"TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "OBS_CONF", "UNIT_MULT", "DECIMALS"}
        nonstandard = [c for c in raw.columns if c not in standard]
        if not nonstandard:
            raise ValueError("Could not infer the country column from the response.")
        country_col = nonstandard[0]

    # Parse time and values
    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    # Wide panel: time x country
    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
    )

    return panel, raw


# -------------------------
# Example usage
# -------------------------
countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
infl_panel, infl_long = fetch_ecb_hicp_inflation_panel(
    countries=countries,
    start="2000-01",
    end="2025-12"   # optional
)
#%%
# -----------------------------------
# Fetch Ukraine inflation time series

def fetch_ukraine_cpi_prev_month_raw(
    start="2000-01",
    end="2025-12",
    timeout=60
):
    """
    Fetch Ukraine CPI (previous month = 100) from the SSSU SDMX API v3 and return
    the raw SDMX-CSV as a DataFrame (no date/numeric parsing).
    """
    base = "https://stat.gov.ua/sdmx/workspaces/default:integration/registry/sdmx/3.0/data"
    agency = "SSSU"
    flow = "DF_PRICE_CHANGE_CONSUMER_GOODS_SERVICE"
    version = "~"
    key = "INDEX_CONSUMPRICE.PREV_MONTH.UA00000000000000000.0.M"

    url = f"{base}/dataflow/{agency}/{flow}/{version}/{key}"
    params = {"c[TIME_PERIOD]": f"ge:{start}+le:{end}"}
    headers = {
        "Accept": "application/vnd.sdmx.data+csv;version=2.0.0;labels=id;timeFormat=normalized;keys=both",
        "User-Agent": "Mozilla/5.0",
    }

    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text), dtype=str)

    # --- MINIMAL FIX: some responses include metadata rows.
    # Keep only rows that look like monthly observations and have OBS_VALUE.
    raw = raw.loc[
        raw["TIME_PERIOD"].astype(str).str.match(r"^\d{4}-M\d{2}$", na=False)
        & raw["OBS_VALUE"].notna()
    ].copy()

    return raw


# Example
ua_raw = fetch_ukraine_cpi_prev_month_raw(start="2000-01", end="2025-12")
print(ua_raw.head())
print(ua_raw["TIME_PERIOD"].unique()[:12])
print(ua_raw["OBS_VALUE"].unique()[:12])



# ua_raw is your DataFrame as read from the SDMX-CSV response
# (i.e., it already has columns like TIME_PERIOD, OBS_VALUE)

def ua_raw_to_monthly_series(ua_raw: pd.DataFrame) -> pd.Series:
    """
    Build a clean monthly time series from SSSU SDMX-CSV raw output.

    Input:
      ua_raw: DataFrame with at least TIME_PERIOD like '2000-M01' and OBS_VALUE strings.

    Output:
      pd.Series indexed by month-start Timestamp, name='UA_IDX_PREV_MONTH_100'
    """
    if "TIME_PERIOD" not in ua_raw.columns or "OBS_VALUE" not in ua_raw.columns:
        raise ValueError(f"ua_raw must contain TIME_PERIOD and OBS_VALUE. Columns: {list(ua_raw.columns)}")

    s = ua_raw[["TIME_PERIOD", "OBS_VALUE"]].copy()

    # Keep only true monthly tokens like YYYY-Mmm (defensive)
    s["TIME_PERIOD"] = s["TIME_PERIOD"].astype(str).str.strip()
    s = s[s["TIME_PERIOD"].str.match(r"^\d{4}-M\d{2}$", na=False)]

    # Convert 'YYYY-Mmm' -> Timestamp at month start
    # Example: '2000-M01' -> '2000-01-01'
    s["TIME_PERIOD"] = pd.to_datetime(
        s["TIME_PERIOD"].str.replace(r"^(\d{4})-M(\d{2})$", r"\1-\2-01", regex=True),
        errors="coerce"
    )

    # Values
    s["OBS_VALUE"] = pd.to_numeric(s["OBS_VALUE"].astype(str).str.replace(",", ".", regex=False),
                                   errors="coerce")

    s = s.dropna(subset=["TIME_PERIOD", "OBS_VALUE"]).sort_values("TIME_PERIOD")

    out = s.set_index("TIME_PERIOD")["OBS_VALUE"].rename("UA_IDX_PREV_MONTH_100")

    # If duplicates exist for a month (shouldn't, but safe): keep last
    out = out.groupby(level=0).last()

    return out

# Build the monthly series (prev month = 100)
ua_idx = ua_raw_to_monthly_series(ua_raw)

# Optional: restrict window (month-start)
ua_idx = ua_idx.loc["2000-01-01":"2025-12-01"]

# If you still need y/y inflation (%):
def cpi_prev_month_index_to_yoy_inflation(idx_prev_month_100: pd.Series) -> pd.Series:
    monthly_factor = (idx_prev_month_100 / 100.0).astype(float)
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    return ((yoy_factor - 1.0) * 100.0).rename("UA")

ua_yoy = cpi_prev_month_index_to_yoy_inflation(ua_idx)
#%%
# Ensure month-start indices match
infl_panel = infl_panel.copy()
infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")
#ua_yoy.index = pd.to_datetime(ua_yoy.index).to_period("M").to_timestamp(how="start")

#infl_panel = infl_panel.join(ua_yoy, how="left")

#%%
# ------------------------------------------------------------
# Plot the inflation panel (one line per country)
# Assumes `infl_panel` is the wide DataFrame returned above:
#   index   = datetime (monthly)
#   columns = country codes
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for country in infl_panel.columns:
    plt.plot(infl_panel.index, infl_panel[country], label=country, linewidth=1)

plt.axhline(0, color="black", linewidth=0.8, linestyle="--")

plt.xlabel("Time")
plt.ylabel("Inflation rate (y/y, %)")
plt.title("HICP Inflation Panel (ECB Data Portal)")
plt.legend(ncol=3, fontsize=9, frameon=False)
plt.tight_layout()
plt.show()


# -------------------------
# 0) Prepare data
# -------------------------
df = infl_panel.copy().sort_index().dropna()

# -------------------------
# 1) ADF unit-root test (levels only)
# -------------------------
print("\n=== ADF unit-root tests (levels) ===")

adf_results = []
for c in df.columns:
    stat, pval, _, _, _, _ = adfuller(df[c], autolag="AIC")
    adf_results.append({
        "country": c,
        "ADF_stat": stat,
        "pvalue": pval
    })

adf_table = pd.DataFrame(adf_results).sort_values("pvalue")
print(adf_table.to_string(index=False))

# -------------------------
# 2) Granger causality: X → UA
#    (bivariate, simple ranking)
# -------------------------
maxlag = 6   # keep small for undergrads

print("\n=== Granger causality tests: X → FR ===")

granger_out = []

for c in df.columns:
    if c == "FR":
        continue

    data_gc = df[["FR", c]]

    try:
        res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)

        # keep the smallest p-value across lags
        min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))

        granger_out.append({
            "country": c,
            "min_pvalue": min_p
        })

    except Exception as e:
        print(f"Granger test failed for {c}: {e}")

granger_rank = (
    pd.DataFrame(granger_out)
    .sort_values("min_pvalue")
    .reset_index(drop=True)
)

print("\n=== Ranking of countries by Granger causality for FR ===")
print(granger_rank.to_string(index=False))

# -------------------------
# 3) Simple VAR with BIC
#    (UA + top 2 predictors)
# -------------------------
top_countries = granger_rank["country"].iloc[:2].tolist()
var_vars = ["FR"] + top_countries

print("\nVAR variables:", var_vars)

X_var = df[var_vars]

# lag selection by BIC
model = VAR(X_var)
lag_selection = model.select_order(maxlags=6)
p = lag_selection.selected_orders["bic"]
p = max(1, p)

print("\n=== VAR lag selection (BIC) ===")
print(lag_selection.summary())
print(f"Selected lag order p = {p}")

# estimate VAR
var_res = model.fit(p)
print("\n=== VAR estimation results ===")
print(var_res.summary())
#%%
# -----------------------------------
# (c) Compute pioneer weights and plot
# -----------------------------------

# Assuming you have the function compute_pioneer_weights_angles defined somewhere,
# and that it accepts a T x N DataFrame (time x countries) and returns a DataFrame of same shape

# After running ecb_hicp_panel_var_granger.py, you have infl_panel
# with columns: DE, FR, IT, ES, NL, BE, AT, PT, IE, FI, GR, UA
#from pdm import compute_pioneer_weights_angles,compute_pioneer_weights_distance, compute_granger_weights, compute_lagged_correlation_weights,compute_multivariate_regression_weights, compute_transfer_entropy_weights, compute_linear_pooling_weights,compute_median_pooling,pooled_forecast


def _leave_one_out_mean(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the leave-one-out mean for each expert.

    For expert i, this is the mean of all other experts at each time step.
    Used internally by all methods as the cross-sectional benchmark m_{-i}.

    Parameters
    ----------
    X : pd.DataFrame
        (T x N) matrix of expert estimates, already cast to float.

    Returns
    -------
    m_minus : pd.DataFrame
        Same shape as X. Column i contains the mean of all columns except i.
    """
    m_minus = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for col in X.columns:
        others = X.drop(columns=col)
        m_minus[col] = others.mean(axis=1)
    return m_minus


# ---------------------------------------------------------------------------
# 2. PDM with angles (preferred method)
# ---------------------------------------------------------------------------

def compute_pioneer_weights_angles(
    forecasts: pd.DataFrame,
    step: float = 1.0,
) -> pd.DataFrame:
    """
    PDM with angle-based weighting (Equation 4-5 of the paper).

    The angle theta between the movement vector and the horizontal captures
    the *speed* of convergence. The weight attributed to expert i is:
        w_i^t = delta_distance * delta_orientation * |theta_{-i}| / (|theta_{-i}| + |theta_i|)

    where theta = arccos((s^2 + u_y * v_y) / (sqrt(s^2 + u_y^2) * sqrt(s^2 + v_y^2)))
    and s is the time step between observations.

    This is the preferred approach in the paper.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) DataFrame where rows are time periods and columns are experts.
        Values must be numeric (int or float).
    step : float
        Time step between observations (the x-component of both vectors).
        Default is 1.0 for unit-spaced observations. Set to the actual
        inter-observation interval when observations are not unit-spaced
        (e.g., 12 for annual data indexed monthly).

    Returns
    -------
    weights : pd.DataFrame
        Same shape as ``forecasts``. Contains normalised pioneer weights in
        [0, 1] that sum to 1 across experts at each time step where at least
        one pioneer is detected.  Rows with no pioneer contain NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> forecasts = pd.DataFrame({
    ...     "E1": [1.0, 1.1, 1.2, 1.3],
    ...     "E2": [0.5, 0.5, 0.9, 1.2],
    ...     "E3": [0.4, 0.4, 0.8, 1.1],
    ... })
    >>> w = compute_pioneer_weights_angles(forecasts)
    >>> pooled = pooled_forecast(forecasts, w)
    """
    X = forecasts.astype(float)
    m_minus = _leave_one_out_mean(X)

    delta_X = X.diff()
    delta_m = m_minus.diff()

    # Step 1: distance reduction
    distance = (X - m_minus).abs()
    distance_prev = distance.shift(1)
    cond_distance = distance < distance_prev

    # Step 2: orientation (peers move more — checked via angles)
    s2 = step ** 2

    def _angle(dy):
        """Angle between the movement vector (step, dy) and horizontal (step, 0)."""
        # theta = arccos(s^2 / (sqrt(s^2 + dy^2) * s))  = arctan(|dy| / s)
        return np.arctan2(dy.abs(), step)

    theta_i = _angle(delta_X)    # expert's own movement angle
    theta_mi = _angle(delta_m)   # peers' movement angle

    cond_orientation = theta_mi > theta_i

    # Step 3: proportion (angle-based)
    denom = theta_mi + theta_i
    proportion = theta_mi / denom

    mask = cond_distance & cond_orientation & (denom > 0)
    raw = proportion.where(mask, 0.0)

    row_sums = raw.sum(axis=1)
    weights = raw.div(row_sums.replace(0.0, np.nan), axis=0)
    return weights

def pooled_forecast(
    forecasts: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute the supervisor's pooled estimate: S_t = sum_i w_i^t * x_i^t.

    At time steps where no pioneer is detected (all weights are NaN or sum
    to zero), the pooled estimate falls back to the simple cross-sectional
    mean.  This fallback corresponds to the initialisation rule w_i^0 = 1/m
    described in the paper.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) expert forecasts.
    weights : pd.DataFrame
        (T x N) weights produced by any of the weight-computation functions.

    Returns
    -------
    pooled : pd.Series
        Length-T pooled estimate.

    Examples
    --------
    >>> w = compute_pioneer_weights_angles(forecasts)
    >>> pooled = pooled_forecast(forecasts, w)
    """
    forecasts = forecasts.astype(float)
    weights = weights.astype(float)

    weighted_sum = (forecasts * weights).sum(axis=1, min_count=1)

    weight_sums = weights.sum(axis=1, min_count=1)
    no_pioneer = weight_sums.isna() | (weight_sums == 0)

    fallback_mean = forecasts.mean(axis=1)

    pooled = weighted_sum.copy()
    pooled[no_pioneer] = fallback_mean[no_pioneer]
    return pooled


# Backward-compatible alias
pooled_forecast_simple = pooled_forecast


#---PartA: full panel, all countries as "experts"--
panel = infl_panel.dropna()
w_angles = compute_pioneer_weights_angles(panel)
# Average weights by subperiod
periods = {
    "I (2002-07)":("2002-01", "2007-12"),
    "II (2008-12)":("2008-01", "2012-12"),
    "III (2013-19)":("2013-01", "2019-12"),
    "IV (2020-21)":("2020-01", "2021-12"),
    "V (2022-23)":("2022-01", "2023-12"),
    "VI (2024-25)":("2024-01", "2025-12"),
}
for name, (s,e) in periods.items():
    sub = w_angles.loc[s:e]
    print(f"\n{name}")
    print(sub.mean().sort_values(ascending=False))
#---PartB: EU countries as experts, Ukraine as target--
eu_cols = [c for c in panel.columns if c != "FR"]
eu_panel = panel[eu_cols]
w_eu = compute_pioneer_weights_angles(eu_panel)
forecast_ua =pooled_forecast(eu_panel, w_eu)

# RMSE vsactual Ukraine inflation
import numpy as np
actual_ua = panel["FR"]
rmse = np.sqrt(((forecast_ua- actual_ua)** 2).mean())
print(f"RMSE (PDM angles): {rmse:.4f}")

# Apply to the inflation panel
pioneer_weights = compute_pioneer_weights_angles(df)  # T x 12 DataFrame

# Inspect a snippet
print(pioneer_weights.head())

# Plot each country's pioneer weight over time
plt.figure(figsize=(12, 6))

for country in pioneer_weights.columns:
    plt.plot(pioneer_weights.index, pioneer_weights[country], label=country, linewidth=1)

plt.xlabel("Time")
plt.ylabel("Pioneer weight")
plt.title("Time-varying Pioneer Weights of HICP Inflation (ECB Panel)")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.legend(ncol=3, fontsize=9, frameon=False)
plt.tight_layout()
plt.show()

# Alternatively, a heatmap
import seaborn as sns

plt.figure(figsize=(14, 6))
sns.heatmap(pioneer_weights.T, cmap="coolwarm", center=0, cbar_kws={'label': 'Pioneer weight'})
plt.xlabel("Time")
plt.ylabel("Country")
plt.title("Heatmap of Pioneer Weights over Time")
plt.tight_layout()
plt.show()

# Identify countries with non-zero pioneer weights
nonzero_summary = (pioneer_weights != 0).sum()
print("Number of months with non-zero pioneer weight per country:")
print(nonzero_summary[nonzero_summary > 0])
