"""
compute_ctl_atl.py
------------------
Extracts Chronic Training Load (CTL) and Acute Training Load (ATL)
from a Strava activities CSV export, then produces a cleaned dataset
ready to feed into the GP surrogate (gp_surrogate.py).

What CTL and ATL are
--------------------
Both are exponentially weighted moving averages of daily training load
(TRIMP — Training Impulse).  The only difference is the time constant:

    CTL  (τ = 42 days)  — "fitness":  slow-moving, reflects long-term
                          aerobic adaptation.  A four-mile run at CTL=60
                          feels easier than at CTL=20.

    ATL  (τ = 7 days)   — "fatigue":  fast-moving, reflects recent
                          accumulated stress.  High ATL → you're tired.

At each day d:
    CTL_d = CTL_{d-1} + (TRIMP_d - CTL_{d-1}) * (1 - exp(-1/42))
    ATL_d = ATL_{d-1} + (TRIMP_d - ATL_{d-1}) * (1 - exp(-1/7))

On rest days TRIMP_d = 0, so both decay toward zero automatically.

TRIMP source (in priority order)
---------------------------------
1. Strava "Relative Effort"  — HR-zone-weighted load, Strava's own TRIMP.
2. (Moving time in min) × (avg HR / 100)  — simple Banister TRIMP proxy.
3. Distance in km  — last resort when HR is unavailable.

Output columns (gp_ready_runs.csv)
-----------------------------------
date, distance_km, pace_min_per_km, elevation_gain_m,
avg_hr, max_hr, trimp, CTL, ATL
"""

import numpy as np
import pandas as pd

# ==============================================================================
#  CONFIG
# ==============================================================================

CSV_PATH   = "activities.csv"       # your Strava export
OUT_PATH   = "gp_ready_runs.csv"    # cleaned output for the GP

CTL_TAU    = 42.0   # days — fitness time constant
ATL_TAU    = 7.0    # days — fatigue time constant

# Exponential decay factors (computed once)
K_CTL = 1 - np.exp(-1.0 / CTL_TAU)
K_ATL = 1 - np.exp(-1.0 / ATL_TAU)

HR_MAX_CUTOFF = 220   # drop rows where max HR looks like a sensor error


# ==============================================================================
#  LOAD & FILTER TO RUNS
# ==============================================================================

def load_runs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    runs = df[df["Activity Type"] == "Run"].copy()
    runs["date"] = pd.to_datetime(
        runs["Activity Date"], format="%b %d, %Y, %I:%M:%S %p"
    )
    runs = runs.sort_values("date").reset_index(drop=True)

    # Keep only runs that have heart rate — drops pre-2020 data where HR
    # was unavailable, so CTL/ATL are built entirely on clean TRIMP values
    before = len(runs)
    runs = runs[runs["Average Heart Rate"].notna()].reset_index(drop=True)
    print(f"Loaded {before} run activities, kept {len(runs)} with HR data  "
          f"({runs['date'].min().date()} → {runs['date'].max().date()})")
    return runs


# ==============================================================================
#  TRIMP  (training load proxy per activity)
# ==============================================================================

def compute_trimp(runs: pd.DataFrame) -> pd.Series:
    """
    Return a Series of TRIMP values aligned to runs.index.

    Priority:
      1. Strava Relative Effort  (HR-zone weighted, best option)
      2. Banister proxy: (moving_time_min) × (avg_hr / 100)
      3. Distance proxy: distance_km  (no HR data at all)
    """
    trimp = runs["Relative Effort"].copy()

    # Fallback 2: Banister proxy
    mask2 = trimp.isna() & runs["Average Heart Rate"].notna()
    trimp.loc[mask2] = (
        (runs.loc[mask2, "Moving Time"] / 60.0)   # seconds → minutes
        * (runs.loc[mask2, "Average Heart Rate"] / 100.0)
    )

    # Fallback 3: distance proxy
    mask3 = trimp.isna()
    trimp.loc[mask3] = runs.loc[mask3, "Distance.1"] / 1000.0  # m → km

    n_re  = runs["Relative Effort"].notna().sum()
    n_ban = mask2.sum()
    n_dst = mask3.sum()
    print(f"TRIMP sources — Relative Effort: {n_re}, "
          f"Banister proxy: {n_ban}, Distance proxy: {n_dst}")

    return trimp.fillna(0.0)


# ==============================================================================
#  CTL / ATL  via daily EMA
# ==============================================================================

def compute_ctl_atl(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily TRIMP series spanning the full date range, then compute
    CTL and ATL as exponentially weighted moving averages.

    Returns the daily DataFrame (one row per calendar day) with columns:
        date, daily_trimp, CTL, ATL
    """
    runs = runs.copy()
    runs["trimp"] = compute_trimp(runs)
    runs["day"]   = runs["date"].dt.normalize()   # strip time → date only

    # Sum TRIMP within the same day (two runs in one day)
    daily = (
        runs.groupby("day")["trimp"]
        .sum()
        .rename("daily_trimp")
        .reset_index()
        .rename(columns={"day": "date"})
    )

    # Expand to every calendar day (rest days = 0 TRIMP)
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_range, fill_value=0.0)
        .rename_axis("date")
        .reset_index()
    )

    # EMA pass
    ctl_vals = np.zeros(len(daily))
    atl_vals = np.zeros(len(daily))
    ctl, atl = 0.0, 0.0
    for i, row in enumerate(daily.itertuples()):
        ctl = ctl + (row.daily_trimp - ctl) * K_CTL
        atl = atl + (row.daily_trimp - atl) * K_ATL
        ctl_vals[i] = ctl
        atl_vals[i] = atl

    daily["CTL"] = ctl_vals
    daily["ATL"] = atl_vals
    return daily


# ==============================================================================
#  CLEAN FEATURE COLUMNS
# ==============================================================================

def build_feature_df(runs: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """
    Join per-run features with the CTL/ATL values from the *start* of that day
    (i.e. before the run contributes), then clean and rename columns.
    """
    runs = runs.copy()
    runs["day"] = runs["date"].dt.normalize()

    # Shift daily CTL/ATL by one day so we use the value *before* today's run
    daily_shifted = daily.copy()
    daily_shifted["date"] = daily_shifted["date"] + pd.Timedelta(days=1)
    daily_shifted = daily_shifted.rename(columns={"date": "day"})

    merged = runs.merge(daily_shifted[["day", "CTL", "ATL"]], on="day", how="left")

    # Derive pace (min / km)  from moving time and distance
    merged["distance_km"]       = merged["Distance.1"] / 1000.0
    merged["moving_time_min"]   = merged["Moving Time"] / 60.0
    merged["pace_min_per_km"]   = (
        merged["moving_time_min"] / merged["distance_km"]
    )

    out = merged[[
        "date",
        "distance_km",
        "pace_min_per_km",
        "Elevation Gain",
        "Average Heart Rate",
        "Max Heart Rate.1",
        "trimp",
        "CTL",
        "ATL",
    ]].rename(columns={
        "Elevation Gain":    "elevation_gain_m",
        "Average Heart Rate": "avg_hr",
        "Max Heart Rate.1":  "max_hr",
    })

    # Drop rows missing the HR targets (GP needs labels)
    before = len(out)
    out = out.dropna(subset=["avg_hr", "max_hr"])
    print(f"Dropped {before - len(out)} rows with missing HR  →  {len(out)} rows remain")

    # Sanity-check: remove likely sensor errors
    out = out[out["max_hr"] < HR_MAX_CUTOFF]

    # Drop rows where pace is infinite / NaN (zero-distance entries)
    out = out[out["pace_min_per_km"].notna() & np.isfinite(out["pace_min_per_km"])]

    print(f"Final dataset: {len(out)} rows")
    return out.reset_index(drop=True)


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":
    runs  = load_runs(CSV_PATH)
    daily = compute_ctl_atl(runs)

    # Attach trimp back onto runs before building features
    runs["trimp"] = compute_trimp(runs)

    gp_df = build_feature_df(runs, daily)

    print("\nSample output:")
    print(gp_df.head(10).to_string())
    print("\nColumn stats:")
    print(gp_df.describe().to_string())

    gp_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to '{OUT_PATH}'")

    # -------------------------------------------------------------------------
    #  Quick sanity plot — CTL and ATL over time
    # -------------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["CTL"], label="CTL (fitness, τ=42d)", linewidth=1.5)
    ax.plot(daily["date"], daily["ATL"], label="ATL (fatigue, τ=7d)",  linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Training Load (TRIMP units)")
    ax.set_title("CTL & ATL over time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("ctl_atl_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved CTL/ATL plot to 'ctl_atl_history.png'")
