"""
BelFund Donor EV Scorer — Feature Engineering
===============================================
Transforms raw donor transaction tables into a flat feature matrix
ready for model training. One row per donor per campaign (Action_Group).

Feature groups produced:
  1. Global all-history RFM aggregates        (prefix: all_)
  2. Five T-anchored fidelization windows     (prefix: w_0_6m_, w_6_12m_, etc.)
  3. Recent 12-month window                   (prefix: recent_)
  4. Derived momentum / trend features        (no prefix)
  5. Selection history features               (prefix: all_, windowed)

Column mapping (matches real pipeline):
  cid   = Ind_id           donor identifier
  gdate = Pdate            payment date
  gpost = DateGiftCreated  gift creation date
  gamt  = Amount           gift amount (€)
  sdate = SelectionDate    campaign selection date
  scost = Cost_unit        per-piece mailing cost (€)

Usage:
  python src/preprocess.py

  Or import:
    from src.preprocess import build_features
    features_df = build_features(gifts_df, selections_df, scope_df)
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config from environment ────────────────────────────────────────────────
ACTIVE_THRESHOLD_DAYS = int(os.getenv("ACTIVE_THRESHOLD_DAYS", "730"))
LAPSED_18M_DAYS       = int(os.getenv("LAPSED_18M_DAYS", "548"))
FORECAST_YRS          = 3
DISCOUNT_RATE         = 0.10
DATA_DIR              = Path(os.getenv("DATA_DIR",   "data/simulated"))
OUTPUT_DIR            = Path(os.getenv("OUTPUT_DIR", "data/features"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Column name constants ─────────────────────────────────────────────────
CID   = "Ind_id"
GDATE = "Pdate"
GPOST = "DateGiftCreated"
GAMT  = "Amount"
SDATE = "SelectionDate"
SCOST = "Cost_unit"
AG    = "Action_Group"


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _safe_div(num, den, fill=0.0):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(fill)


def _ensure(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _prefix(df, cid_col, pfx):
    if not pfx:
        return df
    return df.rename(columns={c: f"{pfx}{c}" for c in df.columns if c != cid_col})


@dataclass(frozen=True)
class Window:
    label: str
    start: pd.Timestamp
    end:   pd.Timestamp


def build_fid_windows(T: pd.Timestamp):
    """
    Five T-anchored fidelization windows, recency-first.

    w_0_6m     : T−6m → T        current engagement pulse
    w_6_12m    : T−12m → T−6m    recent-year baseline
    w_12_24m   : T−24m → T−12m   FID qualifying window (≥1 gift = eligible)
    w_24_36m   : T−36m → T−24m   medium-term reference
    w_36m_plus : history → T−36m  long-term legacy
    """
    mo = lambda n: T + pd.DateOffset(months=n)
    return [
        Window("w_0_6m",     mo(-6),                     T),
        Window("w_6_12m",    mo(-12),                    mo(-6)),
        Window("w_12_24m",   mo(-24),                    mo(-12)),
        Window("w_24_36m",   mo(-36),                    mo(-24)),
        Window("w_36m_plus", pd.Timestamp("1900-01-01"), mo(-36)),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# GIFT AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════

def agg_gifts(
    gifts: pd.DataFrame,
    cid: str, gdate: str, gamt: str, T: pd.Timestamp,
    start=None, end=None,
    prefix: str = "",
    include_global: bool = False,
) -> pd.DataFrame:
    """
    Aggregate gift history into donor-level features for one time window.

    Parameters
    ----------
    gifts         : pre-parsed gifts DataFrame (dates as datetime64, amounts as float)
    cid           : donor ID column name
    gdate         : payment date column name
    gamt          : amount column name
    T             : reference date (campaign selection date)
    start / end   : window boundaries (None = open-ended)
    prefix        : column prefix for output features
    include_global: compute additional global stats (recency, CLTV proxy etc.)
    """
    mask = gifts[gdate] < T
    if start is not None:
        mask = mask & (gifts[gdate] >= start)
    if end is not None:
        mask = mask & (gifts[gdate] < end)

    keep = [cid, gdate, gamt] + [c for c in ["Flag_SDD", "OP"] if c in gifts.columns]
    df = gifts.loc[mask, keep]

    if df.empty:
        return pd.DataFrame({cid: pd.Series(dtype=gifts[cid].dtype)})

    df = df.sort_values([cid, gdate])
    df = df.assign(_gap=df.groupby(cid, sort=False)[gdate].diff().dt.days)

    out = df.groupby(cid, sort=False).agg(
        gifts_n      = (gdate,  "count"),
        gifts_sum    = (gamt,   "sum"),
        gifts_mean   = (gamt,   "mean"),
        gifts_median = (gamt,   "median"),
        gifts_min    = (gamt,   "min"),
        gifts_max    = (gamt,   "max"),
        gifts_std    = (gamt,   "std"),
        gap_mean     = ("_gap", "mean"),
        gap_std      = ("_gap", "std"),
        first_dt     = (gdate,  "min"),
        last_dt      = (gdate,  "max"),
    ).reset_index()

    out["gifts_std"] = out["gifts_std"].fillna(0.0)
    out["gap_mean"]  = out["gap_mean"].fillna(0.0)
    out["gap_std"]   = out["gap_std"].fillna(0.0)

    out["gifts_cv"]       = _safe_div(out["gifts_std"],  out["gifts_mean"])
    out["gap_cv"]         = _safe_div(out["gap_std"],    out["gap_mean"].replace(0, np.nan))
    out["gifts_sum_log"]  = np.log1p(out["gifts_sum"].clip(0))
    out["gifts_mean_log"] = np.log1p(out["gifts_mean"].clip(0))
    out["gifts_max_log"]  = np.log1p(out["gifts_max"].clip(0))

    # Skew — vectorised moment formula
    pos = df[df[gamt] > 0].copy()
    if not pos.empty and len(pos) >= 3:
        m2 = pos.groupby(cid, sort=False)[gamt].var(ddof=0)
        m3 = pos.groupby(cid, sort=False)[gamt].apply(
            lambda s: ((s - s.mean()) ** 3).mean()
        )
        skew_vals = (m3 / (m2 ** 1.5).replace(0, np.nan)).fillna(0.0)
        out["gifts_skew"] = out[cid].map(skew_vals).fillna(0.0)
    else:
        out["gifts_skew"] = 0.0

    if start is not None and end is not None:
        wdays = (end - start).days
        out["last_gift_pos"] = (
            (out["last_dt"] - start).dt.days / max(wdays, 1)
        ).clip(0, 1)
    else:
        out["last_gift_pos"] = 0.0

    # SDD flag
    if "Flag_SDD" in df.columns:
        sdd = (
            df.assign(_sdd=(df["Flag_SDD"].fillna("N") == "Y").astype(int))
            .groupby(cid, sort=False)
            .agg(sdd_n=("_sdd", "sum"))
            .reset_index()
        )
        out = out.merge(sdd, on=cid, how="left")
        out["sdd_share"] = _safe_div(out["sdd_n"], out["gifts_n"])
    else:
        out["sdd_n"] = out["sdd_share"] = 0.0

    # Standing order flag
    if "OP" in df.columns:
        op = (
            df.assign(_op=(df["OP"].fillna("N") == "Y").astype(int))
            .groupby(cid, sort=False)
            .agg(op_n=("_op", "sum"))
            .reset_index()
        )
        out = out.merge(op, on=cid, how="left")
        out["op_share"] = _safe_div(out["op_n"], out["gifts_n"])
    else:
        out["op_n"] = out["op_share"] = 0.0

    if include_global:
        out["days_since_last"]   = (T - out["last_dt"]).dt.days
        out["days_since_first"]  = (T - out["first_dt"]).dt.days
        out["active_span_days"]  = (out["last_dt"] - out["first_dt"]).dt.days.clip(0)
        out["relationship_days"] = np.maximum(out["days_since_first"], 1)
        out["relationship_yrs"]  = out["relationship_days"] / 365.25
        out["active_span_yrs"]   = out["active_span_days"]  / 365.25
        out["months_since_last"] = (out["days_since_last"] / 30.44).round(1)

        out["recency_score"] = (
            1.0 - out["days_since_last"] / ACTIVE_THRESHOLD_DAYS
        ).clip(0, 1)

        bins   = [-1, 90, 180, 365, 545, ACTIVE_THRESHOLD_DAYS + 1]
        labels = [0, 1, 2, 3, 4]
        out["recency_band"] = pd.cut(
            out["days_since_last"], bins=bins, labels=labels, right=True
        ).astype(float).fillna(4.0)

        out["is_fid_eligible"] = (
            out["days_since_last"] < ACTIVE_THRESHOLD_DAYS
        ).astype(int)

        out["freq_per_yr_active"] = _safe_div(
            out["gifts_n"], out["active_span_yrs"].replace(0, np.nan)
        )
        out["freq_per_yr_full"] = _safe_div(
            out["gifts_n"], out["relationship_yrs"]
        )
        out["amt_per_yr_active"] = _safe_div(
            out["gifts_sum"], out["active_span_yrs"].replace(0, np.nan)
        )
        out["giving_consistency"] = np.maximum(1 - out["gifts_cv"], 0)
        out["gap_regularity"]     = np.maximum(1 - out["gap_cv"],   0)

        # Naive CLTV proxy — replaced by Bayesian step if enabled
        discount_pv            = sum(
            1 / (1 + DISCOUNT_RATE) ** t for t in range(1, FORECAST_YRS + 1)
        )
        out["cltv_proxy"]      = out["gifts_mean"] * out["freq_per_yr_active"] * FORECAST_YRS
        out["cltv_discounted"] = out["gifts_mean"] * out["freq_per_yr_active"] * discount_pv
        out["annual_run_rate"] = out["gifts_mean"] * out["freq_per_yr_active"]

        # IQR and robust stats
        if not pos.empty:
            q25 = pos.groupby(cid, sort=False)[gamt].quantile(0.25)
            q75 = pos.groupby(cid, sort=False)[gamt].quantile(0.75)
            med = pos.groupby(cid, sort=False)[gamt].quantile(0.50)
            out["iqr_gift"]    = out[cid].map(q75 - q25).fillna(0.0)
            out["median_gift"] = out[cid].map(med).fillna(0.0)
        else:
            out["iqr_gift"] = out["median_gift"] = 0.0

        out["robust_cv"]      = _safe_div(out["iqr_gift"], out["median_gift"].replace(0, np.nan))
        out["mean_med_ratio"] = _safe_div(out["gifts_mean"], out["median_gift"].replace(0, np.nan))

    out = out.drop(columns=["first_dt", "last_dt"], errors="ignore")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _prefix(out, cid, prefix)


# ═══════════════════════════════════════════════════════════════════════════
# SELECTION AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════

def agg_selections(
    sels: pd.DataFrame,
    cid: str, sdate: str, scost: str, T: pd.Timestamp,
    start=None, end=None,
    prefix: str = "",
    include_global: bool = False,
) -> pd.DataFrame:
    """Aggregate selection history into donor-level features for one window."""
    mask = sels[sdate] < T
    if start is not None:
        mask = mask & (sels[sdate] >= start)
    if end is not None:
        mask = mask & (sels[sdate] < end)

    df = sels.loc[mask, [cid, sdate, scost]]

    if df.empty:
        return pd.DataFrame({cid: pd.Series(dtype=sels[cid].dtype)})

    out = df.groupby(cid, sort=False).agg(
        sel_n         = (sdate, "count"),
        sel_cost_sum  = (scost, "sum"),
        sel_cost_mean = (scost, "mean"),
        first_sel     = (sdate, "min"),
        last_sel      = (sdate, "max"),
    ).reset_index()

    out["sel_cost_log"] = np.log1p(out["sel_cost_sum"].clip(0))

    if include_global:
        out["days_since_last_sel"] = (T - out["last_sel"]).dt.days
        out["sel_span_days"]       = (out["last_sel"] - out["first_sel"]).dt.days.clip(0)

    out = out.drop(columns=["first_sel", "last_sel"], errors="ignore")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _prefix(out, cid, prefix)


# ═══════════════════════════════════════════════════════════════════════════
# RECENT WINDOW AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════

def agg_recent_window(
    gifts: pd.DataFrame,
    cid: str, gdate: str, gamt: str,
    T: pd.Timestamp, recent_months: int = 12,
) -> pd.DataFrame:
    """Aggregate gift behaviour in the N months immediately before T."""
    if gifts.empty:
        return pd.DataFrame({cid: pd.Series(dtype=gifts[cid].dtype)})

    # Last gift from full pre-T history
    gifts_sorted = gifts.sort_values([cid, gdate])
    last_gifts = (
        gifts_sorted.groupby(cid, sort=False)[[gdate, gamt]]
        .last()
        .rename(columns={gdate: "_last_gift_dt", gamt: "last_gift_amt"})
        .reset_index()
    )

    window_start = T - pd.DateOffset(months=recent_months)
    df = gifts_sorted.loc[gifts_sorted[gdate] >= window_start]

    if df.empty:
        return pd.DataFrame({cid: pd.Series(dtype=gifts[cid].dtype)})

    df = df.assign(_gap=df.groupby(cid, sort=False)[gdate].diff().dt.days)

    out = df.groupby(cid, sort=False).agg(
        recent_gifts_n    = (gdate,  "count"),
        recent_gifts_sum  = (gamt,   "sum"),
        recent_gifts_mean = (gamt,   "mean"),
        recent_gifts_max  = (gamt,   "max"),
        recent_gifts_std  = (gamt,   "std"),
        recent_gap_mean   = ("_gap", "mean"),
    ).reset_index()

    out["recent_gifts_std"] = out["recent_gifts_std"].fillna(0.0)
    out["recent_gap_mean"]  = out["recent_gap_mean"].fillna(0.0)
    out["recent_gifts_cv"]  = _safe_div(out["recent_gifts_std"], out["recent_gifts_mean"])
    out["recent_sum_log"]   = np.log1p(out["recent_gifts_sum"].clip(0))
    out["recent_mean_log"]  = np.log1p(out["recent_gifts_mean"].clip(0))

    if "Flag_SDD" in df.columns:
        sdd_agg = (
            df.assign(_sdd=(df["Flag_SDD"].fillna("N") == "Y").astype(int))
            .groupby(cid, sort=False)
            .agg(recent_sdd_n=("_sdd", "sum"))
            .reset_index()
        )
        out = out.merge(sdd_agg, on=cid, how="left")
        out["recent_sdd_share"] = _safe_div(out["recent_sdd_n"], out["recent_gifts_n"])
    else:
        out["recent_sdd_n"] = out["recent_sdd_share"] = 0.0

    out = out.merge(last_gifts[[cid, "last_gift_amt"]], on=cid, how="left")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FEATURE BUILDER — one row per donor per campaign date T
# ═══════════════════════════════════════════════════════════════════════════

def build_features_for_T(
    gifts: pd.DataFrame,
    sels: pd.DataFrame,
    T: pd.Timestamp,
    ag_label: str = "?",
) -> pd.DataFrame:
    """
    Build one feature row per donor for a single campaign reference date T.

    Steps
    -----
    1. Global all-history features
    2. Five T-anchored fidelization windows
    3. Recent 12-month window
    4. Derived momentum / trend / engagement features

    Parameters
    ----------
    gifts     : gift history (parsed dates and amounts)
    sels      : selection history (parsed dates and costs)
    T         : campaign selection date
    ag_label  : action group label for logging

    Returns
    -------
    DataFrame with one row per donor, all numeric features
    """
    # ── Parse dates and amounts once ─────────────────────────────────────────
    gifts = gifts.copy()
    gifts[GDATE] = pd.to_datetime(gifts[GDATE], errors="coerce")
    gifts[GAMT]  = pd.to_numeric(gifts[GAMT],  errors="coerce")
    if GPOST in gifts.columns:
        gifts[GPOST] = pd.to_datetime(gifts[GPOST], errors="coerce")

    sels = sels.copy()
    sels[SDATE] = pd.to_datetime(sels[SDATE], errors="coerce")
    sels[SCOST] = pd.to_numeric(sels[SCOST],  errors="coerce").fillna(0.0)

    # ── 1. Global all-history ─────────────────────────────────────────────────
    g_all = agg_gifts(gifts, CID, GDATE, GAMT, T=T, end=T,
                      prefix="all_", include_global=True)
    s_all = agg_selections(sels, CID, SDATE, SCOST, T=T, end=T,
                           prefix="all_", include_global=True)
    feat = g_all.merge(s_all, on=CID, how="outer")

    feat = _ensure(feat, [
        "all_gifts_n", "all_gifts_sum", "all_gifts_median", "all_gifts_mean",
        "all_gifts_max", "all_gifts_std", "all_gifts_cv",
        "all_sel_n", "all_sel_cost_sum",
        "all_days_since_last", "all_months_since_last",
        "all_recency_score", "all_recency_band",
        "all_gap_mean", "all_gap_cv",
        "all_freq_per_yr_active", "all_freq_per_yr_full",
        "all_amt_per_yr_active",
        "all_cltv_proxy", "all_cltv_discounted", "all_annual_run_rate",
        "all_giving_consistency", "all_gap_regularity",
        "all_sdd_share", "all_sdd_n",
        "all_relationship_yrs", "all_active_span_yrs",
        "all_median_gift", "all_iqr_gift", "all_robust_cv",
    ])

    feat["all_resp_rate"]     = _safe_div(feat["all_gifts_n"],    feat["all_sel_n"].replace(0, np.nan))
    feat["all_net_amount"]    = feat["all_gifts_sum"].fillna(0)   - feat["all_sel_cost_sum"].fillna(0)
    feat["all_roi_like"]      = _safe_div(feat["all_gifts_sum"],  feat["all_sel_cost_sum"].replace(0, np.nan))
    feat["all_cost_per_gift"] = _safe_div(feat["all_sel_cost_sum"], feat["all_gifts_n"].replace(0, np.nan))

    # ── 2. Five T-anchored windows — collect then single merge ────────────────
    window_frames = []
    for w in build_fid_windows(T):
        gw = agg_gifts(gifts, CID, GDATE, GAMT, T=T,
                       start=w.start, end=w.end, prefix=f"{w.label}_")
        sw = agg_selections(sels, CID, SDATE, SCOST, T=T,
                            start=w.start, end=w.end, prefix=f"{w.label}_")
        m = gw.merge(sw, on=CID, how="outer")
        m = _ensure(m, [
            f"{w.label}_gifts_n",    f"{w.label}_gifts_sum",
            f"{w.label}_gifts_mean", f"{w.label}_gifts_max",
            f"{w.label}_sel_n",      f"{w.label}_sel_cost_sum",
        ])
        m[f"{w.label}_resp_rate"]     = _safe_div(m[f"{w.label}_gifts_n"],      m[f"{w.label}_sel_n"].replace(0, np.nan))
        m[f"{w.label}_net_amount"]    = m[f"{w.label}_gifts_sum"].fillna(0)     - m[f"{w.label}_sel_cost_sum"].fillna(0)
        m[f"{w.label}_roi_like"]      = _safe_div(m[f"{w.label}_gifts_sum"],    m[f"{w.label}_sel_cost_sum"].replace(0, np.nan))
        m[f"{w.label}_cost_per_gift"] = _safe_div(m[f"{w.label}_sel_cost_sum"], m[f"{w.label}_gifts_n"].replace(0, np.nan))
        window_frames.append(m)

    all_window = window_frames[0]
    for wf in window_frames[1:]:
        all_window = all_window.merge(wf, on=CID, how="outer")
    feat = feat.merge(all_window, on=CID, how="left")

    # ── 3. Recent 12-month window ─────────────────────────────────────────────
    gifts_pre_T = gifts.loc[gifts[GDATE] < T]
    recent = agg_recent_window(gifts_pre_T, CID, GDATE, GAMT, T=T, recent_months=12)
    feat   = feat.merge(recent, on=CID, how="left")

    feat = _ensure(feat, [
        "recent_gifts_n", "recent_gifts_sum", "recent_gifts_mean",
        "recent_gifts_max", "recent_gifts_cv",
        "recent_sdd_n", "recent_sdd_share",
        "last_gift_amt",
    ])

    # ── 4. Derived momentum / trend / engagement features ────────────────────
    feat = _ensure(feat, [
        "w_0_6m_gifts_n",      "w_6_12m_gifts_n",      "w_12_24m_gifts_n",
        "w_0_6m_gifts_sum",    "w_6_12m_gifts_sum",     "w_12_24m_gifts_sum",
        "w_0_6m_gifts_mean",   "w_6_12m_gifts_mean",    "w_12_24m_gifts_mean",
        "w_0_6m_gifts_max",    "w_6_12m_gifts_max",
        "w_0_6m_resp_rate",    "w_6_12m_resp_rate",     "w_12_24m_resp_rate",
        "w_0_6m_roi_like",     "w_6_12m_roi_like",
        "w_0_6m_sel_n",        "w_6_12m_sel_n",         "w_12_24m_sel_n",
        "w_0_6m_sel_cost_sum", "w_6_12m_sel_cost_sum",
        "w_24_36m_gifts_n",    "w_24_36m_gifts_mean",   "w_24_36m_resp_rate",
        "all_gifts_median",    "all_resp_rate",
        "all_days_since_last", "all_gap_mean",
        "all_gifts_mean",      "all_gifts_max",
        "all_recency_score",   "all_giving_consistency",
        "all_cltv_discounted", "all_sdd_share", "recent_sdd_share",
        "all_freq_per_yr_active",
    ])

    # Frequency momentum
    feat["momentum_n_0_6_vs_6_12"]   = feat["w_0_6m_gifts_n"]  - feat["w_6_12m_gifts_n"]
    feat["momentum_n_6_12_vs_12_24"] = feat["w_6_12m_gifts_n"] - feat["w_12_24m_gifts_n"]

    # Amount momentum
    feat["momentum_amt_0_6_vs_6_12"]   = feat["w_0_6m_gifts_mean"]  - feat["w_6_12m_gifts_mean"]
    feat["momentum_amt_6_12_vs_12_24"] = feat["w_6_12m_gifts_mean"] - feat["w_12_24m_gifts_mean"]

    # Composite acceleration: −2 (decelerating) to +2 (accelerating)
    feat["giving_acceleration"] = (
        np.sign(feat["momentum_n_0_6_vs_6_12"]) +
        np.sign(feat["momentum_amt_0_6_vs_6_12"])
    )

    # Upgrade / downgrade signals
    base_med = pd.to_numeric(feat["all_gifts_median"], errors="coerce").replace(0, np.nan)
    feat["last_gift_vs_median"]   = _safe_div(feat["last_gift_amt"],     base_med)
    feat["last_gift_vs_allmax"]   = _safe_div(feat["last_gift_amt"],     feat["all_gifts_max"].replace(0, np.nan))
    feat["recent_mean_vs_allmed"] = _safe_div(feat["recent_gifts_mean"], base_med)
    feat["recent_max_vs_allmax"]  = _safe_div(feat["recent_gifts_max"],  feat["all_gifts_max"].replace(0, np.nan))

    feat["flag_step_up"]       = (feat["last_gift_vs_median"] > 1.25).astype(int)
    feat["flag_step_down"]     = (feat["last_gift_vs_median"] < 0.75).astype(int)
    feat["flag_active_0_6m"]   = (feat["w_0_6m_gifts_n"]   > 0).astype(int)
    feat["flag_active_6_12m"]  = (feat["w_6_12m_gifts_n"]  > 0).astype(int)
    feat["flag_active_12_24m"] = (feat["w_12_24m_gifts_n"] > 0).astype(int)

    # Response rate trend
    feat["rr_trend_0_6_vs_6_12"]   = feat["w_0_6m_resp_rate"]  - feat["w_6_12m_resp_rate"]
    feat["rr_trend_6_12_vs_12_24"] = feat["w_6_12m_resp_rate"] - feat["w_12_24m_resp_rate"]
    feat["rr_trend_vs_alltime"]    = feat["w_0_6m_resp_rate"]  - feat["all_resp_rate"]

    # Recent productivity vs history
    feat["recent_window_productivity"] = _safe_div(
        feat["w_0_6m_gifts_mean"], feat["all_gifts_mean"].replace(0, np.nan))
    feat["recent_vs_24_36_mean"] = _safe_div(
        feat["w_0_6m_gifts_mean"], feat["w_24_36m_gifts_mean"].replace(0, np.nan))

    # SDD stability
    feat["sdd_stability"] = _safe_div(
        feat["recent_sdd_share"], feat["all_sdd_share"].replace(0, np.nan)
    ).clip(0, 2)

    # Engagement depth: recency × consistency × recent_productivity
    feat["engagement_depth"] = (
        feat["all_recency_score"].fillna(0) *
        feat["all_giving_consistency"].fillna(0) *
        feat["recent_window_productivity"].clip(0, 1).fillna(0)
    )

    # Lapse risk: high when both recent productivity and recency are low
    feat["lapse_risk_score"] = (
        (1 - feat["recent_window_productivity"].clip(0, 1).fillna(0)) *
        (1 - feat["all_recency_score"].fillna(0))
    )

    # ROI trend
    feat["roi_trend_0_6_vs_6_12"] = feat["w_0_6m_roi_like"] - feat["w_6_12m_roi_like"]

    # Seasonal half-year ratio
    feat["seasonal_h1_h2_ratio"] = _safe_div(
        feat["w_0_6m_gifts_sum"], feat["w_6_12m_gifts_sum"].replace(0, np.nan)
    ).clip(0, 5)

    # Naive recency-adjusted CLTV
    feat["recency_adj_cltv"] = (
        feat["all_cltv_discounted"].fillna(0) * feat["all_recency_score"].fillna(0)
    )

    # Attach reference date
    feat["SelectionDate"] = T
    feat["Action_Group"]  = ag_label

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — run across all in-scope campaigns
# ═══════════════════════════════════════════════════════════════════════════

def build_features(
    gifts_df: pd.DataFrame,
    selections_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    filter_fid: bool = True,
) -> pd.DataFrame:
    """
    Build feature rows for every campaign in scope_df.

    Parameters
    ----------
    gifts_df      : full gift history
    selections_df : full selection history
    scope_df      : in-scope campaigns — must have Action_Group and SelectionDate
    filter_fid    : drop donors with days_since_last >= ACTIVE_THRESHOLD_DAYS

    Returns
    -------
    Combined feature DataFrame, one row per donor × campaign
    """
    ag_col = "Action_Group" if "Action_Group" in scope_df.columns else "Action_group"
    action_groups = scope_df[ag_col].unique()
    print(f"Building features for {len(action_groups)} campaigns...")

    all_frames = []

    for ag in action_groups:
        # Derive T = latest SelectionDate in this campaign
        ag_sels = selections_df[selections_df[ag_col] == ag]
        if ag_sels.empty:
            print(f"  [AG {ag}] No selections — skipping")
            continue

        T = pd.to_datetime(ag_sels["SelectionDate"]).max()
        if pd.isna(T):
            print(f"  [AG {ag}] No valid SelectionDate — skipping")
            continue

        # Donor pool for this campaign
        donor_ids = ag_sels[CID].unique()
        gifts_ag  = gifts_df[gifts_df[CID].isin(donor_ids)]
        sels_ag   = selections_df[selections_df[CID].isin(donor_ids)]

        feat = build_features_for_T(gifts_ag, sels_ag, T=T, ag_label=ag)

        # Filter to FID-eligible donors only
        if filter_fid and "all_days_since_last" in feat.columns:
            before = len(feat)
            feat = feat[feat["all_days_since_last"] < ACTIVE_THRESHOLD_DAYS].copy()
            dropped = before - len(feat)
            if dropped:
                print(f"  [AG {ag}] T={T.date()} | {len(feat):,} donors "
                      f"({dropped:,} lapsed dropped)")
        else:
            print(f"  [AG {ag}] T={T.date()} | {len(feat):,} donors")

        all_frames.append(feat)
        gc.collect()

    if not all_frames:
        raise ValueError("No feature rows produced. Check scope_df and selections_df.")

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    print(f"\nFeature matrix: {combined.shape[0]:,} rows × {combined.shape[1]:,} columns")
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("BelFund Donor EV Scorer — Feature Engineering")
    print("=" * 65)

    # ── Load simulated data ────────────────────────────────────────────────
    print(f"\nLoading data from: {DATA_DIR}")

    gifts_df = pd.read_csv(DATA_DIR / "gifts_valid_sim.csv", low_memory=False)
    sels_df  = pd.read_csv(DATA_DIR / "tab_sel_sim.csv",     low_memory=False)
    scope_df = pd.read_excel(DATA_DIR / "PELICANO_FID_complete_sim.xlsx")

    # Parse dates
    gifts_df["Pdate"]           = pd.to_datetime(gifts_df["Pdate"],           errors="coerce")
    gifts_df["DateGiftCreated"] = pd.to_datetime(gifts_df["DateGiftCreated"], errors="coerce")
    sels_df["SelectionDate"]    = pd.to_datetime(
        sels_df["Date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    sels_df.rename(columns={"Action_Group": "Action_Group"}, inplace=True)
    scope_df["SelectionDate"]   = pd.to_datetime(scope_df["Post_Date"], errors="coerce")

    print(f"  gifts_df:  {gifts_df.shape}")
    print(f"  sels_df:   {sels_df.shape}")
    print(f"  scope_df:  {scope_df.shape}")
    print(f"  Active threshold: {ACTIVE_THRESHOLD_DAYS} days")

    # ── Add cost to selections ─────────────────────────────────────────────
    camps_df = pd.read_csv(DATA_DIR / "actions_groups_sim.csv")
    acts_df  = pd.read_csv(DATA_DIR / "actions_sim.csv")
    acts_df  = acts_df.merge(camps_df[["Action_Group", "Cost_unit"]], on="Action_Group", how="left")
    sels_df  = sels_df.merge(acts_df[["Action_id", "Cost_unit"]], on="Action_id", how="left")
    sels_df["Cost_unit"] = sels_df["Cost_unit"].fillna(sels_df["Cost_unit"].mean())

    # Rename to match pipeline column names
    sels_df.rename(columns={"Action_Group": "Action_Group"}, inplace=True)

    # ── Build features ─────────────────────────────────────────────────────
    features_df = build_features(
        gifts_df      = gifts_df,
        selections_df = sels_df,
        scope_df      = scope_df,
        filter_fid    = True,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Shape: {features_df.shape}")
    print(f"\nSample feature stats:")
    key_cols = [
        "all_gifts_n", "all_gifts_mean", "all_days_since_last",
        "all_recency_score", "all_resp_rate", "recency_adj_cltv",
        "engagement_depth", "lapse_risk_score",
    ]
    available = [c for c in key_cols if c in features_df.columns]
    print(features_df[available].describe(percentiles=[.25, .50, .75]).round(3).to_string())


if __name__ == "__main__":
    main()
