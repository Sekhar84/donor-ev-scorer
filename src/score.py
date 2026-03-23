"""
BelFund Donor EV Scorer — EV Scoring and Selection
=====================================================
Loads trained models and produces EV scores + selection decisions
for a new campaign or for evaluating the test set.

EV formula:
  prob_donate   = calibrated_classifier.predict_proba()
  pred_amount   = exp(regressor.predict()) — log-space inverse
  ev_eur        = prob_donate × pred_amount
  net_ev_eur    = ev_eur − cost_unit
  selected      = ev_eur > cost_unit          (FID campaigns)

  For REAC campaigns (set CAMPAIGN_TYPE=REAC in .env):
  ev_decile     = qcut(ev_eur, q=10)
  selected      = ev_decile >= (10 - EV_DECILE_CUTOFF)

Outputs:
  data/scores/scored_<date>.parquet   — full scored dataset
  data/scores/selected_<date>.parquet — selected donors only
  data/scores/summary_<date>.json     — campaign-level summary

Usage:
  # Score the full feature dataset (train+val+test)
  python src/score.py

  # Score a specific split only
  python src/score.py --split test

  # Import and use in FastAPI
  from src.score import load_models, score_donors
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
FEATURES_PATH  = Path(os.getenv("OUTPUT_DIR", "data/features")) / "features.parquet"
MODELS_DIR     = Path("models")
SCORES_DIR     = Path("data/scores")
SCORES_DIR.mkdir(parents=True, exist_ok=True)

CAMPAIGN_TYPE    = os.getenv("CAMPAIGN_TYPE",    "FID")
COST_PER_MAIL    = float(os.getenv("COST_PER_MAIL", "1.50"))
EV_DECILE_CUTOFF = int(os.getenv("EV_DECILE_CUTOFF", "3"))

NON_FEATURE_COLS = {
    "Ind_id", "Action_Group", "SelectionDate",
    "Gave_Donation", "Amount",
}


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_models():
    """
    Load all trained model artefacts from models/.

    Returns dict with keys:
      clf          — XGBoost propensity classifier
      calibrator   — isotonic or Platt calibrator
      cal_type     — calibrator type string
      reg          — XGBoost amount regressor
      imp_p        — propensity imputer
      imp_a        — amount imputer (same donors, same features)
      amount_cap   — p90 cap fitted on train responders
      feature_cols — list of 225 feature column names
    """
    required = [
        "propensity_model.joblib",
        "propensity_calibrator.joblib",
        "amount_model.joblib",
        "imputer_p.joblib",
        "feature_cols.joblib",
        "amount_cap.joblib",
    ]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artefacts: {missing}\n"
            f"Run python src/train.py first."
        )

    cal_type = "Isotonic"
    if (MODELS_DIR / "calibrator_type.joblib").exists():
        cal_type = joblib.load(MODELS_DIR / "calibrator_type.joblib")

    # Load exact feature names model was trained on (may differ from feature_cols
    # if SimpleImputer dropped all-NaN columns during training)
    if (MODELS_DIR / "model_feature_names_p.joblib").exists():
        model_feat_p = joblib.load(MODELS_DIR / "model_feature_names_p.joblib")
        model_feat_a = joblib.load(MODELS_DIR / "model_feature_names_a.joblib")
    else:
        model_feat_p = model_feat_a = None

    return {
        "clf":             joblib.load(MODELS_DIR / "propensity_model.joblib"),
        "calibrator":      joblib.load(MODELS_DIR / "propensity_calibrator.joblib"),
        "cal_type":        cal_type,
        "reg":             joblib.load(MODELS_DIR / "amount_model.joblib"),
        "imp_p":           joblib.load(MODELS_DIR / "imputer_p.joblib"),
        "amount_cap":      joblib.load(MODELS_DIR / "amount_cap.joblib"),
        "feature_cols":    joblib.load(MODELS_DIR / "feature_cols.joblib"),
        "model_feat_p":    model_feat_p,
        "model_feat_a":    model_feat_a,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CORE SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def score_donors(
    df: pd.DataFrame,
    models: dict,
    cost_col: str = "all_sel_cost_mean",
    campaign_type: str = None,
    ev_decile_cutoff: int = None,
) -> pd.DataFrame:
    """
    Score donors and apply selection logic.

    Parameters
    ----------
    df               : feature DataFrame — must contain feature_cols columns
    models           : dict from load_models()
    cost_col         : column name for per-donor campaign cost
                       falls back to COST_PER_MAIL env var if not found
    campaign_type    : "FID" or "REAC" — overrides CAMPAIGN_TYPE env var
    ev_decile_cutoff : top N deciles to select (REAC only)

    Returns
    -------
    DataFrame with original columns plus:
      prob_donate_raw   — raw XGBoost probability
      prob_donate       — calibrated probability
      pred_amount_eur   — predicted gift amount (€, capped)
      ev_eur            — expected value (€)
      cost_unit         — mailing cost used for selection
      net_ev_eur        — ev_eur - cost_unit
      roi               — ev_eur / cost_unit
      ev_decile         — EV decile (10=highest, 1=lowest)
      selected          — True/False selection decision
    """
    camp_type   = campaign_type    or CAMPAIGN_TYPE
    dec_cutoff  = ev_decile_cutoff or EV_DECILE_CUTOFF
    feature_cols = models["feature_cols"]

    # ── Validate features ──────────────────────────────────────────────────
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        # Fill missing with 0 — consistent with training imputation
        for c in missing_cols:
            df[c] = 0.0

    X = df[feature_cols].copy()

    # ── Impute ────────────────────────────────────────────────────────────
    X_arr = models["imp_p"].transform(X)

    # Use exact feature names the model was trained with
    # This handles the case where SimpleImputer dropped all-NaN columns
    if models.get("model_feat_p") is not None:
        feat_names_p = models["model_feat_p"]
        feat_names_a = models["model_feat_a"]
    else:
        # Fallback — detect surviving columns
        n_out = X_arr.shape[1]
        if n_out == len(feature_cols):
            feat_names_p = feat_names_a = feature_cols
        else:
            feat_names_p = feat_names_a = [
                feature_cols[i] for i in range(len(feature_cols))
                if not np.all(np.isnan(X.values[:, i]))
            ]
            if len(feat_names_p) != n_out:
                feat_names_p = feat_names_a = [f"f{i}" for i in range(n_out)]

    X_df = pd.DataFrame(X_arr, columns=feat_names_p, index=df.index)

    # ── Propensity — raw then calibrated ─────────────────────────────────
    prob_raw = models["clf"].predict_proba(X_df)[:, 1]

    cal = models["calibrator"]
    cal_type = models["cal_type"]
    if cal_type == "Isotonic":
        prob_cal = cal.predict(prob_raw)
    else:
        prob_cal = cal.predict_proba(prob_raw.reshape(-1, 1))[:, 1]

    # ── Amount prediction — log-space, uncap for scoring ─────────────────
    pred_amount = np.expm1(models["reg"].predict(X_df)).clip(0)

    # ── EV calculation ────────────────────────────────────────────────────
    ev = prob_cal * pred_amount

    # Cost per donor — use campaign-specific cost if available
    if cost_col in df.columns:
        cost = df[cost_col].values.copy()
        cost = np.where(cost <= 0, COST_PER_MAIL, cost)
    else:
        cost = np.full(len(df), COST_PER_MAIL)

    net_ev = ev - cost
    roi    = np.where(cost > 0, ev / cost, 0.0)

    # EV decile — 10 = highest EV donors (best to mail), 1 = lowest
    try:
        ev_rank = pd.qcut(ev, q=10, labels=False, duplicates="drop")
        # ev_rank: 0=lowest EV, 9=highest EV
        # Convert to 1-10 scale where 10=highest EV
        ev_decile = ev_rank + 1  # 1=lowest, 10=highest
    except Exception:
        ev_decile = np.ones(len(ev))

    # ── Selection logic ───────────────────────────────────────────────────
    if camp_type == "REAC":
        # REAC: select top N deciles — decile 10 = highest EV
        selected = pd.Series(ev_decile).ge(10 - dec_cutoff + 1).values
    else:
        # FID: select if EV exceeds cost per mail
        selected = net_ev > 0

    # ── Assemble output ───────────────────────────────────────────────────
    out = df.copy()
    out["prob_donate_raw"] = prob_raw.round(4)
    out["prob_donate"]     = prob_cal.round(4)
    out["pred_amount_eur"] = pred_amount.round(2)
    out["ev_eur"]          = ev.round(2)
    out["cost_unit"]       = cost.round(4)
    out["net_ev_eur"]      = net_ev.round(2)
    out["roi"]             = roi.round(3)
    out["ev_decile"]       = ev_decile
    out["selected"]        = selected

    return out


# ═══════════════════════════════════════════════════════════════════════════
# DECILE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def decile_analysis(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lift table: predicted EV vs actual EV by score decile.
    Requires Gave_Donation and Amount columns (evaluation only).
    """
    df = scored_df.copy()
    if "Gave_Donation" not in df.columns or "Amount" not in df.columns:
        return pd.DataFrame()

    df["actual_ev"] = df["Amount"] * df["Gave_Donation"]

    tbl = (
        df.groupby("ev_decile", observed=True)
        .agg(
            n               = ("ev_decile",    "count"),
            mean_prob       = ("prob_donate",  "mean"),
            mean_ev         = ("ev_eur",       "mean"),
            total_ev        = ("ev_eur",       "sum"),
            mean_actual_ev  = ("actual_ev",    "mean"),
            total_actual_ev = ("actual_ev",    "sum"),
            actual_rr       = ("Gave_Donation","mean"),
        )
        .reset_index()
        .sort_values("ev_decile", ascending=False)
    )

    tbl["lift"] = (tbl["mean_actual_ev"] / tbl["mean_actual_ev"].mean()).round(2)
    return tbl


# ═══════════════════════════════════════════════════════════════════════════
# CAMPAIGN SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def campaign_summary(scored_df: pd.DataFrame, cost_per_mail: float = None) -> dict:
    """
    Compute campaign-level selection summary.
    """
    n_total    = len(scored_df)
    n_selected = int(scored_df["selected"].sum())
    n_rejected = n_total - n_selected

    sel = scored_df[scored_df["selected"]]
    rej = scored_df[~scored_df["selected"]]

    summary = {
        "campaign_type":        CAMPAIGN_TYPE,
        "n_total":              n_total,
        "n_selected":           n_selected,
        "n_rejected":           n_rejected,
        "selection_rate":       round(n_selected / n_total, 4) if n_total else 0,
        "mean_ev_selected":     round(float(sel["ev_eur"].mean()),     2) if len(sel) else 0,
        "mean_ev_rejected":     round(float(rej["ev_eur"].mean()),     2) if len(rej) else 0,
        "mean_prob_selected":   round(float(sel["prob_donate"].mean()),4) if len(sel) else 0,
        "mean_prob_rejected":   round(float(rej["prob_donate"].mean()),4) if len(rej) else 0,
        "total_expected_revenue": round(float(sel["ev_eur"].sum()),    2) if len(sel) else 0,
        "total_mailing_cost":   round(float(sel["cost_unit"].sum()),   2) if len(sel) else 0,
        "expected_net_revenue": round(
            float(sel["ev_eur"].sum()) - float(sel["cost_unit"].sum()), 2
        ) if len(sel) else 0,
        "mean_roi_selected":    round(float(sel["roi"].mean()), 3) if len(sel) else 0,
        "ev_p10":  round(float(scored_df["ev_eur"].quantile(0.10)), 2),
        "ev_p50":  round(float(scored_df["ev_eur"].quantile(0.50)), 2),
        "ev_p90":  round(float(scored_df["ev_eur"].quantile(0.90)), 2),
    }

    # Add actual performance if targets available
    if "Gave_Donation" in scored_df.columns and "Amount" in scored_df.columns:
        summary["actual_rr"]             = round(float(scored_df["Gave_Donation"].mean()), 4)
        summary["actual_rr_selected"]    = round(float(sel["Gave_Donation"].mean()), 4) if len(sel) else 0
        summary["actual_revenue_selected"] = round(float(sel["Amount"].sum()), 2) if len(sel) else 0
        summary["actual_net_selected"]   = round(
            float(sel["Amount"].sum()) - float(sel["cost_unit"].sum()), 2
        ) if len(sel) else 0

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(split: str = "all"):
    print("=" * 65)
    print("BelFund Donor EV Scorer — EV Scoring")
    print("=" * 65)
    print(f"  Campaign type : {CAMPAIGN_TYPE}")
    print(f"  Cost per mail : €{COST_PER_MAIL:.2f} (default, overridden by cost_unit per donor)")
    if CAMPAIGN_TYPE == "REAC":
        print(f"  EV decile cut : top {EV_DECILE_CUTOFF} deciles")

    # ── Load models ────────────────────────────────────────────────────────
    print("\nLoading models...")
    models = load_models()
    print(f"  Propensity:  {models['clf'].__class__.__name__} "
          f"(calibrator: {models['cal_type']})")
    print(f"  Amount:      {models['reg'].__class__.__name__}")
    print(f"  Amount cap:  €{models['amount_cap']:.2f}")
    print(f"  Features:    {len(models['feature_cols'])}")

    # ── Load features ──────────────────────────────────────────────────────
    print(f"\nLoading features: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  Shape: {df.shape}")

    # Filter to requested split
    if split != "all" and "SelectionDate" in df.columns:
        df["SelectionDate"] = pd.to_datetime(df["SelectionDate"])
        campaign_dates = (
            df.groupby("Action_Group")["SelectionDate"].min().sort_values()
        )
        groups  = campaign_dates.index.tolist()
        n       = len(groups)
        n_test  = max(1, int(np.ceil(0.15 * n)))
        n_val   = max(1, int(np.ceil(0.15 * n)))
        n_train = n - n_val - n_test

        if split == "train":
            keep = set(groups[:n_train])
        elif split == "val":
            keep = set(groups[n_train : n_train + n_val])
        elif split == "test":
            keep = set(groups[n_train + n_val:])
        else:
            keep = set(groups)

        df = df[df["Action_Group"].isin(keep)].copy()
        print(f"  Split '{split}': {len(df):,} rows  ({len(keep)} campaigns)")

    # ── Score ──────────────────────────────────────────────────────────────
    print("\nScoring donors...")
    scored = score_donors(df, models, campaign_type=CAMPAIGN_TYPE)

    print(f"\nEV distribution:")
    print(f"  mean=€{scored['ev_eur'].mean():.2f}  "
          f"median=€{scored['ev_eur'].median():.2f}  "
          f"p90=€{scored['ev_eur'].quantile(0.90):.2f}")

    print(f"\nPropensity scores (calibrated):")
    print(f"  mean={scored['prob_donate'].mean():.4f}  "
          f"median={scored['prob_donate'].median():.4f}  "
          f"p90={scored['prob_donate'].quantile(0.90):.4f}")

    # ── Campaign summary ───────────────────────────────────────────────────
    summary = campaign_summary(scored)
    print(f"\nCampaign summary:")
    print(f"  Total donors:     {summary['n_total']:,}")
    print(f"  Selected:         {summary['n_selected']:,} "
          f"({summary['selection_rate']:.1%})")
    print(f"  Rejected:         {summary['n_rejected']:,}")
    print(f"  Mean EV selected: €{summary['mean_ev_selected']:.2f}")
    print(f"  Mean EV rejected: €{summary['mean_ev_rejected']:.2f}")
    print(f"  Total exp. revenue (selected): €{summary['total_expected_revenue']:,.2f}")
    print(f"  Total mailing cost:            €{summary['total_mailing_cost']:,.2f}")
    print(f"  Expected net revenue:          €{summary['expected_net_revenue']:,.2f}")
    print(f"  Mean ROI (selected):           {summary['mean_roi_selected']:.2f}x")

    if "actual_rr" in summary:
        print(f"\nActual performance (test set):")
        print(f"  Overall RR:         {summary['actual_rr']:.2%}")
        print(f"  RR selected donors: {summary['actual_rr_selected']:.2%}")
        print(f"  Actual revenue:     €{summary['actual_revenue_selected']:,.2f}")
        print(f"  Actual net revenue: €{summary['actual_net_selected']:,.2f}")

    # ── Decile analysis ────────────────────────────────────────────────────
    if "Gave_Donation" in scored.columns:
        print(f"\nDecile lift table (decile 10 = highest EV donors):")
        dtbl = decile_analysis(scored)
        if not dtbl.empty:
            print(f"  {'Decile':>8}  {'N':>7}  {'Mean EV':>9}  "
                  f"{'Act EV':>9}  {'Act RR':>8}  {'Lift':>6}")
            print(f"  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*6}")
            for _, row in dtbl.iterrows():
                print(f"  {int(row['ev_decile']):>8}  "
                      f"{int(row['n']):>7,}  "
                      f"€{row['mean_ev']:>8.2f}  "
                      f"€{row['mean_actual_ev']:>8.2f}  "
                      f"{row['actual_rr']:>8.2%}  "
                      f"{row['lift']:>6.2f}x")

    # ── Save outputs ───────────────────────────────────────────────────────
    today = date.today().strftime("%Y%m%d")
    tag   = f"{split}_{today}" if split != "all" else today

    scored_path   = SCORES_DIR / f"scored_{tag}.parquet"
    selected_path = SCORES_DIR / f"selected_{tag}.parquet"
    summary_path  = SCORES_DIR / f"summary_{tag}.json"

    scored.to_parquet(scored_path, index=False)
    scored[scored["selected"]].to_parquet(selected_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved:")
    print(f"  {scored_path}   ({len(scored):,} rows)")
    print(f"  {selected_path} ({summary['n_selected']:,} selected donors)")
    print(f"  {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BelFund EV Scorer")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="test",
        help="Which data split to score (default: test)"
    )
    args = parser.parse_args()
    main(split=args.split)
