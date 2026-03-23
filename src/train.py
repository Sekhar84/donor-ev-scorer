"""
BelFund Donor EV Scorer — Model Training
=========================================
Trains two models per campaign cycle:

  1. Propensity model  — binary classifier predicting P(donor gives | mailed)
                         Target: Gave_Donation (0/1)
                         Metric: ROC-AUC on validation set

  2. Amount model      — regressor predicting E(gift amount | donor gives)
                         Target: log1p(Amount), responders only, capped at p90
                         Metric: MAE in original € space on validation set

Both models use XGBoost by default — fast, handles missing values natively,
works well with RFM-style features without scaling.

Split strategy: time-based campaign-level (no leakage)
  Earlier campaigns → train (70%)
  Middle campaigns  → val   (15%)
  Latest campaigns  → test  (15%)

Usage:
  python src/train.py

Outputs saved to models/:
  propensity_model.joblib   — trained classifier
  amount_model.joblib       — trained regressor
  imputer_p.joblib          — imputer fit on train (propensity)
  imputer_a.joblib          — imputer fit on train (amount)
  amount_cap.joblib         — p90 cap fitted on train responders
  feature_cols.joblib       — feature column list (for scoring)
  training_metrics.json     — AUC, MAE, RMSE on val and test sets
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from pathlib import Path
from dotenv import load_dotenv

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

load_dotenv()
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
FEATURES_PATH   = Path(os.getenv("OUTPUT_DIR", "data/features")) / "features.parquet"
MODELS_DIR      = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED     = int(os.getenv("RANDOM_SEED",     "42"))
FRAC_TRAIN      = float(os.getenv("FRAC_TRAIN",    "0.70"))
FRAC_VAL        = float(os.getenv("FRAC_VAL",      "0.15"))
FRAC_TEST       = float(os.getenv("FRAC_TEST",      "0.15"))
RESPONSE_CAP_PCT= float(os.getenv("RESPONSE_CAP_PCT", "0.90"))

# ── MLflow config ─────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI",    "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "belfund-ev-scorer-fid")

# Point MLflow to local mlruns/ folder (no server needed)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Columns that are identifiers or targets — never features
NON_FEATURE_COLS = {
    "Ind_id", "Action_Group", "SelectionDate",
    "Gave_Donation", "Amount",
}


# ═══════════════════════════════════════════════════════════════════════════
# SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def time_split_by_campaign(df, group_col="Action_Group", date_col="SelectionDate"):
    """
    Split campaigns chronologically into train / val / test.
    Splitting at campaign level prevents temporal leakage:
    earlier campaigns → train, middle → val, latest → test.
    """
    campaign_dates = (
        pd.to_datetime(df[date_col], errors="coerce")
        .groupby(df[group_col])
        .min()
        .sort_values()
        .dropna()
    )
    groups  = campaign_dates.index.tolist()
    n       = len(groups)

    n_test  = max(1, int(np.ceil(FRAC_TEST  * n)))
    n_val   = max(1, int(np.ceil(FRAC_VAL   * n)))
    n_train = n - n_val - n_test

    assert n_train > 0, f"Not enough campaigns ({n}) for requested split fractions"

    train_g = set(groups[:n_train])
    val_g   = set(groups[n_train : n_train + n_val])
    test_g  = set(groups[n_train + n_val :])

    print(f"\nCampaign split:")
    print(f"  Train : {len(train_g)} campaigns | "
          f"up to {campaign_dates.reindex(list(train_g)).max().date()}")
    print(f"  Val   : {len(val_g)} campaigns  | "
          f"from {campaign_dates.reindex(list(val_g)).min().date()}")
    print(f"  Test  : {len(test_g)} campaigns  | "
          f"from {campaign_dates.reindex(list(test_g)).min().date()}")

    train_mask = df[group_col].isin(train_g)
    val_mask   = df[group_col].isin(val_g)
    test_mask  = df[group_col].isin(test_g)

    return train_mask, val_mask, test_mask


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def clf_metrics(y_true, y_prob, label=""):
    """Classifier metrics — AUC and average precision."""
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    rr  = float(np.mean(y_true))
    print(f"  {label:6s} | AUC={auc:.4f}  AP={ap:.4f}  RR={rr:.2%}")
    return {"auc": round(auc, 4), "ap": round(ap, 4), "rr": round(rr, 4)}


def reg_metrics(y_true, y_pred, label=""):
    """Regressor metrics in original € space."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    med  = float(np.median(np.abs(y_true - y_pred)))
    print(f"  {label:6s} | MAE=€{mae:.2f}  RMSE=€{rmse:.2f}  MedAE=€{med:.2f}")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "medae": round(med, 2)}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(
    run_name:      str   = "xgboost-dual-model",
    max_depth:     int   = 5,
    learning_rate: float = 0.05,
    n_estimators:  int   = 300,
    subsample:     float = 0.8,
):
    print("=" * 65)
    print("BelFund Donor EV Scorer — Model Training")
    print("=" * 65)

    # ── Load features ──────────────────────────────────────────────────────
    print(f"\nLoading: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"Shape: {df.shape}")
    print(f"Response rate: {df['Gave_Donation'].mean():.2%}")
    print(f"Responders:    {df['Gave_Donation'].sum():,}")

    # ── Feature columns ────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Feature columns: {len(feature_cols)}")

    # ── Time-based split ────────────────────────────────────────────────────
    tr_mask, va_mask, te_mask = time_split_by_campaign(df)

    print(f"\nRow counts:")
    print(f"  Train : {tr_mask.sum():,} rows  "
          f"(RR={df.loc[tr_mask,'Gave_Donation'].mean():.2%})")
    print(f"  Val   : {va_mask.sum():,} rows  "
          f"(RR={df.loc[va_mask,'Gave_Donation'].mean():.2%})")
    print(f"  Test  : {te_mask.sum():,} rows  "
          f"(RR={df.loc[te_mask,'Gave_Donation'].mean():.2%})")

    # ── Open MLflow run ────────────────────────────────────────────────────
    # Everything inside this block is recorded as one run
    with mlflow.start_run(run_name=run_name) as run:

        print(f"\nMLflow run ID: {run.info.run_id}")

        # ── Log parameters — all hyperparameters and config ────────────────
        # Parameters are inputs to the training run
        mlflow.log_params({
            # Data
            "n_features":          len(feature_cols),
            "train_rows":          int(tr_mask.sum()),
            "val_rows":            int(va_mask.sum()),
            "test_rows":           int(te_mask.sum()),
            "train_rr":            round(float(df.loc[tr_mask,"Gave_Donation"].mean()), 4),
            "response_cap_pct":    RESPONSE_CAP_PCT,
            # Propensity model
            "clf_n_estimators":    n_estimators,
            "clf_max_depth":       max_depth,
            "clf_learning_rate":   learning_rate,
            "clf_subsample":       subsample,
            "clf_colsample_bytree":0.8,
            "clf_min_child_weight":10,
            # Amount model
            "reg_n_estimators":    n_estimators,
            "reg_max_depth":       max_depth,
            "reg_learning_rate":   learning_rate,
            "reg_subsample":       subsample,
            "reg_colsample_bytree":0.8,
            "reg_min_child_weight":5,
            # Split
            "frac_train":          FRAC_TRAIN,
            "frac_val":            FRAC_VAL,
            "frac_test":           FRAC_TEST,
            "random_seed":         RANDOM_SEED,
        })

        # ── Log tags — metadata about this run ─────────────────────────────
        # Tags are searchable labels, not numeric values
        mlflow.set_tags({
            "campaign_type":  os.getenv("CAMPAIGN_TYPE", "FID"),
            "client":         os.getenv("CLIENT_NAME",   "BelFund"),
            "model_version":  os.getenv("MODEL_VERSION", "v1"),
            "sim_mode":       os.getenv("SIM_MODE",      "true"),
            "framework":      "xgboost",
        })

        # ── Propensity model inputs ─────────────────────────────────────────
        X_train_p = df.loc[tr_mask, feature_cols]
        X_val_p   = df.loc[va_mask, feature_cols]
        X_test_p  = df.loc[te_mask, feature_cols]

        y_train_p = df.loc[tr_mask, "Gave_Donation"].values
        y_val_p   = df.loc[va_mask, "Gave_Donation"].values
        y_test_p  = df.loc[te_mask, "Gave_Donation"].values

        # Impute missing values — fit on train only
        imp_p = SimpleImputer(strategy="median")
        X_train_p_arr = imp_p.fit_transform(X_train_p)
        X_val_p_arr   = imp_p.transform(X_val_p)
        X_test_p_arr  = imp_p.transform(X_test_p)

        # SimpleImputer drops all-NaN columns — get surviving feature names
        feature_cols_p = [feature_cols[i] for i in range(len(feature_cols))
                          if not np.all(np.isnan(X_train_p.values[:, i]))]
        if len(feature_cols_p) != X_train_p_arr.shape[1]:
            feature_cols_p = [f"f{i}" for i in range(X_train_p_arr.shape[1])]

        X_train_p_df = pd.DataFrame(X_train_p_arr, columns=feature_cols_p)
        X_val_p_df   = pd.DataFrame(X_val_p_arr,   columns=feature_cols_p)
        X_test_p_df  = pd.DataFrame(X_test_p_arr,  columns=feature_cols_p)

        # ── Amount model inputs ─────────────────────────────────────────────
        tr_resp = tr_mask & (df["Amount"] > 0)
        va_resp = va_mask & (df["Amount"] > 0)
        te_resp = te_mask & (df["Amount"] > 0)

        AMOUNT_CAP = float(df.loc[tr_resp, "Amount"].quantile(RESPONSE_CAP_PCT))
        print(f"\nAmount cap (p{int(RESPONSE_CAP_PCT*100)} of train responders): "
              f"€{AMOUNT_CAP:.2f}")

        # Log amount cap as parameter
        mlflow.log_param("amount_cap_eur", round(AMOUNT_CAP, 2))

        def cap_and_log(s):
            return np.log1p(s.clip(upper=AMOUNT_CAP))

        X_train_a = df.loc[tr_resp, feature_cols]
        X_val_a   = df.loc[va_resp, feature_cols]
        X_test_a  = df.loc[te_resp, feature_cols]

        y_train_a     = df.loc[tr_resp, "Amount"].values
        y_val_a       = df.loc[va_resp, "Amount"].values
        y_test_a      = df.loc[te_resp, "Amount"].values
        y_train_a_log = cap_and_log(pd.Series(y_train_a)).values
        y_val_a_log   = cap_and_log(pd.Series(y_val_a)).values
        y_test_a_log  = cap_and_log(pd.Series(y_test_a)).values

        imp_a = SimpleImputer(strategy="median")
        X_train_a_arr = imp_a.fit_transform(X_train_a)
        X_val_a_arr   = imp_a.transform(X_val_a)
        X_test_a_arr  = imp_a.transform(X_test_a)

        feature_cols_a = [feature_cols[i] for i in range(len(feature_cols))
                          if not np.all(np.isnan(X_train_a.values[:, i]))]
        if len(feature_cols_a) != X_train_a_arr.shape[1]:
            feature_cols_a = [f"f{i}" for i in range(X_train_a_arr.shape[1])]

        X_train_a_df = pd.DataFrame(X_train_a_arr, columns=feature_cols_a)
        X_val_a_df   = pd.DataFrame(X_val_a_arr,   columns=feature_cols_a)
        X_test_a_df  = pd.DataFrame(X_test_a_arr,  columns=feature_cols_a)

        print(f"\nAmount model:")
        print(f"  Train responders: {tr_resp.sum():,}")
        print(f"  Val responders:   {va_resp.sum():,}")
        print(f"  Test responders:  {te_resp.sum():,}")

        # ── Train propensity model ──────────────────────────────────────────
        print("\n" + "─" * 65)
        print("Training propensity model (XGBoost classifier)...")

        scale_pos_weight = float((y_train_p == 0).sum() / (y_train_p == 1).sum())
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}  "
              f"(handles class imbalance)")
        mlflow.log_param("clf_scale_pos_weight", round(scale_pos_weight, 2))

        clf = XGBClassifier(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            learning_rate     = learning_rate,
            subsample         = subsample,
            colsample_bytree  = 0.8,
            min_child_weight  = 10,
            scale_pos_weight  = scale_pos_weight,
            use_label_encoder = False,
            eval_metric       = "auc",
            early_stopping_rounds = 20,
            random_state      = RANDOM_SEED,
            n_jobs            = -1,
            verbosity         = 0,
        )

        clf.fit(
            X_train_p_df, y_train_p,
            eval_set=[(X_val_p_df, y_val_p)],
            verbose=False,
        )

        print(f"  Best iteration: {clf.best_iteration}")
        mlflow.log_param("clf_best_iteration", clf.best_iteration)

        print("\nPropensity model metrics (raw scores):")
        prob_val_p   = clf.predict_proba(X_val_p_df)[:, 1]
        prob_test_p  = clf.predict_proba(X_test_p_df)[:, 1]
        prob_train_p = clf.predict_proba(X_train_p_df)[:, 1]

        clf_m = {
            "train": clf_metrics(y_train_p, prob_train_p, "train"),
            "val":   clf_metrics(y_val_p,   prob_val_p,   "val"),
            "test":  clf_metrics(y_test_p,  prob_test_p,  "test"),
        }

        # ── Calibration comparison ─────────────────────────────────────────
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prob_val_p, y_val_p)
        prob_val_iso  = iso.predict(prob_val_p)
        prob_test_iso = iso.predict(prob_test_p)

        platt = LogisticRegression(C=1.0)
        platt.fit(prob_val_p.reshape(-1, 1), y_val_p)
        prob_val_platt  = platt.predict_proba(prob_val_p.reshape(-1, 1))[:, 1]
        prob_test_platt = platt.predict_proba(prob_test_p.reshape(-1, 1))[:, 1]

        actual_rr = float(y_val_p.mean())

        print(f"\nCalibration comparison (val set):")
        print(f"  {'Method':<12}  {'AUC':>6}  {'Mean score':>10}  "
              f"{'Actual RR':>10}  {'Cal error':>10}  {'Score std':>10}")
        print(f"  {'─'*12}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

        results = {}
        for name, probs in [("Raw",      prob_val_p),
                             ("Isotonic", prob_val_iso),
                             ("Platt",    prob_val_platt)]:
            auc_c = roc_auc_score(y_val_p, probs)
            cal_e = abs(probs.mean() - actual_rr)
            spread = probs.std()
            results[name] = {
                "auc": auc_c, "cal_error": cal_e,
                "mean": probs.mean(), "std": spread
            }
            print(f"  {name:<12}  {auc_c:>6.4f}  {probs.mean():>10.4f}  "
                  f"{actual_rr:>10.4f}  {cal_e:>10.4f}  {spread:>10.4f}")

        auc_threshold = results["Raw"]["auc"] - 0.002
        candidates = {
            k: v for k, v in results.items()
            if k != "Raw" and v["auc"] >= auc_threshold
        }
        best_cal_name = min(candidates, key=lambda k: candidates[k]["cal_error"])
        print(f"\n  Best calibrator: {best_cal_name} "
              f"(cal_error={results[best_cal_name]['cal_error']:.4f})")

        if best_cal_name == "Isotonic":
            calibrator     = iso
            prob_val_cal   = prob_val_iso
            prob_test_cal  = prob_test_iso
            prob_train_cal = iso.predict(prob_train_p)
        else:
            calibrator     = platt
            prob_val_cal   = prob_val_platt
            prob_test_cal  = prob_test_platt
            prob_train_cal = platt.predict_proba(prob_train_p.reshape(-1, 1))[:, 1]

        print(f"\nFinal propensity metrics (calibrated — {best_cal_name}):")
        clf_m_cal = {
            "train": clf_metrics(y_train_p, prob_train_cal, "train"),
            "val":   clf_metrics(y_val_p,   prob_val_cal,   "val"),
            "test":  clf_metrics(y_test_p,  prob_test_cal,  "test"),
        }

        # Top features
        fi = pd.Series(
            clf.feature_importances_,
            index=clf.get_booster().feature_names
        ).sort_values(ascending=False)
        print("\nTop 10 features (propensity):")
        for feat, imp in fi.head(10).items():
            print(f"  {feat:<45} {imp:.4f}")

        # ── Train amount model ──────────────────────────────────────────────
        print("\n" + "─" * 65)
        print("Training amount model (XGBoost regressor, log-space)...")

        reg = XGBRegressor(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = 0.8,
            min_child_weight = 5,
            early_stopping_rounds = 20,
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
            verbosity        = 0,
        )

        reg.fit(
            X_train_a_df, y_train_a_log,
            eval_set=[(X_val_a_df, y_val_a_log)],
            verbose=False,
        )

        print(f"  Best iteration: {reg.best_iteration}")
        mlflow.log_param("reg_best_iteration", reg.best_iteration)

        pred_val_a   = np.expm1(reg.predict(X_val_a_df)).clip(0)
        pred_test_a  = np.expm1(reg.predict(X_test_a_df)).clip(0)
        pred_train_a = np.expm1(reg.predict(X_train_a_df)).clip(0)

        print("\nAmount model metrics (€ space):")
        reg_m = {
            "train": reg_metrics(y_train_a, pred_train_a, "train"),
            "val":   reg_metrics(y_val_a,   pred_val_a,   "val"),
            "test":  reg_metrics(y_test_a,  pred_test_a,  "test"),
        }

        fi_r = pd.Series(
            reg.feature_importances_,
            index=reg.get_booster().feature_names
        ).sort_values(ascending=False)
        print("\nTop 10 features (amount):")
        for feat, imp in fi_r.head(10).items():
            print(f"  {feat:<45} {imp:.4f}")

        # ── Log all metrics to MLflow ───────────────────────────────────────
        # Metrics are numeric outputs — logged here after both models are done
        mlflow.log_metrics({
            # Propensity — raw
            "clf_train_auc_raw":   clf_m["train"]["auc"],
            "clf_val_auc_raw":     clf_m["val"]["auc"],
            "clf_test_auc_raw":    clf_m["test"]["auc"],
            "clf_val_ap_raw":      clf_m["val"]["ap"],
            # Propensity — calibrated
            "clf_train_auc_cal":   clf_m_cal["train"]["auc"],
            "clf_val_auc_cal":     clf_m_cal["val"]["auc"],
            "clf_test_auc_cal":    clf_m_cal["test"]["auc"],
            # Calibration quality
            "cal_error_isotonic":  float(results["Isotonic"]["cal_error"]),
            "cal_error_platt":     float(results["Platt"]["cal_error"]),
            "cal_score_std":       float(results[best_cal_name]["std"]),
            # Amount model
            "reg_train_mae":       reg_m["train"]["mae"],
            "reg_val_mae":         reg_m["val"]["mae"],
            "reg_test_mae":        reg_m["test"]["mae"],
            "reg_train_rmse":      reg_m["train"]["rmse"],
            "reg_val_rmse":        reg_m["val"]["rmse"],
            "reg_test_rmse":       reg_m["test"]["rmse"],
            "reg_val_medae":       reg_m["val"]["medae"],
        })
        mlflow.log_param("calibrator", best_cal_name)

        # ── EV sanity check ─────────────────────────────────────────────────
        print("\n" + "─" * 65)
        print("EV sanity check (test set — calibrated scores, actual cost):")

        pred_test_amt_all = np.expm1(reg.predict(X_test_p_df)).clip(0)
        ev_test = prob_test_cal * pred_test_amt_all

        if "all_sel_cost_mean" in df.columns:
            cost_test = df.loc[te_mask, "all_sel_cost_mean"].values
            cost_test = np.where(cost_test <= 0,
                                 float(os.getenv("COST_PER_MAIL", "1.50")),
                                 cost_test)
        else:
            cost_test = np.full(len(ev_test), float(os.getenv("COST_PER_MAIL", "1.50")))

        net_ev   = ev_test - cost_test
        selected = (net_ev > 0).sum()

        print(f"  Test donors scored:    {len(ev_test):,}")
        print(f"  Cost per mail:  mean=€{cost_test.mean():.2f}  "
              f"median=€{np.median(cost_test):.2f}")
        print(f"  EV distribution (calibrated):")
        print(f"    mean=€{ev_test.mean():.2f}  "
              f"median=€{np.median(ev_test):.2f}  "
              f"p90=€{np.percentile(ev_test, 90):.2f}")
        print(f"  Donors selected (EV > cost): {selected:,} "
              f"({selected/len(ev_test):.1%})")
        print(f"  Donors not selected:         {len(ev_test)-selected:,} "
              f"({(len(ev_test)-selected)/len(ev_test):.1%})")

        # Log EV metrics
        mlflow.log_metrics({
            "ev_mean_eur":        round(float(ev_test.mean()), 2),
            "ev_median_eur":      round(float(np.median(ev_test)), 2),
            "ev_p90_eur":         round(float(np.percentile(ev_test, 90)), 2),
            "selection_rate":     round(float(selected / len(ev_test)), 4),
            "cost_mean_eur":      round(float(cost_test.mean()), 2),
        })

        # ── Save artefacts ──────────────────────────────────────────────────
        print("\n" + "─" * 65)
        print("Saving artefacts...")

        joblib.dump(clf,          MODELS_DIR / "propensity_model.joblib")
        joblib.dump(calibrator,   MODELS_DIR / "propensity_calibrator.joblib")
        joblib.dump(reg,          MODELS_DIR / "amount_model.joblib")
        joblib.dump(imp_p,        MODELS_DIR / "imputer_p.joblib")
        joblib.dump(imp_a,        MODELS_DIR / "imputer_a.joblib")
        joblib.dump(AMOUNT_CAP,   MODELS_DIR / "amount_cap.joblib")
        joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
        joblib.dump(best_cal_name, MODELS_DIR / "calibrator_type.joblib")

        model_feature_names_p = clf.get_booster().feature_names
        model_feature_names_a = reg.get_booster().feature_names
        joblib.dump(model_feature_names_p, MODELS_DIR / "model_feature_names_p.joblib")
        joblib.dump(model_feature_names_a, MODELS_DIR / "model_feature_names_a.joblib")

        # ── Log artefacts to MLflow ─────────────────────────────────────────
        # Artefacts are files attached to a run — downloadable from the UI
        mlflow.log_artifacts(str(MODELS_DIR), artifact_path="models")

        metrics = {
            "propensity_raw":        clf_m,
            "propensity_calibrated": clf_m_cal,
            "calibrator":            best_cal_name,
            "calibration_comparison": {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in results.items()
            },
            "amount":                reg_m,
            "amount_cap":            float(AMOUNT_CAP),
            "n_features":            len(feature_cols),
            "train_rows":            int(tr_mask.sum()),
            "val_rows":              int(va_mask.sum()),
            "test_rows":             int(te_mask.sum()),
            "cost_source":           "cost_unit per campaign (all_sel_cost_mean)",
            "note":                  "EV = calibrated_P(donate) x predicted_amount. Select if EV > cost_unit",
        }
        with open(MODELS_DIR / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

        print(f"  propensity_model.joblib")
        print(f"  propensity_calibrator.joblib  ({best_cal_name})")
        print(f"  calibrator_type.joblib")
        print(f"  amount_model.joblib")
        print(f"  imputer_p.joblib")
        print(f"  imputer_a.joblib")
        print(f"  amount_cap.joblib             (€{AMOUNT_CAP:.2f})")
        print(f"  feature_cols.joblib           ({len(feature_cols)} features)")
        print(f"  training_metrics.json")
        print(f"  → all logged to MLflow run: {run.info.run_id[:8]}...")

        print("\n" + "=" * 65)
        print("Training complete")
        print(f"  Propensity val AUC (raw):        {clf_m['val']['auc']:.4f}")
        print(f"  Propensity val AUC (calibrated): {clf_m_cal['val']['auc']:.4f}")
        print(f"  Calibrator used:                 {best_cal_name}")
        print(f"  Amount val MAE:                  €{reg_m['val']['mae']:.2f}")
        print(f"  MLflow experiment:               {MLFLOW_EXPERIMENT_NAME}")
        print(f"  MLflow run ID:                   {run.info.run_id}")
        print(f"  View UI:                         mlflow ui")
        print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BelFund EV Scorer — Model Training")
    parser.add_argument("--run-name",       default="xgboost-dual-model", help="MLflow run name")
    parser.add_argument("--max-depth",      type=int,   default=5,    help="XGBoost max_depth")
    parser.add_argument("--learning-rate",  type=float, default=0.05, help="XGBoost learning_rate")
    parser.add_argument("--n-estimators",   type=int,   default=300,  help="XGBoost n_estimators")
    parser.add_argument("--subsample",      type=float, default=0.8,  help="XGBoost subsample")
    args = parser.parse_args()
    main(
        run_name      = args.run_name,
        max_depth     = args.max_depth,
        learning_rate = args.learning_rate,
        n_estimators  = args.n_estimators,
        subsample     = args.subsample,
    )
