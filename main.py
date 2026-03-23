"""
BelFund Donor EV Scorer — FastAPI Application
===============================================
Serves the EV scoring pipeline via HTTP endpoints.

Endpoints:
  GET  /health        — liveness check, model status
  GET  /model-info    — model version, metrics, features
  POST /score         — score a list of donors, return EV scores
  POST /select        — score donors and return selection decision
  GET  /docs          — interactive API docs (FastAPI auto-generated)

Usage:
  # Local development
  python -m uvicorn main:app --reload --port 8000

  # Docker
  docker run -p 8080:80 donor-ev-scorer

Example request:
  curl -X POST http://localhost:8000/score \\
    -H "Content-Type: application/json" \\
    -d @examples/sample_request.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
CAMPAIGN_TYPE    = os.getenv("CAMPAIGN_TYPE",    "FID")
COST_PER_MAIL    = float(os.getenv("COST_PER_MAIL", "1.50"))
EV_DECILE_CUTOFF = int(os.getenv("EV_DECILE_CUTOFF", "3"))
MODEL_VERSION    = os.getenv("MODEL_VERSION",    "v1")
MODELS_DIR       = Path("models")
METRICS_PATH     = MODELS_DIR / "training_metrics.json"

# ── Global model state ─────────────────────────────────────────────────────
_models = {}
_metrics = {}


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP — load models once at boot
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global _models, _metrics
    try:
        from src.score import load_models
        _models = load_models()
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                _metrics = json.load(f)
        print(f"Models loaded: propensity + amount ({_models['cal_type']} calibration)")
    except Exception as e:
        print(f"WARNING: Could not load models — {e}")
        print("Start the API and train models first: python src/train.py")
    yield
    _models.clear()


app = FastAPI(
    title="BelFund Donor EV Scorer",
    description=(
        "Expected value scoring API for direct mail donor selection. "
        "Scores donors using a dual model pipeline: "
        "P(donate) × predicted_amount = EV. "
        "Select donors where EV > cost_per_mail."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class DonorFeatures(BaseModel):
    """Feature vector for a single donor at campaign reference date T."""

    ind_id:                  int   = Field(..., description="Donor identifier")

    # Global history
    all_gifts_n:             float = Field(0.0, description="Total number of gifts ever")
    all_gifts_mean:          float = Field(0.0, description="Mean gift amount (€)")
    all_gifts_sum:           float = Field(0.0, description="Total lifetime giving (€)")
    all_gifts_median:        float = Field(0.0, description="Median gift amount (€)")
    all_gifts_max:           float = Field(0.0, description="Maximum gift amount (€)")
    all_days_since_last:     float = Field(730.0, description="Days since last gift")
    all_recency_score:       float = Field(0.0, description="Recency score 0-1 (1=very recent)")
    all_freq_per_yr_active:  float = Field(0.0, description="Gifts per active year")
    all_resp_rate:           float = Field(0.0, description="Historical response rate")
    all_sel_n:               float = Field(0.0, description="Times selected for campaigns")
    all_sel_cost_mean:       float = Field(1.50, description="Mean campaign cost per mail (€)")
    recency_adj_cltv:        float = Field(0.0, description="Recency-adjusted CLTV proxy")
    engagement_depth:        float = Field(0.0, description="Engagement depth score 0-1")
    lapse_risk_score:        float = Field(0.0, description="Lapse risk score 0-1")

    # Recent window (0-6 months)
    w_0_6m_gifts_n:          float = Field(0.0, description="Gifts in last 6 months")
    w_0_6m_gifts_mean:       float = Field(0.0, description="Mean gift last 6 months (€)")
    w_0_6m_resp_rate:        float = Field(0.0, description="Response rate last 6 months")

    # Prior window (6-12 months)
    w_6_12m_gifts_n:         float = Field(0.0, description="Gifts 6-12 months ago")
    w_6_12m_gifts_mean:      float = Field(0.0, description="Mean gift 6-12 months ago (€)")

    # Campaign cost override (optional)
    cost_unit:               Optional[float] = Field(None, description="Campaign cost per mail (€) — overrides default")

    class Config:
        json_schema_extra = {
            "example": {
                "ind_id": 100001,
                "all_gifts_n": 8,
                "all_gifts_mean": 35.0,
                "all_gifts_sum": 280.0,
                "all_gifts_median": 30.0,
                "all_gifts_max": 75.0,
                "all_days_since_last": 180,
                "all_recency_score": 0.75,
                "all_freq_per_yr_active": 2.5,
                "all_resp_rate": 0.45,
                "all_sel_n": 12,
                "all_sel_cost_mean": 1.10,
                "recency_adj_cltv": 85.0,
                "engagement_depth": 0.62,
                "lapse_risk_score": 0.15,
                "w_0_6m_gifts_n": 1,
                "w_0_6m_gifts_mean": 40.0,
                "w_0_6m_resp_rate": 0.50,
                "w_6_12m_gifts_n": 1,
                "w_6_12m_gifts_mean": 30.0,
                "cost_unit": 1.10
            }
        }


class ScoreRequest(BaseModel):
    campaign_type: str        = Field("FID", description="FID or REAC")
    cost_per_mail: float      = Field(1.50,  description="Default cost per mail (€) if not per-donor")
    ev_decile_cutoff: int     = Field(3,     description="Top N deciles to select (REAC only)")
    donors: List[DonorFeatures]

    class Config:
        json_schema_extra = {
            "example": {
                "campaign_type": "FID",
                "cost_per_mail": 1.10,
                "ev_decile_cutoff": 3,
                "donors": [DonorFeatures.Config.json_schema_extra["example"]]
            }
        }


class DonorScore(BaseModel):
    ind_id:          int
    prob_donate:     float = Field(..., description="Calibrated P(donate)")
    pred_amount_eur: float = Field(..., description="Predicted gift amount (€)")
    ev_eur:          float = Field(..., description="Expected value = P(donate) × amount")
    cost_unit:       float = Field(..., description="Mailing cost used")
    net_ev_eur:      float = Field(..., description="EV − cost")
    roi:             float = Field(..., description="EV / cost")
    ev_decile:       int   = Field(..., description="EV decile (10=best, 1=worst)")
    selected:        bool  = Field(..., description="Selection decision")


class ScoreSummary(BaseModel):
    campaign_type:          str
    n_total:                int
    n_selected:             int
    n_rejected:             int
    selection_rate:         float
    mean_ev_selected:       float
    mean_ev_rejected:       float
    total_expected_revenue: float
    total_mailing_cost:     float
    expected_net_revenue:   float
    mean_roi_selected:      float


class ScoreResponse(BaseModel):
    model_version:  str
    scored_at:      str
    scores:         List[DonorScore]
    summary:        ScoreSummary


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    campaign_type: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version:      str
    campaign_type:      str
    calibrator:         str
    n_features:         int
    amount_cap_eur:     float
    propensity_val_auc: Optional[float]
    amount_val_mae:     Optional[float]
    cost_source:        str


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    Liveness check. Returns model loaded status.
    AWS load balancers ping this every 30 seconds.
    """
    return HealthResponse(
        status        = "healthy",
        model_loaded  = bool(_models),
        campaign_type = CAMPAIGN_TYPE,
        model_version = MODEL_VERSION,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    """Return model metadata and training metrics."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    prop_auc = None
    amt_mae  = None
    if _metrics:
        try:
            prop_auc = _metrics["propensity_calibrated"]["val"]["auc"]
            amt_mae  = _metrics["amount"]["val"]["mae"]
        except (KeyError, TypeError):
            pass

    return ModelInfoResponse(
        model_version      = MODEL_VERSION,
        campaign_type      = CAMPAIGN_TYPE,
        calibrator         = _models.get("cal_type", "unknown"),
        n_features         = len(_models.get("feature_cols", [])),
        amount_cap_eur     = float(_models.get("amount_cap", 0)),
        propensity_val_auc = prop_auc,
        amount_val_mae     = amt_mae,
        cost_source        = "cost_unit per donor (all_sel_cost_mean) or request default",
    )


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score(request: ScoreRequest):
    """
    Score a list of donors and return EV scores with selection decisions.

    For FID campaigns: donor is selected if EV > cost_per_mail.
    For REAC campaigns: top N EV deciles are selected.

    The cost used per donor is:
      1. donor.cost_unit if provided in the request
      2. request.cost_per_mail (default fallback)
    """
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded — run python src/train.py first")

    if not request.donors:
        raise HTTPException(status_code=400, detail="donors list is empty")

    # ── Build feature DataFrame ────────────────────────────────────────────
    feature_cols = _models["feature_cols"]
    rows = []
    costs = []

    for d in request.donors:
        donor_dict = d.model_dump(exclude={"ind_id", "cost_unit"})
        row = {col: donor_dict.get(col, 0.0) for col in feature_cols}
        rows.append(row)
        cost = d.cost_unit if d.cost_unit is not None else request.cost_per_mail
        costs.append(cost)

    df = pd.DataFrame(rows, columns=feature_cols).fillna(0.0)
    df["all_sel_cost_mean"] = costs  # inject per-donor cost

    # ── Score ──────────────────────────────────────────────────────────────
    from src.score import score_donors
    scored = score_donors(
        df,
        _models,
        cost_col         = "all_sel_cost_mean",
        campaign_type    = request.campaign_type,
        ev_decile_cutoff = request.ev_decile_cutoff,
    )

    # ── Build response ─────────────────────────────────────────────────────
    ind_ids  = [d.ind_id for d in request.donors]
    scores_out = []
    for i, row in scored.iterrows():
        scores_out.append(DonorScore(
            ind_id          = ind_ids[i],
            prob_donate     = float(row["prob_donate"]),
            pred_amount_eur = float(row["pred_amount_eur"]),
            ev_eur          = float(row["ev_eur"]),
            cost_unit       = float(row["cost_unit"]),
            net_ev_eur      = float(row["net_ev_eur"]),
            roi             = float(row["roi"]),
            ev_decile       = int(row["ev_decile"]) if pd.notna(row["ev_decile"]) else 5,
            selected        = bool(row["selected"]),
        ))

    # ── Summary ────────────────────────────────────────────────────────────
    sel = scored[scored["selected"]]
    rej = scored[~scored["selected"]]
    n   = len(scored)

    summary = ScoreSummary(
        campaign_type          = request.campaign_type,
        n_total                = n,
        n_selected             = int(scored["selected"].sum()),
        n_rejected             = int((~scored["selected"]).sum()),
        selection_rate         = round(float(scored["selected"].mean()), 4),
        mean_ev_selected       = round(float(sel["ev_eur"].mean()),      2) if len(sel) else 0.0,
        mean_ev_rejected       = round(float(rej["ev_eur"].mean()),      2) if len(rej) else 0.0,
        total_expected_revenue = round(float(sel["ev_eur"].sum()),       2) if len(sel) else 0.0,
        total_mailing_cost     = round(float(sel["cost_unit"].sum()),    2) if len(sel) else 0.0,
        expected_net_revenue   = round(
            float(sel["ev_eur"].sum()) - float(sel["cost_unit"].sum()), 2
        ) if len(sel) else 0.0,
        mean_roi_selected      = round(float(sel["roi"].mean()),         3) if len(sel) else 0.0,
    )

    return ScoreResponse(
        model_version = MODEL_VERSION,
        scored_at     = date.today().isoformat(),
        scores        = scores_out,
        summary       = summary,
    )


@app.post("/select", response_model=ScoreResponse, tags=["Scoring"])
def select(request: ScoreRequest):
    """
    Identical to /score but returns only selected donors in scores list.
    Useful when you only want the mailing list, not the full scored population.
    """
    full_response = score(request)
    selected_scores = [s for s in full_response.scores if s.selected]
    full_response.scores = selected_scores
    return full_response
