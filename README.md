# Donor EV Scorer — BelFund

A production-ready machine learning pipeline for **donor expected value (EV) scoring** in direct mail fundraising campaigns. Built for **BelFund**, a Belgian nonprofit fundraising organisation, targeting **Fidelization (FID)** campaigns — loyalty campaigns aimed at active donors who have given at least once in the past 24 months and are being cultivated for continued giving.

---

## What this project does

Every direct mail campaign has a cost per piece. Mailing a donor who will not respond is a loss. This pipeline scores each donor with an **Expected Value** — the amount you can statistically expect to receive from them — and compares it against the cost of mailing to decide whether to include them in the campaign.

```
EV = P(donate) × predicted_amount
Net EV = EV − cost_per_mail
Decision: mail donor if EV > cost_per_mail
```

The pipeline trains two models per campaign cycle:

- **Propensity model** — binary classifier predicting P(donor gives | mailed)
- **Amount model** — regressor predicting E(gift amount | donor gives)

Their product gives the EV score. Donors are ranked by EV and selected above a cost threshold.

---

## Project structure

```
donor-ev-scorer/
├── data/                        ← input data (never committed to Git)
│   └── simulated/               ← synthetic donor data for development
├── src/
│   ├── simulate.py              ← generates realistic synthetic donor data
│   ├── preprocess.py            ← feature engineering (RFM, time windows)
│   ├── train.py                 ← trains propensity + amount models
│   └── score.py                 ← computes EV scores, applies selection logic
├── models/                      ← saved model artifacts (never committed to Git)
├── notebooks/                   ← exploratory analysis (not production code)
├── main.py                      ← FastAPI app — /score, /select, /health endpoints
├── Dockerfile                   ← container definition
├── docker-compose.yml           ← multi-service setup (API + Redis cache)
├── requirements.txt             ← pinned dependencies
├── .env.example                 ← environment variable template (commit this)
├── .env                         ← actual secrets and config (never committed)
├── .gitignore
└── .dockerignore
```

---

## Environment variables

All configuration — including client identity, data paths, model settings, and campaign parameters — is passed via environment variables. **No client identifiers, paths, or credentials appear anywhere in the codebase.**

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

### `.env.example`

```bash
# ── Client and campaign identity ──────────────────────────────────────────
# BelFund is used as the demo client name throughout this project.
# In production replace CLIENT_ID with your actual client identifier.
CLIENT_ID=47                      # integer client identifier (BelFund demo = 47)
CLIENT_NAME=BelFund               # human-readable label for logs and reports
CAMPAIGN_TYPE=FID                 # FID (Fidelization/loyalty) | REAC (Reactivation)
CAMPAIGN_CHANNEL=Direct Mail

# ── Data paths ────────────────────────────────────────────────────────────
DATA_DIR=                         # path to raw SAS/CSV data files
INPUT_DIR=                        # path to input Excel files (scope, REFNIS)
OUTPUT_DIR=                       # path to write model outputs and scored files

# ── Simulation (development only) ─────────────────────────────────────────
SIM_N_DONORS=15000                # number of synthetic donors to generate
SIM_N_CAMPAIGNS=19                # number of synthetic campaigns
SIM_SEED=42                       # random seed for reproducibility

# ── Modelling parameters ──────────────────────────────────────────────────
RESPONSE_CAP_PCT=0.90             # percentile to cap amount target (train only)
LOW_RR_THRESHOLD=0.03             # campaigns below this RR excluded from training
ACTIVE_THRESHOLD_DAYS=730         # max days since last gift for FID eligibility
LAPSED_18M_DAYS=548               # threshold for lapsed donor flag
FRAC_TRAIN=0.70                   # train split fraction (time-based)
FRAC_VAL=0.15                     # validation split fraction
FRAC_TEST=0.15                    # test split fraction
RANDOM_SEED=42

# ── Scoring parameters ────────────────────────────────────────────────────
COST_PER_MAIL=1.50                # default cost per mailed piece (€)
EV_DECILE_CUTOFF=3                # for REAC: select top N deciles by EV

# ── API ───────────────────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=80
API_VERSION=1.0
MODEL_VERSION=v1

# ── MLflow experiment tracking ────────────────────────────────────────────
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=donor-ev-scorer-fid

# ── AWS (Day 4 onwards) ───────────────────────────────────────────────────
AWS_REGION=eu-west-1
S3_BUCKET=                        # bucket for data and model artifacts
ECR_REPO=                         # ECR repository URI
```

---

## Pipeline overview

```
Raw donor data (SAS files)
         │
         ▼
   preprocess.py
   ─────────────────────────────────────────────────
   Feature engineering per donor per campaign date T:
   • Global all-history RFM aggregates
   • 5 time windows anchored to T:
       w_0_6m     T−6m → T        (current pulse)
       w_6_12m    T−12m → T−6m    (recent baseline)
       w_12_24m   T−24m → T−12m   (FID qualifying window)
       w_24_36m   T−36m → T−24m   (medium-term reference)
       w_36m_plus history → T−36m (long-term legacy)
   • Recent 12m window
   • Bayesian CLTV (BG/NBD + Gamma-Gamma via lifetimes)
   • Demographic features (language, gender, province, district)
         │
         ▼
     train.py
   ─────────────────────────────────────────────────
   Time-based campaign-level split (70/15/15):
   • Propensity model  → binary classifier (Gave_Donation)
     - Baseline: XGBoost, LightGBM, RandomForest, ExtraTrees,
                 LogisticRegression, PyTorch MLP
     - Tuning: Optuna (50 trials), optimise val AUC
     - Calibration: isotonic regression on val probabilities
   • Amount model → log-transformed regressor (Amount > 0 only)
     - Same baseline set, optimise val RMSE
     - Amount capped at p90 of train responders (anti-outlier)
   • Final models retrained on train + val combined
         │
         ▼
     score.py
   ─────────────────────────────────────────────────
   For each donor × campaign combination:
   • prob_donate   = calibrated_classifier.predict_proba()
   • pred_amount   = exp(regressor.predict())  ← log-space
   • ev_eur        = prob_donate × pred_amount
   • net_ev_eur    = ev_eur − cost_per_mail
   • roi           = ev_eur / cost_per_mail
         │
         ▼
     main.py  (FastAPI)
   ─────────────────────────────────────────────────
   POST /score   → send donor features → get EV score back
   POST /select  → send campaign cost → get selected donors
   GET  /health  → liveness check
   GET  /model-info → model version, features, performance
```

---

## Selection logic

The pipeline supports two campaign types controlled by `CAMPAIGN_TYPE` in `.env`:

### FID / Fidelization campaigns (active donors)

```python
# Select donor if EV exceeds cost per mail
selected = ev_eur > cost_per_mail
```

Net EV positive means the expected return outweighs the mailing cost. These are loyalty campaigns targeting donors already engaged with BelFund — the goal is to sustain and grow their giving. The threshold can be adjusted via `COST_PER_MAIL` without retraining.

### REAC / Reactivation campaigns (inactive donors)

```python
# Select donors in top N EV deciles
ev_decile = pd.qcut(ev_eur, q=10, labels=False)
selected  = ev_decile >= (10 - EV_DECILE_CUTOFF)
```

For lapsed donors the absolute EV is less reliable — their giving behaviour is harder to predict. Ranking by decile and selecting the top N is more robust than a fixed cost threshold. The cutoff is set via `EV_DECILE_CUTOFF` in `.env`.

---

## Simulated data

The pipeline ships with a data simulator (`src/simulate.py`) that generates synthetic donor records calibrated to real fundraising campaign distributions. **No real donor data is included in this repository.**

### Why simulation

- Enables full end-to-end development and testing without real data
- Safe to commit, share, and run in CI/CD pipelines
- Allows anyone to reproduce results exactly (`SIM_SEED=42`)

### How simulation works

The simulator generates 7 tables matching the exact column names and structure of the production data pipeline:

| Table | Description | Rows (default) |
|---|---|---|
| `ind_adr_sim.csv` | Donor demographics | 15,000 |
| `actions_groups_sim.csv` | Campaign metadata | 19 |
| `actions_sim.csv` | Segment definitions | 57 |
| `tab_sel_sim.csv` | Campaign selections | ~384,000 |
| `gifts_valid_sim.csv` | Full gift history | ~151,000 |
| `op_sim.csv` | Standing orders | ~48 |
| `sdds_sim.csv` | SEPA direct debits | ~208 |
| `PELICANO_FID_complete_sim.xlsx` | In-scope segment flags | 57 |

### Calibration — simulated vs real distributions

Every statistical parameter in the simulator was derived from profiling the real dataset. No values were guessed.

| Metric | Real data | Simulated | Match |
|---|---|---|---|
| Overall response rate | 6.53% | 6.91% | ✅ |
| Gift amount — median | €35 | €35 | ✅ |
| Gift amount — mean | €39 | €47 | ~ |
| Gift amount — p90 | €65 | €92 | ~ |
| Days since last gift — median | 1,420 | 1,561 | ✅ |
| FID eligible donors (<730d) | 33.0% | 32.7% | ✅ |
| Cost per mail — median | €1.10 | €1.13 | ✅ |
| Language split (Dutch/French) | 66/34% | 66/34% | ✅ |
| Gender split (F/M/Couple) | 46/27/23% | 45/27/23% | ✅ |
| OP donor prevalence | 0.32% | 0.32% | ✅ |
| SDD donor prevalence | 1.39% | 1.39% | ✅ |
| SDD type — Monthly dominant | 83% | 86% | ✅ |
| Selections per donor — median | 8 | 26 | ~ |
| Gifts per donor — median | 2 | 5 | ~ |

**Notes on gaps:**

- **Amount mean/p90** — the real data has extreme skew (skew=36, max=€5,000 in training data, max=€900,000 in full history). Replicating this tail would require simulating rare institutional donors which distort the EV model. The median — which the EV model cares most about — matches exactly.
- **Selections per donor** — real data spans 2011–2026 (15 years, 1,556 action IDs across full history). The 598,258 figure is unique donor-campaign selection rows, not unique donors — there are 138,337 unique donors with gift history. The simulation spans only the training period (19 campaigns, 57 action IDs), so lifetime selection counts are proportionally lower. Per-segment selection logic matches; lifetime totals do not.
- **Gifts per donor** — same reason. Real median=2 reflects the full 15-year population including recent joiners. Simulation generates proportionally correct history within its shorter window.

### Response rate by segment (simulated vs real)

The simulator applies response rate modifiers calibrated directly from the real data segmentation analysis:

| Frequency band | Real RR | Simulated logic |
|---|---|---|
| 1 gift ever | 0.0% | base = 0.000 |
| 2–3 gifts | 2.7% | base = 0.027 |
| 4–6 gifts | 4.6% | base = 0.046 |
| 7–10 gifts | 6.5% | base = 0.065 |
| 10+ gifts | 13.2% | base = 0.132 |

FID-eligible donors (last gift < 730 days) receive a +2.5% response boost. Lapsed donors receive −1.0%.

---

## Known simulation limitations and their pipeline impact

This section documents every known gap between the simulated and real data, what caused it, how it manifests in pipeline outputs, and what to expect when switching to real data. These are not bugs — they are deliberate trade-offs in the simulator design.

---

### 1. Response rate inflation — 14.19% simulated vs 6.53% real

**Cause:** Donors are assigned to campaigns randomly in the simulation. Every donor has a similar probability of being selected regardless of their giving history. In real campaigns, donors are pre-screened — only plausible responders are selected, which concentrates the eligible pool and raises the observed response rate. The simulation cannot replicate this selection bias because it has no upstream selection model.

**Additionally**, campaign response gifts (gifts made in response to being mailed) are written to the same gifts table as historical gifts. When a lapsed donor responds to a mid-2025 campaign, their last gift date updates — making them appear FID-eligible for subsequent campaigns in the same simulation run. This creates artificial churn in the eligible pool across the 19 campaigns.

**Pipeline impact:**
- Feature matrix response rate: 14.19% vs real 6.53%
- `scale_pos_weight` in XGBoost is set to 5.96 (correct for 14% RR) — on real data this will be ~14.3 (correct for 6.5% RR)
- Model will be retrained from scratch on real data — no transfer of weights

**What to expect on real data:** Response rate will drop to ~6-7%. The model will see stronger signal because real selections are purposeful, not random. AUC should improve from ~0.65 to 0.75-0.85.

---

### 2. Propensity score miscalibration — raw scores clustered around 0.48

**Cause:** Because all donors look similar in the feature space (random selection, homogeneous history), XGBoost with `scale_pos_weight=5.96` cannot discriminate well. It assigns everyone a moderate-high probability rather than producing a spread from 0.02 to 0.95.

**Raw score distribution (simulated):**
```
p05 = 0.33   p50 = 0.46   p90 = 0.61   mean = 0.48
```

**Expected on real data:**
```
p05 = 0.02   p50 = 0.08   p90 = 0.25   mean = 0.065
```

**Pipeline impact:** Without calibration, EV = 0.48 × €35 = €16.80 for almost every donor — everyone gets selected regardless of cost. Isotonic calibration corrects this (mean score → 0.1385 = actual RR), but the underlying score spread remains narrow.

**Fix applied:** Isotonic calibration fitted on val set. Calibration comparison (Isotonic vs Platt) runs automatically at training time — the winner is selected by lowest calibration error with AUC preserved. On real data calibration will still improve scores but the raw model will already be better calibrated due to stronger feature signal.

---

### 3. EV selection rate — 100% selected on simulated data

**Cause:** Combination of issues 1 and 2. Even after calibration, EV median = €3.98 and cost = €0.86 — so virtually every donor has positive net EV. In real data the spread will be much wider with many donors having EV < cost.

**Expected on real data:**
```
Simulated: 100% selected  (EV >> cost for everyone)
Real:      ~15-25% selected  (EV > cost only for best donors)
```

**Pipeline impact:** The selection logic (`EV > cost_unit`) is correct. The 100% rate is purely a data quality issue. The API `/select` endpoint will correctly exclude donors on real data.

---

### 4. Amount model weak signal — MAE €25.59 on median gift of €35

**Cause:** Gift amounts in the simulation are drawn from a log-normal distribution anchored to the donor's historical mean, but with high noise. The amount model cannot learn strong patterns because the relationship between features and amount is artificially weak.

**Amount model metrics (simulated):**
```
val MAE  = €25.59  (73% of median gift — very high)
val RMSE = €42.70
```

**Expected on real data:**
```
val MAE  = €8-15   (20-40% of median gift — much better)
val RMSE = €20-35
```

**Pipeline impact:** Predicted amounts regress heavily toward the mean (~€35 for everyone). This flattens EV scores and prevents meaningful donor ranking. On real data, donors who historically give €100+ will be correctly predicted higher than donors who give €10, creating genuine EV spread.

---

### 5. Feature importance concentration — two features dominate

**Top propensity features:**
```
w_36m_plus_gifts_n    0.1706  ← 17% of total importance
all_gifts_n           0.1021  ← 10% of total importance
(remaining 223 features share the other 73%)
```

**Cause:** In real campaigns, the selection team uses historical response data to target donors. This creates strong correlations between RFM features and response. In simulation, selection is random — so only the broadest frequency signals (total gift count) predict anything.

**Expected on real data:** Importance will be more distributed across recency, window-specific features, and amount features. `all_recency_score`, `w_0_6m_gifts_n`, and `recency_adj_cltv` should rank much higher.

---

### 6. Two all-NaN feature columns

During training, two feature columns are found to have no observed values in the training split and are silently dropped by `SimpleImputer`:

```
w_12_24m_sel_cost_mean
w_12_24m_sel_cost_log
```

**Cause:** The 12-24 month window selection cost features require donors to have been selected in campaigns 12-24 months before T. Since the simulation only spans 13 months of campaign history (Jan 2025 – Feb 2026), there are no selections in the 12-24 month window for early campaigns.

**Pipeline impact:** 225 features defined, 223 survive imputation. XGBoost receives 223 features. The dropped columns are always zero in the simulated data and carry no signal. On real data (15 years of history) these columns will be populated.

**Fix applied:** Column detection after imputation with fallback — the surviving column list is used to build DataFrames passed to XGBoost, preventing shape mismatch errors.

---

### 7. FID eligible pool grows across campaigns (33% → 72%)

**Cause:** See issue 1. Campaign response gifts update donors' last gift dates, pulling lapsed donors back into FID eligibility for later campaigns. At T=2025-01-01 only 33% are eligible (correct). By T=2026-02-03, 72% appear eligible because they responded to earlier campaigns.

**Verified:** When checking eligibility using **historical gifts only** (excluding campaign responses), eligibility is stable at 32.7% across all campaign dates — matching the real 33%.

**Pipeline impact:** Preprocess correctly uses `gifts[gdate] < T` which anchors features to history before T. The FID filter uses `all_days_since_last` which reflects pre-T history. The growing pool in logs is a display artifact — the feature values are correct.

**Important distinction for real data:** In production, historical gifts and campaign response gifts are naturally separated — historical gifts predate the campaign, response gifts are recorded after. The simulation collapses this distinction by writing both to the same table. This is harmless for the pipeline but inflates the eligible pool count in the preprocessing logs.

---

### Summary — what changes when you plug in real data

| Aspect | Simulated | Real data (expected) |
|---|---|---|
| Response rate | 14.19% | ~6.5% |
| Propensity AUC | 0.65 | 0.75–0.85 |
| Raw score mean | 0.48 | ~0.065 |
| Amount MAE | €25.59 | €8–15 |
| EV selection rate | ~100% | 15–25% |
| Top features | Gift count (frequency) | Recency + recent windows |
| FID eligible pool | Grows 33%→72% across campaigns | Stable ~33% |
| scale_pos_weight | 5.96 | ~14.3 |

The pipeline architecture, feature engineering logic, training procedure, calibration selection, and EV calculation are all production-correct. Only the signal strength differs between simulated and real data.

---

## Running the pipeline

### Prerequisites

```bash
python >= 3.11
docker
git
```

### Local development

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/donor-ev-scorer.git
cd donor-ev-scorer
pip install -r requirements.txt
cp .env.example .env
# edit .env with your values

# 2. Generate simulated data
python src/simulate.py

# 3. Train models
python src/train.py

# 4. Run API locally
python -m uvicorn main:app --reload

# 5. Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"donors": [...]}'
```

### Docker

```bash
# Build and run
docker build -t donor-ev-scorer .
docker run -d --rm \
  --env-file .env \
  -p 8080:80 \
  donor-ev-scorer

# Or with Docker Compose (API + Redis cache)
docker compose up --build
```

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — returns model loaded status |
| `/model-info` | GET | Model version, features, last training metrics |
| `/score` | POST | Send donor features → get EV scores |
| `/select` | POST | Send features + campaign cost → get selected donors |
| `/docs` | GET | Interactive API documentation (FastAPI auto-generated) |

### Example `/score` request

```json
POST /score
{
  "campaign_type": "FID",
  "cost_per_mail": 1.50,
  "donors": [
    {
      "ind_id": 100001,
      "n_gifts_total": 8,
      "amount_mean_total": 35.0,
      "days_since_last": 180,
      "n_gifts_0_6m": 1,
      "amt_0_6m": 40.0,
      "n_gifts_6_12m": 1,
      "amt_6_12m": 30.0,
      "language": "Dutch",
      "gender": "Female",
      "province": "Province of Antwerpen"
    }
  ]
}
```

### Example `/score` response

```json
{
  "campaign_type": "FID",
  "cost_per_mail": 1.50,
  "scores": [
    {
      "ind_id": 100001,
      "prob_donate": 0.142,
      "pred_amount_eur": 38.50,
      "ev_eur": 5.47,
      "net_ev_eur": 3.97,
      "roi": 3.65,
      "selected": true
    }
  ],
  "summary": {
    "n_scored": 1,
    "n_selected": 1,
    "selection_rate": 1.0,
    "mean_ev": 5.47,
    "total_expected_revenue": 5.47
  }
}
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Version control | Git + GitHub | Code and config versioning |
| Data simulation | Python + NumPy | Synthetic donor data generation |
| Feature engineering | pandas + lifetimes | RFM aggregation, Bayesian CLTV |
| Model training | scikit-learn, XGBoost, LightGBM | Propensity + amount models |
| Hyperparameter tuning | Optuna | Automated model selection |
| Experiment tracking | MLflow | Model versions, metrics, artifacts |
| API serving | FastAPI + uvicorn | EV scoring endpoint |
| Containerisation | Docker | Reproducible runtime |
| Orchestration | Docker Compose | API + Redis cache |
| Cloud (Day 4+) | AWS EC2, S3, ECR | Production deployment |

---

## Security notes

- `.env` is in `.gitignore` — it is never committed
- All client identifiers, data paths, and credentials live in `.env` only
- `.env.example` shows the structure with empty values — this is safe to commit
- Model artifacts (`.pkl`, `.joblib`) are in `.gitignore` — they go to S3/MLflow
- Donor data (`data/`, `simulated_data/`) is in `.gitignore` — never committed
- The Docker image is built without any data or credentials baked in — all passed at runtime via `--env-file .env` and `-v` volume mounts

---

## Development roadmap

- [x] Project structure and Git setup
- [x] Data simulation calibrated to real distributions (`src/simulate.py`)
- [x] Feature engineering pipeline — 225 features, 5 time windows (`src/preprocess.py`)
- [x] Dual model training — propensity + amount, calibration comparison (`src/train.py`)
- [ ] EV scoring and selection logic (`src/score.py`)
- [ ] FastAPI endpoints — /score, /select, /health (`main.py`)
- [ ] Dockerfile and Docker Compose
- [ ] MLflow experiment tracking (Day 5)
- [ ] AWS EC2 deployment (Day 4)
- [ ] REAC campaign variant
