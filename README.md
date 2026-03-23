# Donor EV Scorer — BelFund(a simulated charity)

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
- **Selections per donor** — real data spans 2011–2026 (15 years, 1,556 action IDs). The simulation spans 2025–2026 (19 campaigns, 57 action IDs). The per-segment selection logic matches; the lifetime count does not.
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
- [x] Data simulation calibrated to real distributions
- [ ] Feature engineering pipeline (`preprocess.py`)
- [ ] Dual model training (`train.py`)
- [ ] EV scoring logic (`score.py`)
- [ ] FastAPI endpoints (`main.py`)
- [ ] Dockerfile and Docker Compose
- [ ] MLflow experiment tracking
- [ ] AWS EC2 deployment
- [ ] REAC campaign variant
