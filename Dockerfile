FROM python:3.11.14-slim

WORKDIR /app

# ── Install dependencies first (cached layer) ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────
COPY src/ ./src/
COPY main.py .
COPY .env.example .

# ── Run full pipeline inside container ────────────────────────────────────
# simulate.py generates data/simulated/ → preprocess builds features →
# train.py trains both models and saves artefacts to models/
# In production: mount real data via -v and skip simulate step
ENV SIM_MODE=true
ENV SIM_N_DONORS=15000
ENV SIM_N_CAMPAIGNS=19
ENV SIM_SEED=42
ENV CAMPAIGN_TYPE=FID
ENV COST_PER_MAIL=1.50
ENV MODEL_VERSION=v1
ENV CLIENT_NAME=BelFund

RUN python src/simulate.py && \
    python src/preprocess.py && \
    python src/train.py

# ── Expose port and start API ─────────────────────────────────────────────
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
