# ──────────────────────────────────────────────────────────────
# FraudShield — Dockerfile multi-cible
# Usage :
#   docker build --target api -t fraudshield-api .
#   docker build --target dash -t fraudshield-dash .
# ──────────────────────────────────────────────────────────────

# ── Base commune ──────────────────────────────────────────────
FROM python:3.12-slim AS base

LABEL maintainer="KARA — BIIS"
LABEL description="FraudShield · Détection de fraude bancaire"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source (structure plate)
COPY features.py explain.py database.py api.py app.py ./

# Artefacts du modèle
COPY artifacts/lgbm_model.joblib artifacts/
COPY artifacts/optimal_threshold.joblib artifacts/
COPY artifacts/merchant_encoder.joblib artifacts/
COPY artifacts/metrics.json artifacts/
COPY artifacts/cost_analysis.json artifacts/
COPY artifacts/shap_global_importance.csv artifacts/
COPY artifacts/X_test.parquet artifacts/
COPY artifacts/y_test.parquet artifacts/

# CSS dashboard
COPY assets/ assets/

# ── Cible API (Hugging Face Spaces) ──────────────────────────
FROM base AS api
ENV PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 2 --timeout-keep-alive 120"]

# ── Cible Dashboard (Render) ─────────────────────────────────
FROM base AS dash
ENV PORT=8050
EXPOSE 8050
CMD ["sh", "-c", "gunicorn app:server --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
