"""
api.py — API temps réel · FraudShield
Endpoints :
  POST /predict         → score + SHAP + résumé copilote pour 1 transaction
  POST /predict/batch   → scoring par lot (JSON)
  POST /predict/csv     → scoring d'un fichier CSV uploadé
  GET  /health          → healthcheck
  GET  /model/info      → métriques, seuil, features
  GET  /model/shap      → importance SHAP globale
  GET  /history         → historique (Supabase)
  GET  /alerts/pending  → alertes en attente
  POST /alerts/decide   → feedback analyste

Déploiement : Hugging Face Spaces (Docker) ou local
"""
from dotenv import load_dotenv
load_dotenv()

import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import (
    ARTIFACT_DIR,
    FEATURE_LABELS_FR,
    FEATURE_NAMES,
    MERCHANT_CATEGORIES_ORDERED,
    RAW_FEATURES,
    engineer_features,
    load_merchant_encoder,
)
from explain import explain_single, generate_summary
from database import (
    get_admin_stats,
    get_pending_alerts,
    get_recent_transactions,
    is_configured as db_configured,
    log_transaction,
    submit_decision,
)

# ── État global ────────────────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement des artefacts au démarrage."""
    state["model"] = joblib.load(ARTIFACT_DIR / "lgbm_model.joblib")
    state["threshold"] = joblib.load(ARTIFACT_DIR / "optimal_threshold.joblib")
    state["encoder"] = load_merchant_encoder()
    with open(ARTIFACT_DIR / "metrics.json") as f:
        state["metrics"] = json.load(f)
    with open(ARTIFACT_DIR / "cost_analysis.json") as f:
        state["cost"] = json.load(f)
    state["start_time"] = time.time()
    state["request_count"] = 0
    state["db_enabled"] = db_configured()
    yield
    state.clear()


app = FastAPI(
    title="FraudShield API",
    description="Détection de fraude bancaire temps réel — LightGBM + SHAP + Copilote IA",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════════════════════
# SCHÉMAS
# ══════════════════════════════════════════════════════════════

class TransactionInput(BaseModel):
    amount: float = Field(..., ge=0, description="Montant en FCFA")
    hour: int = Field(..., ge=0, le=23)
    merchant_category: str
    distance_from_home_km: float = Field(..., ge=0)
    card_present: int = Field(..., ge=0, le=1)
    account_age_days: int = Field(..., ge=0)
    txn_last_hour: int = Field(..., ge=0)
    avg_amount_30d: float = Field(..., ge=0)
    foreign_country: int = Field(..., ge=0, le=1)
    num_declines_today: int = Field(..., ge=0)

    model_config = {"json_schema_extra": {"examples": [{
        "amount": 450.0, "hour": 2, "merchant_category": "electronics",
        "distance_from_home_km": 120.0, "card_present": 0,
        "account_age_days": 90, "txn_last_hour": 3,
        "avg_amount_30d": 55.0, "foreign_country": 1, "num_declines_today": 2,
    }]}}


class BatchInput(BaseModel):
    transactions: list[TransactionInput]


class DecisionInput(BaseModel):
    transaction_id: int
    decision: str = Field(..., description="confirmed_fraud | false_positive | escalated | pending")
    user_id: str
    comment: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════

def _risk_level(score: float) -> str:
    if score >= state["threshold"]:
        return "ÉLEVÉ"
    if score >= state["threshold"] * 0.6:
        return "MOYEN"
    return "FAIBLE"


def _score_one(txn_dict: dict) -> dict:
    """Score une transaction et retourne le résultat complet."""
    t0 = time.time()

    if txn_dict["merchant_category"] not in MERCHANT_CATEGORIES_ORDERED:
        raise HTTPException(422, f"Catégorie inconnue : '{txn_dict['merchant_category']}'")

    raw_df = pd.DataFrame([txn_dict])
    X = engineer_features(raw_df, state["encoder"], is_training=False)
    result = explain_single(X)
    summary = generate_summary(result, mode="auto")
    elapsed = (time.time() - t0) * 1000
    state["request_count"] += 1

    # Log en base (non-bloquant)
    db_id = None
    if state.get("db_enabled"):
        db_id = log_transaction(
            raw_data=txn_dict, score_result=result, summary=summary,
            source="api", processing_time_ms=elapsed,
        )

    return {
        "probability": result["probability"],
        "prediction": result["prediction"],
        "prediction_label": result["prediction_label"],
        "threshold": result["threshold"],
        "risk_level": _risk_level(result["probability"]),
        "top_reasons": result["top_reasons"],
        "shap_values": result["shap_values"],
        "analyst_summary": {k: v for k, v in summary.items() if k != "llm_response_time_ms"},
        "processing_time_ms": round(elapsed, 2),
        "transaction_id": db_id,
    }


# ══════════════════════════════════════════════════════════════
# ENDPOINTS SCORING
# ══════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(txn: TransactionInput):
    """Score une transaction unique avec explicabilité SHAP complète."""
    return _score_one(txn.model_dump())


@app.post("/predict/batch")
async def predict_batch(batch: BatchInput):
    """Score un lot de transactions."""
    t0 = time.time()
    records = [t.model_dump() for t in batch.transactions]
    raw_df = pd.DataFrame(records)

    unknown = set(raw_df["merchant_category"]) - set(MERCHANT_CATEGORIES_ORDERED)
    if unknown:
        raise HTTPException(422, f"Catégories inconnues : {unknown}")

    X = engineer_features(raw_df, state["encoder"], is_training=False)
    proba = state["model"].predict_proba(X)[:, 1]
    preds = (proba >= state["threshold"]).astype(int)

    results = []
    for i in range(len(raw_df)):
        result = explain_single(X.iloc[[i]])
        results.append({
            "index": i,
            "probability": round(float(proba[i]), 4),
            "prediction": int(preds[i]),
            "prediction_label": "FRAUDE" if preds[i] == 1 else "LÉGITIME",
            "risk_level": _risk_level(proba[i]),
            "top_reasons": result["top_reasons"],
        })

    results.sort(key=lambda x: x["probability"], reverse=True)
    fraud_mask = preds == 1
    state["request_count"] += len(records)

    return {
        "total": len(records),
        "fraud_detected": int(fraud_mask.sum()),
        "fraud_rate": round(float(fraud_mask.mean()), 4),
        "total_amount_at_risk": round(float(raw_df.loc[fraud_mask, "amount"].sum()), 2),
        "processing_time_ms": round((time.time() - t0) * 1000, 2),
        "results": results,
    }


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Score un fichier CSV uploadé."""
    t0 = time.time()
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Format CSV requis")

    content = await file.read()
    try:
        raw_df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"Erreur lecture CSV : {e}")

    missing = set(RAW_FEATURES) - set(raw_df.columns)
    if missing:
        raise HTTPException(422, f"Colonnes manquantes : {missing}")

    X = engineer_features(raw_df, state["encoder"], is_training=False)
    proba = state["model"].predict_proba(X)[:, 1]
    preds = (proba >= state["threshold"]).astype(int)

    raw_df["fraud_probability"] = np.round(proba, 4)
    raw_df["fraud_prediction"] = preds
    raw_df["risk_level"] = [_risk_level(p) for p in proba]
    fraud_mask = preds == 1
    state["request_count"] += len(raw_df)

    return {
        "filename": file.filename,
        "total": len(raw_df),
        "fraud_detected": int(fraud_mask.sum()),
        "fraud_rate": round(float(fraud_mask.mean()), 4),
        "total_amount_at_risk": round(float(raw_df.loc[fraud_mask, "amount"].sum()), 2),
        "processing_time_ms": round((time.time() - t0) * 1000, 2),
        "results": raw_df.sort_values("fraud_probability", ascending=False).head(100).to_dict("records"),
    }


# ══════════════════════════════════════════════════════════════
# ENDPOINTS MODÈLE
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    uptime = time.time() - state.get("start_time", time.time())
    return {
        "status": "healthy",
        "model_loaded": "model" in state,
        "threshold": state.get("threshold"),
        "database": "connected" if state.get("db_enabled") else "not configured",
        "uptime_seconds": round(uptime),
        "requests_served": state.get("request_count", 0),
    }


@app.get("/model/info")
async def model_info():
    cost = state["cost"]
    lgb_metrics = state["metrics"].get("lightgbm_tuned", {})
    return {
        "model": "LightGBM (GBDT)",
        "features": FEATURE_NAMES,
        "feature_labels_fr": FEATURE_LABELS_FR,
        "threshold": state["threshold"],
        "metrics": {
            "pr_auc": lgb_metrics.get("pr_auc"),
            "roc_auc": lgb_metrics.get("roc_auc"),
            "precision": cost["at_optimal"]["precision"],
            "recall": cost["at_optimal"]["recall"],
            "f1": cost["at_optimal"]["f1"],
        },
        "cost_model": {"cost_fp_fcfa": cost["cost_fp_fixed_fcfa"], "cost_fn": "montant réel"},
        "merchant_categories": MERCHANT_CATEGORIES_ORDERED,
    }


@app.get("/model/shap")
async def model_shap():
    path = ARTIFACT_DIR / "shap_global_importance.csv"
    if not path.exists():
        raise HTTPException(404, "SHAP global non disponible")
    return {"features": pd.read_csv(path).to_dict("records")}


# ══════════════════════════════════════════════════════════════
# ENDPOINTS BASE DE DONNÉES
# ══════════════════════════════════════════════════════════════

def _require_db():
    if not state.get("db_enabled"):
        raise HTTPException(503, "Base de données non configurée")


@app.get("/history")
async def history(limit: int = 50, fraud_only: bool = False):
    _require_db()
    return get_recent_transactions(limit=limit, fraud_only=fraud_only)


@app.get("/alerts/pending")
async def pending_alerts(limit: int = 50):
    _require_db()
    return get_pending_alerts(limit=limit)


@app.post("/alerts/decide")
async def decide_alert(data: DecisionInput):
    _require_db()
    try:
        did = submit_decision(data.transaction_id, data.decision, data.user_id, data.comment)
        return {"decision_id": did, "status": "ok"}
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/stats")
async def stats():
    _require_db()
    return get_admin_stats()


# ══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
