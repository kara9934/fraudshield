"""
features.py — Pipeline de feature engineering · FraudShield
Cœur partagé : entraînement, API temps réel, dashboard.
Une seule source de vérité pour toutes les transformations.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

logger = logging.getLogger("fraudshield.features")

# ── Chemins ────────────────────────────────────────────────────
ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# ── Colonnes brutes attendues (sans is_fraud) ──────────────────
RAW_FEATURES = [
    "amount", "hour", "merchant_category", "distance_from_home_km",
    "card_present", "account_age_days", "txn_last_hour",
    "avg_amount_30d", "foreign_country", "num_declines_today",
]

# ── Catégories marchands (ordre fixe, taux fraude croissant EDA) ─
MERCHANT_CATEGORIES_ORDERED = [
    "gas", "grocery", "restaurant", "health", "retail",
    "atm", "entertainment", "online_service", "electronics", "travel",
]

HIGH_RISK_CATEGORIES = {"travel", "electronics", "online_service"}

# ── Features finales (15) ──────────────────────────────────────
FEATURE_NAMES = [
    "amount", "hour", "distance_from_home_km", "card_present",
    "account_age_days", "txn_last_hour", "avg_amount_30d",
    "foreign_country", "num_declines_today", "merchant_encoded",
    "amount_ratio", "is_night", "log_amount", "log_distance",
    "high_risk_category",
]

FEATURE_LABELS_FR = {
    "amount": "Montant (FCFA)",
    "hour": "Heure",
    "distance_from_home_km": "Distance du domicile (km)",
    "card_present": "Carte présente",
    "account_age_days": "Ancienneté du compte (jours)",
    "txn_last_hour": "Transactions dernière heure",
    "avg_amount_30d": "Montant moyen 30j",
    "foreign_country": "Transaction à l'étranger",
    "num_declines_today": "Refus aujourd'hui",
    "merchant_encoded": "Catégorie marchand (encodée)",
    "amount_ratio": "Ratio montant / moy. 30j",
    "is_night": "Transaction de nuit",
    "log_amount": "Log(montant)",
    "log_distance": "Log(distance)",
    "high_risk_category": "Catégorie à haut risque",
}


# ── Encodeur merchant_category ─────────────────────────────────

def build_merchant_encoder(df: pd.DataFrame) -> dict:
    """Construit l'encodeur ordinal (taux fraude croissant). Fit sur train."""
    fraud_rate = df.groupby("merchant_category")["is_fraud"].mean().sort_values()
    encoder = {cat: idx for idx, cat in enumerate(fraud_rate.index)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, ARTIFACT_DIR / "merchant_encoder.joblib")
    logger.info("Encodeur merchant sauvegardé (%d catégories)", len(encoder))
    return encoder


def load_merchant_encoder() -> dict:
    """Charge l'encodeur depuis les artefacts."""
    path = ARTIFACT_DIR / "merchant_encoder.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Encodeur introuvable : {path}")
    return joblib.load(path)


# ── Feature engineering ────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    merchant_encoder: dict | None = None,
    is_training: bool = False,
) -> pd.DataFrame:
    """
    Transforme les données brutes en features pour le modèle.

    Paramètres
    ----------
    df : DataFrame avec au minimum les colonnes RAW_FEATURES.
    merchant_encoder : dict {catégorie: code ordinal}. Si None, charge depuis artifacts/.
    is_training : si True, conserve la colonne is_fraud.

    Retourne
    --------
    DataFrame de 15 features, prêt pour le modèle.
    """
    df = df.copy()

    # Validation
    missing = set(RAW_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    # Encodage merchant_category
    if merchant_encoder is None:
        merchant_encoder = load_merchant_encoder()
    df["merchant_encoded"] = (
        df["merchant_category"].map(merchant_encoder).fillna(-1).astype(int)
    )

    # Features dérivées
    df["amount_ratio"] = df["amount"] / df["avg_amount_30d"].clip(lower=1.0)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    df["log_distance"] = np.log1p(df["distance_from_home_km"])
    df["high_risk_category"] = df["merchant_category"].isin(HIGH_RISK_CATEGORIES).astype(int)

    # Sélection des colonnes de sortie
    cols = FEATURE_NAMES.copy()
    if is_training and "is_fraud" in df.columns:
        cols.append("is_fraud")

    return df[cols]
