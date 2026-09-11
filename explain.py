"""
explain.py — Explicabilité SHAP + Copilote IA analyste · FraudShield
Fusionne les anciens explain.py et copilot.py.

Fournit :
  - explain_single(X_row) → dict SHAP pour une transaction
  - explain_batch(X_df)   → (shap_values, expected_value)
  - get_global_importance(X_df) → DataFrame importance SHAP

  - generate_summary(explain_result, mode) → résumé analyste
    Modes : "rules" (déterministe, hors-ligne) | "llm" (Groq/Llama 3.3 70B) | "auto"
"""

import logging
import os
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

from features import ARTIFACT_DIR, FEATURE_LABELS_FR

logger = logging.getLogger("fraudshield.explain")

# ══════════════════════════════════════════════════════════════
# CHARGEMENT LAZY (une seule fois en mémoire)
# ══════════════════════════════════════════════════════════════

_model = None
_explainer = None
_threshold = None


def _load():
    global _model, _explainer, _threshold
    if _model is None:
        _model = joblib.load(ARTIFACT_DIR / "lgbm_model.joblib")
        _explainer = shap.TreeExplainer(_model)
        _threshold = joblib.load(ARTIFACT_DIR / "optimal_threshold.joblib")
        logger.info("Modèle et explainer SHAP chargés (seuil=%.4f)", _threshold)
    return _model, _explainer, _threshold


# ══════════════════════════════════════════════════════════════
# EXPLICATION D'UNE TRANSACTION
# ══════════════════════════════════════════════════════════════

def explain_single(X_row: pd.DataFrame) -> dict:
    """
    Explique une transaction unique.

    Paramètres
    ----------
    X_row : DataFrame 1 ligne × 15 features

    Retourne
    --------
    dict avec probability, prediction, prediction_label, threshold,
    base_value, shap_values (list triée), top_reasons (3 textes)
    """
    model, explainer, threshold = _load()

    proba = model.predict_proba(X_row)[:, 1][0]
    pred = int(proba >= threshold)

    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1][0]
    else:
        sv = sv[0]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    features = X_row.columns.tolist()
    values = X_row.iloc[0].tolist()

    details = sorted(
        [
            {
                "feature": feat,
                "value": round(float(val), 4),
                "shap_value": round(float(shap_v), 4),
                "label_fr": FEATURE_LABELS_FR.get(feat, feat),
            }
            for feat, val, shap_v in zip(features, values, sv)
        ],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    top_reasons = []
    for d in details[:3]:
        direction = "↑ augmente" if d["shap_value"] > 0 else "↓ diminue"
        top_reasons.append(f"{d['label_fr']} = {d['value']} ({direction} le risque)")

    return {
        "probability": round(float(proba), 4),
        "prediction": pred,
        "prediction_label": "FRAUDE" if pred == 1 else "LÉGITIME",
        "threshold": round(float(threshold), 4),
        "base_value": round(float(base_val), 4),
        "shap_values": details,
        "top_reasons": top_reasons,
    }


# ══════════════════════════════════════════════════════════════
# EXPLICATION PAR LOT
# ══════════════════════════════════════════════════════════════

def explain_batch(X_df: pd.DataFrame) -> tuple:
    """Retourne (shap_values_array, expected_value) pour un lot."""
    _, explainer, _ = _load()
    sv = explainer.shap_values(X_df)
    if isinstance(sv, list):
        sv = sv[1]
    bv = explainer.expected_value
    if isinstance(bv, (list, np.ndarray)):
        bv = bv[1]
    return sv, bv


def get_global_importance(X_df: pd.DataFrame) -> pd.DataFrame:
    """Importance SHAP moyenne (|SHAP|), triée décroissante."""
    sv, _ = explain_batch(X_df)
    imp = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "label_fr": [FEATURE_LABELS_FR.get(f, f) for f in X_df.columns],
    }).sort_values("mean_abs_shap", ascending=False)
    imp["pct"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum() * 100
    return imp.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# COPILOTE IA — CONSTANTES
# ══════════════════════════════════════════════════════════════

RISK_THRESHOLDS = {"high": 0.83, "medium": 0.50}

FEATURE_NARRATIVES = {
    "amount_ratio": {
        "high_up": "le montant représente {val:.1f}× la dépense habituelle de ce client",
        "normal": "le montant est dans la fourchette habituelle du client",
    },
    "distance_from_home_km": {
        "high_up": "la transaction a lieu à {val:.0f} km du domicile",
        "normal": "la transaction est proche du domicile ({val:.1f} km)",
    },
    "hour": {
        "night": "effectuée à {val:.0f}h (créneau nocturne à risque)",
        "normal": "effectuée à {val:.0f}h (horaire courant)",
    },
    "account_age_days": {
        "young": "le compte n'a que {val:.0f} jours d'ancienneté",
        "normal": "le compte est établi ({val:.0f} jours)",
    },
    "txn_last_hour": {
        "high_up": "{val:.0f} transactions dans la dernière heure (vélocité élevée)",
        "normal": "activité transactionnelle normale",
    },
    "num_declines_today": {
        "high_up": "{val:.0f} refus déjà enregistrés aujourd'hui",
        "normal": "aucun refus antérieur aujourd'hui",
    },
    "card_present": {"absent": "transaction sans carte physique (CNP)", "present": "carte physiquement présente"},
    "foreign_country": {"foreign": "transaction depuis l'étranger", "local": "transaction locale"},
    "merchant_encoded": {"high_risk": "catégorie marchand à risque élevé", "normal": "catégorie marchand standard"},
    "is_night": {"night": "transaction de nuit", "day": "transaction de jour"},
    "high_risk_category": {"yes": "catégorie à haut risque (travel/electronics/online)", "no": "catégorie standard"},
}

RECOMMENDATIONS = {
    "high": [
        "Bloquer la transaction et déclencher une vérification immédiate",
        "Contacter le porteur de carte pour confirmation",
        "Documenter l'incident dans le registre anti-fraude",
    ],
    "medium": [
        "Placer la transaction en file d'attente pour revue manuelle",
        "Vérifier l'historique récent du client pour d'autres anomalies",
        "Surveiller les prochaines transactions de ce compte",
    ],
    "low": ["Aucune action requise — transaction dans les paramètres normaux"],
}


# ══════════════════════════════════════════════════════════════
# COPILOTE — MODE RULE-BASED
# ══════════════════════════════════════════════════════════════

def _get_risk_level(score: float) -> str:
    if score >= RISK_THRESHOLDS["high"]:
        return "high"
    if score >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _narrate_feature(feature: str, value: float, shap_value: float) -> Optional[str]:
    """Génère une phrase narrative pour une feature."""
    narr = FEATURE_NARRATIVES.get(feature)
    if not narr:
        return None

    if feature == "amount_ratio":
        return narr["high_up"].format(val=value) if value > 2.0 and shap_value > 0 else narr["normal"].format(val=value)
    if feature == "distance_from_home_km":
        return narr["high_up"].format(val=value) if value > 30 and shap_value > 0 else narr["normal"].format(val=value)
    if feature == "hour":
        return narr["night"].format(val=value) if (value <= 5 or value >= 23) and shap_value > 0 else narr["normal"].format(val=value)
    if feature == "account_age_days":
        return narr["young"].format(val=value) if value < 180 and shap_value > 0 else narr["normal"].format(val=value)
    if feature == "txn_last_hour":
        return narr["high_up"].format(val=value) if value >= 2 and shap_value > 0 else narr["normal"]
    if feature == "num_declines_today":
        return narr["high_up"].format(val=value) if value >= 1 and shap_value > 0 else narr["normal"]
    if feature == "card_present":
        return narr["absent"] if value == 0 else narr["present"]
    if feature == "foreign_country":
        return narr["foreign"] if value == 1 else narr["local"]
    if feature == "high_risk_category":
        return narr["yes"] if value == 1 else narr["no"]
    if feature == "is_night":
        return narr["night"] if value == 1 else narr["day"]
    return None


def generate_summary_rules(explain_result: dict) -> dict:
    """Résumé analyste déterministe (hors-ligne, gratuit)."""
    score = explain_result["probability"]
    shap_details = explain_result["shap_values"]
    risk = _get_risk_level(score)

    # Headline
    headlines = {
        "high": f"Transaction à haut risque (score {score:.1%}). Décision recommandée : BLOQUER.",
        "medium": f"Transaction suspecte (score {score:.1%}). Revue manuelle recommandée.",
        "low": f"Transaction légitime (score {score:.1%}). Aucune anomalie détectée.",
    }

    # Facteurs narratifs
    factors = []
    for sv in shap_details[:6]:
        narr = _narrate_feature(sv["feature"], sv["value"], sv["shap_value"])
        if narr and narr not in factors:
            factors.append(narr)
        if len(factors) >= 4:
            break

    # Analyse
    if risk == "high":
        aggravating = [f for f, sv in zip(factors, shap_details) if sv["shap_value"] > 0]
        analysis = (
            f"Cette transaction présente un profil de fraude caractérisé. "
            f"Les principaux signaux d'alerte sont : {'; '.join(aggravating[:3])}. "
            f"Le score de {score:.1%} dépasse largement le seuil opérationnel "
            f"({RISK_THRESHOLDS['high']:.0%})."
        )
    elif risk == "medium":
        analysis = (
            f"Cette transaction présente des éléments inhabituels sans atteindre "
            f"le seuil de fraude avérée. Points d'attention : {'; '.join(factors[:3])}. "
            f"Une vérification complémentaire est conseillée."
        )
    else:
        analysis = (
            f"Cette transaction s'inscrit dans le comportement habituel du client. "
            f"Score de risque très faible ({score:.1%})."
        )

    return {
        "risk_level": risk,
        "headline": headlines[risk],
        "analysis": analysis,
        "factors": factors,
        "recommendations": RECOMMENDATIONS[risk],
        "mode": "rule-based",
    }


# ══════════════════════════════════════════════════════════════
# COPILOTE — MODE LLM (Groq / Llama 3.3 70B gratuit)
# ══════════════════════════════════════════════════════════════

GROQ_MODEL = "openai/gpt-oss-120b"


def generate_summary_llm(explain_result: dict, api_key: Optional[str] = None) -> dict:
    """Résumé via Groq (Llama 3.3 70B). Fallback sur rules si indisponible."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return {**generate_summary_rules(explain_result), "mode": "rule-based (GROQ_API_KEY absente)"}

    try:
        from groq import Groq
    except ImportError:
        return {**generate_summary_rules(explain_result), "mode": "rule-based (groq non installé)"}

    score = explain_result["probability"]
    risk = _get_risk_level(score)
    top_shap = explain_result["shap_values"][:6]

    shap_text = "\n".join([
        f"  - {sv['label_fr']} = {sv['value']} (impact SHAP : {sv['shap_value']:+.4f})"
        for sv in top_shap
    ])

    prompt = f"""Tu es un analyste anti-fraude senior dans une banque en zone UEMOA.
Rédige un résumé opérationnel en français pour un analyste junior.

RÉSULTAT DU MODÈLE :
- Score de fraude : {score:.4f} ({score:.1%})
- Décision automatique : {explain_result['prediction_label']}
- Niveau de risque : {risk.upper()}
- Seuil opérationnel : {RISK_THRESHOLDS['high']:.0%}

FACTEURS EXPLICATIFS (SHAP) :
{shap_text}

CONSIGNES :
1. Verdict clair en UNE phrase.
2. 2-3 phrases sur les facteurs importants, langage métier bancaire.
3. Recommandation d'action concrète.
4. Factuel — l'humain décide, tu informes.
5. Maximum 150 mots, pas de markdown."""

    try:
        t0 = time.time()
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Assistant analyste anti-fraude bancaire UEMOA. Français uniquement."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        elapsed_ms = round((time.time() - t0) * 1000)
        text = response.choices[0].message.content
        if not text or not text.strip():
            return {**generate_summary_rules(explain_result),
                    "mode": "rule-based (réponse LLM vide)"}
        text = text.strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        return {
            "risk_level": risk,
            "headline": lines[0].strip("*").strip() if lines else "Résumé indisponible",
            "analysis": "\n".join(lines[1:]).strip() if len(lines) > 1 else text,
            "factors": [sv["label_fr"] + f" = {sv['value']}" for sv in top_shap[:4]],
            "recommendations": RECOMMENDATIONS[risk],
            "mode": f"llm (Groq/{GROQ_MODEL})",
            "llm_response_time_ms": elapsed_ms,
        }
    except Exception as e:
        logger.warning("Groq indisponible (%s), fallback rule-based", str(e)[:80])
        return {**generate_summary_rules(explain_result),
                "mode": f"rule-based (erreur Groq : {str(e)[:60]})"}


# ══════════════════════════════════════════════════════════════
# INTERFACE UNIFIÉE
# ══════════════════════════════════════════════════════════════

def generate_summary(
    explain_result: dict,
    mode: str = "auto",
    api_key: Optional[str] = None,
) -> dict:
    """
    Interface unifiée du copilote.

    mode : "rules" | "llm" | "auto"
      - rules : toujours déterministe
      - llm : tente Groq, fallback sur rules
      - auto : LLM si clé disponible, sinon rules
    """
    if mode == "rules":
        return generate_summary_rules(explain_result)
    if mode == "llm":
        return generate_summary_llm(explain_result, api_key)
    # auto
    key = api_key or os.environ.get("GROQ_API_KEY")
    if key:
        return generate_summary_llm(explain_result, key)
    return generate_summary_rules(explain_result)
