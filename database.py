"""
database.py — Couche d'accès Supabase · FraudShield
Auth RBAC, journal transactions, décisions analystes, stats admin.

Variables d'environnement :
  SUPABASE_URL         → URL du projet Supabase
  SUPABASE_KEY         → Clé anon (publique, auth côté client)
  SUPABASE_SERVICE_KEY → Clé service_role (opérations serveur, bypass RLS)

Le code fonctionne sans Supabase : is_configured() retourne False,
et le dashboard/API passent en mode dégradé (sans historique ni auth).
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("fraudshield.database")

# ══════════════════════════════════════════════════════════════
# CLIENT SUPABASE (lazy init)
# ══════════════════════════════════════════════════════════════

_client = None
_service_client = None


def is_configured() -> bool:
    """Vérifie si les variables Supabase sont présentes."""
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


def _get_client():
    """Client avec clé anon (opérations authentifiées, soumis au RLS)."""
    global _client
    if _client is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL et SUPABASE_KEY requises")
        _client = create_client(url, key)
    return _client


def _get_service_client():
    """Client avec clé service_role (bypass RLS, opérations serveur)."""
    global _service_client
    if _service_client is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL et SUPABASE_SERVICE_KEY requises")
        _service_client = create_client(url, key)
    return _service_client


# ══════════════════════════════════════════════════════════════
# AUTHENTIFICATION
# ══════════════════════════════════════════════════════════════

def sign_in(email: str, password: str) -> dict:
    """
    Connecte un utilisateur.
    Retourne {user_id, email, full_name, role, access_token}.
    """
    client = _get_client()
    response = client.auth.sign_in_with_password({"email": email, "password": password})

    user_id = response.user.id
    profile = (
        client.table("profiles")
        .select("role, full_name, is_active")
        .eq("id", user_id)
        .single()
        .execute()
    )

    if not profile.data.get("is_active", True):
        raise PermissionError("Compte désactivé. Contactez l'administrateur.")

    logger.info("Connexion : %s (rôle=%s)", email, profile.data.get("role"))
    return {
        "user_id": user_id,
        "email": email,
        "full_name": profile.data.get("full_name", ""),
        "role": profile.data.get("role", "analyst"),
        "access_token": response.session.access_token,
    }


def sign_out():
    """Déconnecte l'utilisateur courant."""
    try:
        _get_client().auth.sign_out()
    except Exception:
        pass


def check_admin(user_id: str) -> bool:
    """Vérifie si l'utilisateur est admin."""
    try:
        result = (
            _get_service_client().table("profiles")
            .select("role").eq("id", user_id).single().execute()
        )
        return result.data.get("role") == "admin"
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# ENREGISTREMENT DES TRANSACTIONS SCORÉES
# ══════════════════════════════════════════════════════════════

def log_transaction(
    raw_data: dict,
    score_result: dict,
    summary: dict,
    source: str = "manual",
    user_id: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
) -> Optional[int]:
    """Enregistre une transaction scorée. Retourne l'id ou None en cas d'erreur."""
    try:
        client = _get_service_client()
        record = {
            "amount": raw_data.get("amount"),
            "hour": raw_data.get("hour"),
            "merchant_category": raw_data.get("merchant_category"),
            "distance_from_home_km": raw_data.get("distance_from_home_km"),
            "card_present": raw_data.get("card_present"),
            "account_age_days": raw_data.get("account_age_days"),
            "txn_last_hour": raw_data.get("txn_last_hour"),
            "avg_amount_30d": raw_data.get("avg_amount_30d"),
            "foreign_country": raw_data.get("foreign_country"),
            "num_declines_today": raw_data.get("num_declines_today"),
            "fraud_probability": score_result["probability"],
            "fraud_prediction": score_result["prediction"],
            "risk_level": summary.get("risk_level", "unknown").upper(),
            "threshold_used": score_result["threshold"],
            "summary_headline": summary.get("headline", ""),
            "summary_analysis": summary.get("analysis", ""),
            "summary_mode": summary.get("mode", "rule-based"),
            "top_reasons": json.dumps(score_result.get("top_reasons", []), ensure_ascii=False),
            "source": source,
            "scored_by": user_id,
            "processing_time_ms": processing_time_ms,
        }
        result = client.table("transactions_scored").insert(record).execute()
        return result.data[0]["id"]
    except Exception as e:
        logger.error("Erreur log_transaction : %s", e)
        return None


# ══════════════════════════════════════════════════════════════
# DÉCISIONS ANALYSTES (FEEDBACK)
# ══════════════════════════════════════════════════════════════

VALID_DECISIONS = {"confirmed_fraud", "false_positive", "escalated", "pending"}


def submit_decision(
    transaction_id: int,
    decision: str,
    user_id: str,
    comment: Optional[str] = None,
) -> Optional[int]:
    """Enregistre la décision d'un analyste. Retourne l'id ou None."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Décision invalide : {decision}. Valeurs : {VALID_DECISIONS}")
    try:
        client = _get_service_client()
        result = client.table("alert_decisions").insert({
            "transaction_id": transaction_id,
            "decision": decision,
            "decided_by": user_id,
            "comment": comment,
        }).execute()
        return result.data[0]["id"]
    except Exception as e:
        logger.error("Erreur submit_decision : %s", e)
        return None


# ══════════════════════════════════════════════════════════════
# REQUÊTES DE LECTURE
# ══════════════════════════════════════════════════════════════

def get_recent_transactions(limit: int = 50, fraud_only: bool = False) -> list[dict]:
    """Récupère les transactions récentes avec leurs décisions."""
    client = _get_service_client()
    query = (
        client.table("transactions_scored")
        .select("*, alert_decisions(decision, comment, decided_at)")
        .order("scored_at", desc=True)
        .limit(limit)
    )
    if fraud_only:
        query = query.eq("fraud_prediction", 1)
    return query.execute().data


def get_transaction_detail(transaction_id: int) -> Optional[dict]:
    """Récupère une transaction avec ses décisions."""
    try:
        result = (
            _get_service_client().table("transactions_scored")
            .select("*, alert_decisions(decision, comment, decided_by, decided_at)")
            .eq("id", transaction_id)
            .single()
            .execute()
        )
        return result.data
    except Exception:
        return None


def get_pending_alerts(limit: int = 50) -> list[dict]:
    """Alertes fraude sans décision."""
    client = _get_service_client()
    alerts = (
        client.table("transactions_scored")
        .select("*")
        .eq("fraud_prediction", 1)
        .order("scored_at", desc=True)
        .limit(limit)
        .execute()
    ).data

    decided_ids = {
        r["transaction_id"]
        for r in client.table("alert_decisions").select("transaction_id").execute().data
    }
    return [t for t in alerts if t["id"] not in decided_ids]


# ══════════════════════════════════════════════════════════════
# STATISTIQUES ADMIN
# ══════════════════════════════════════════════════════════════

def get_admin_stats() -> dict:
    """Statistiques globales (vue admin_stats)."""
    try:
        result = _get_service_client().table("admin_stats").select("*").execute()
        return result.data[0] if result.data else {}
    except Exception as e:
        logger.error("Erreur get_admin_stats : %s", e)
        return {}


# ══════════════════════════════════════════════════════════════
# GESTION UTILISATEURS (admin)
# ══════════════════════════════════════════════════════════════

def list_users() -> list[dict]:
    """Liste tous les utilisateurs."""
    return (
        _get_service_client().table("profiles")
        .select("id, email, full_name, role, is_active, created_at")
        .order("created_at", desc=True)
        .execute()
    ).data


def update_user_role(user_id: str, new_role: str) -> Optional[dict]:
    """Change le rôle d'un utilisateur."""
    if new_role not in ("analyst", "admin"):
        raise ValueError("Rôle invalide")
    result = (
        _get_service_client().table("profiles")
        .update({"role": new_role, "updated_at": datetime.now(timezone.utc).isoformat()})
        .eq("id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None
