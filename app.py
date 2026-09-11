"""
app.py — Dashboard opérationnel · FraudShield
Optimisé pour Render free tier (512 MB RAM, 0.1 CPU).
Chargement lazy : modèle + SHAP chargés au premier clic, pas au démarrage.
"""

from dotenv import load_dotenv
load_dotenv()

import base64
import io
import json
import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dash_table, dcc, html, no_update

from features import (
    FEATURE_LABELS_FR,
    FEATURE_NAMES,
    MERCHANT_CATEGORIES_ORDERED,
    RAW_FEATURES,
    engineer_features,
    load_merchant_encoder,
)
from database import is_configured as db_configured

# ── Chargement minimal au démarrage (pas de modèle, pas de SHAP) ──
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
merchant_encoder = load_merchant_encoder()

with open(ARTIFACT_DIR / "cost_analysis.json", encoding="utf-8") as f:
    cost_info = json.load(f)
with open(ARTIFACT_DIR / "metrics.json", encoding="utf-8") as f:
    metrics_info = json.load(f)

DB_ENABLED = db_configured()

# ── Cache lazy pour objets lourds ──────────────────────────────
_cache = {}

def _get_model():
    """Charge le modèle LightGBM à la demande."""
    if "model" not in _cache:
        import joblib
        _cache["model"] = joblib.load(ARTIFACT_DIR / "lgbm_model.joblib")
    return _cache["model"]

def _get_threshold():
    """Charge le seuil à la demande."""
    if "threshold" not in _cache:
        import joblib
        _cache["threshold"] = joblib.load(ARTIFACT_DIR / "optimal_threshold.joblib")
    return _cache["threshold"]

def _get_test_data():
    """Charge X_test et y_test à la demande (échantillon réduit)."""
    if "X_test" not in _cache:
        _X = pd.read_parquet(ARTIFACT_DIR / "X_test.parquet")
        _y = pd.read_parquet(ARTIFACT_DIR / "y_test.parquet")["is_fraud"]
        idx = _X.sample(n=min(500, len(_X)), random_state=42).index
        _cache["X_test"] = _X.loc[idx].reset_index(drop=True)
        _cache["y_test"] = _y.loc[idx].reset_index(drop=True)
        del _X, _y
    return _cache["X_test"], _cache["y_test"]

def _explain_single(X_row):
    """Wrapper lazy pour explain_single."""
    from explain import explain_single
    return explain_single(X_row)

def _generate_summary(result, mode="auto"):
    """Wrapper lazy pour generate_summary."""
    from explain import generate_summary
    return generate_summary(result, mode=mode)


# ── App ────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
    ],
    suppress_callback_exceptions=True,
    title="FraudShield · Détection de fraude",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # pour Gunicorn


# ══════════════════════════════════════════════════════════════
# COMPOSANTS RÉUTILISABLES
# ══════════════════════════════════════════════════════════════

def score_badge(score):
    thr = _get_threshold()
    if score >= thr:
        return dbc.Badge(f"FRAUDE ({score:.1%})", color="danger", className="fs-6 px-3 py-2")
    if score >= thr * 0.6:
        return dbc.Badge(f"SUSPECT ({score:.1%})", color="warning", className="fs-6 px-3 py-2")
    return dbc.Badge(f"LÉGITIME ({score:.1%})", color="success", className="fs-6 px-3 py-2")


def shap_waterfall_fig(shap_details, title=""):
    top = shap_details[:8]
    labels = [d["label_fr"] for d in top][::-1]
    values = [d["shap_value"] for d in top][::-1]
    colors = ["#E53935" if v > 0 else "#1E88E5" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in values], textposition="auto",
    ))
    fig.update_layout(
        title=dict(text=title, font_size=13),
        xaxis_title="Impact SHAP sur le score", height=320,
        margin=dict(l=10, r=10, t=40, b=30), plot_bgcolor="white",
    )
    fig.add_vline(x=0, line_color="gray", line_width=1)
    return fig


def gauge_fig(score):
    thr = _get_threshold()
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#E53935" if score >= thr else "#43A047"},
            "steps": [
                {"range": [0, thr * 60], "color": "#E8F5E9"},
                {"range": [thr * 60, thr * 100], "color": "#FFF3E0"},
                {"range": [thr * 100, 100], "color": "#FFEBEE"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": thr * 100},
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def copilot_panel(result):
    summary = _generate_summary(result, mode="auto")
    color_map = {"high": "danger", "medium": "warning", "low": "success"}
    color = color_map.get(summary["risk_level"], "secondary")
    return dbc.Card([
        dbc.CardHeader(html.Div([
            html.I(className="bi bi-robot me-2"),
            html.Span("Copilote IA", className="fw-bold"),
            dbc.Badge(summary["mode"], color="light", text_color="dark", className="ms-2"),
        ])),
        dbc.CardBody([
            dbc.Alert(summary["headline"], color=color, className="mb-2"),
            html.P(summary["analysis"], className="mb-2"),
            html.Hr(),
            html.H6("Recommandations :"),
            html.Ul([html.Li(r) for r in summary["recommendations"]]),
        ]),
    ], className="mt-3 border-" + color)


def kpi_card(value, label, color="primary"):
    return dbc.Card(dbc.CardBody([
        html.H4(value, className=f"text-{color} mb-1"), html.P(label, className="mb-0 text-muted small"),
    ]), className="text-center shadow-sm h-100")


# ══════════════════════════════════════════════════════════════
# NAVBAR + SESSION STORE
# ══════════════════════════════════════════════════════════════

navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Div([
                html.Span("🛡️", style={"fontSize": "1.5rem"}),
                html.Span(" FraudShield", className="fw-bold fs-4 ms-2 text-white"),
                html.Span(" · BIIS", className="text-light ms-2 d-none d-md-inline small"),
            ]), width="auto"),
            dbc.Col(html.Div(id="nav-links"), className="d-flex justify-content-end align-items-center"),
        ], align="center", className="w-100"),
    ], fluid=True),
    color="dark", dark=True, className="mb-4",
)


# ══════════════════════════════════════════════════════════════
# PAGE LOGIN
# ══════════════════════════════════════════════════════════════

login_page = dbc.Container([
    dbc.Row(dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.H4([html.Span("🛡️ "), "FraudShield — Connexion"], className="mb-0 text-center")),
            dbc.CardBody([
                dbc.Label("Email"),
                dbc.Input(id="login-email", type="email", placeholder="analyste@banque.ci"),
                dbc.Label("Mot de passe", className="mt-2"),
                dbc.Input(id="login-password", type="password"),
                dbc.Button("Se connecter", id="btn-login", color="primary", className="w-100 mt-3"),
                html.Div(id="login-error", className="mt-2"),
                html.Hr(),
                html.P([
                    html.Small("Mode démo sans compte : "),
                    dbc.Button("Accès direct", id="btn-demo", color="outline-secondary", size="sm"),
                ], className="text-center text-muted"),
            ]),
        ], className="shadow", style={"maxWidth": "400px", "margin": "80px auto"}),
    ], md=6, lg=4), justify="center"),
], fluid=True)


# ══════════════════════════════════════════════════════════════
# PAGE SAISIE
# ══════════════════════════════════════════════════════════════

manual_form = dbc.Card([
    dbc.CardHeader(html.H5([html.I(className="bi bi-pencil-square me-2"), "Saisie manuelle"], className="mb-0")),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([dbc.Label("Montant (FCFA)"), dbc.Input(id="in-amount", type="number", value=150, min=0, step=0.01)], md=4),
            dbc.Col([dbc.Label("Heure (0-23)"), dbc.Input(id="in-hour", type="number", value=14, min=0, max=23)], md=4),
            dbc.Col([dbc.Label("Catégorie marchand"), dbc.Select(id="in-merchant", value="retail",
                     options=[{"label": c.capitalize(), "value": c} for c in MERCHANT_CATEGORIES_ORDERED])], md=4),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("Distance domicile (km)"), dbc.Input(id="in-distance", type="number", value=5.0, min=0, step=0.1)], md=3),
            dbc.Col([dbc.Label("Carte présente"), dbc.Select(id="in-card", value="1",
                     options=[{"label": "Oui", "value": "1"}, {"label": "Non", "value": "0"}])], md=3),
            dbc.Col([dbc.Label("Ancienneté compte (j)"), dbc.Input(id="in-age", type="number", value=800, min=0)], md=3),
            dbc.Col([dbc.Label("Txn dernière heure"), dbc.Input(id="in-txn-hour", type="number", value=0, min=0)], md=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("Montant moyen 30j"), dbc.Input(id="in-avg30", type="number", value=80, min=0, step=0.01)], md=3),
            dbc.Col([dbc.Label("Transaction étranger"), dbc.Select(id="in-foreign", value="0",
                     options=[{"label": "Non", "value": "0"}, {"label": "Oui", "value": "1"}])], md=3),
            dbc.Col([dbc.Label("Refus aujourd'hui"), dbc.Input(id="in-declines", type="number", value=0, min=0)], md=3),
            dbc.Col([dbc.Label(""), dbc.Button([html.I(className="bi bi-shield-check me-2"), "Analyser"],
                     id="btn-score", color="primary", className="w-100 mt-4")], md=3),
        ], className="mb-3"),
    ]),
], className="shadow mb-4")

csv_upload = dbc.Card([
    dbc.CardHeader(html.H5([html.I(className="bi bi-file-earmark-csv me-2"), "Import CSV"], className="mb-0")),
    dbc.CardBody([
        dcc.Upload(
            id="upload-csv",
            children=html.Div(["Glisser-déposer ou ", html.A("sélectionner un fichier CSV", className="text-primary")]),
            style={"borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "8px",
                   "textAlign": "center", "padding": "30px", "borderColor": "#dee2e6"},
        ),
        html.Div(id="csv-status", className="mt-2"),
    ]),
], className="shadow mb-4")

page_saisie = html.Div([
    manual_form, html.Div(id="result-manual", className="mb-4"),
    csv_upload, html.Div(id="result-csv"),
])


# ══════════════════════════════════════════════════════════════
# PAGE ANALYSTE
# ══════════════════════════════════════════════════════════════

page_analyste = html.Div([
    dbc.Card([
        dbc.CardHeader(html.H5([html.I(className="bi bi-search me-2"), "Analyse des transactions"], className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Source"),
                    dbc.Select(id="analyste-source", value="test", options=[
                        {"label": "Jeu de test (démo)", "value": "test"},
                        {"label": "Historique BD", "value": "db", "disabled": not DB_ENABLED},
                    ]),
                ], md=3),
                dbc.Col([
                    dbc.Label("Filtre"),
                    dbc.Select(id="filter-pred", value="all", options=[
                        {"label": "Toutes", "value": "all"},
                        {"label": "Fraudes détectées", "value": "fraud"},
                        {"label": "Légitimes", "value": "legit"},
                        {"label": "Score > 50%", "value": "suspect"},
                    ]),
                ], md=3),
                dbc.Col([
                    dbc.Label("Nb lignes"),
                    dbc.Input(id="filter-topn", type="number", value=30, min=5, max=200),
                ], md=2),
                dbc.Col([
                    dbc.Label(""),
                    dbc.Button([html.I(className="bi bi-table me-2"), "Charger"],
                               id="btn-load-data", color="primary", className="w-100 mt-4"),
                ], md=2),
            ]),
        ]),
    ], className="shadow mb-4"),
    html.Div(id="analyste-table"),
    html.Div(id="analyste-detail"),
    html.Div(id="analyste-feedback"),
])


# ══════════════════════════════════════════════════════════════
# PAGE ADMIN (construite à la demande)
# ══════════════════════════════════════════════════════════════

def build_admin_page():
    lgb_m = metrics_info.get("lightgbm_tuned", {})
    cost = cost_info.get("at_optimal", {})
    thr = _get_threshold()

    kpi_row = dbc.Row([
        dbc.Col(kpi_card(f"{lgb_m.get('pr_auc', 0):.3f}", "PR-AUC", "primary"), md=2),
        dbc.Col(kpi_card(f"{lgb_m.get('roc_auc', 0):.3f}", "ROC-AUC", "info"), md=2),
        dbc.Col(kpi_card(f"{thr:.2f}", "Seuil optimal", "dark"), md=2),
        dbc.Col(kpi_card(f"{cost.get('precision', 0):.1%}", "Précision", "success"), md=2),
        dbc.Col(kpi_card(f"{cost.get('recall', 0):.1%}", "Rappel", "warning"), md=2),
        dbc.Col(kpi_card(f"{cost.get('f1', 0):.3f}", "F1-score", "danger"), md=2),
    ], className="mb-4")

    cost_row = dbc.Row([
        dbc.Col(kpi_card(f"{cost.get('net_benefit_fcfa', 0):,} F", "Bénéfice net", "success"), md=3),
        dbc.Col(kpi_card(f"{cost.get('saved_fcfa', 0):,} F", "Montants sauvés", "primary"), md=3),
        dbc.Col(kpi_card(f"{cost.get('cost_total_fcfa', 0):,} F", "Coût total", "danger"), md=3),
        dbc.Col(kpi_card(f"{cost_info.get('cost_fp_fixed_fcfa', 0):,} F", "Coût/vérification", "secondary"), md=3),
    ], className="mb-4")

    # Importance SHAP globale (depuis le CSV pré-calculé)
    shap_path = ARTIFACT_DIR / "shap_global_importance.csv"
    shap_fig = go.Figure()
    if shap_path.exists():
        imp = pd.read_csv(shap_path).sort_values("pct", ascending=True)
        shap_fig = go.Figure(go.Bar(
            x=imp["pct"], y=imp["feature"].map(FEATURE_LABELS_FR), orientation="h",
            marker_color="#FF7043", text=[f"{v:.1f}%" for v in imp["pct"]], textposition="auto",
        ))
        shap_fig.update_layout(
            title="Importance SHAP globale", height=400,
            margin=dict(l=10, r=10, t=40, b=30), plot_bgcolor="white", xaxis_title="|SHAP| moyen (%)",
        )

    # Confusion matrix (sur échantillon réduit)
    model = _get_model()
    X_t, y_t = _get_test_data()
    proba_test = model.predict_proba(X_t)[:, 1]
    preds_test = (proba_test >= thr).astype(int)
    tp = ((preds_test == 1) & (y_t == 1)).sum()
    fp = ((preds_test == 1) & (y_t == 0)).sum()
    fn = ((preds_test == 0) & (y_t == 1)).sum()
    tn = ((preds_test == 0) & (y_t == 0)).sum()

    cm_fig = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["Prédit Légitime", "Prédit Fraude"],
        y=["Réel Légitime", "Réel Fraude"], colorscale="Oranges", showscale=False,
        text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
        texttemplate="%{text}", textfont={"size": 16},
    ))
    cm_fig.update_layout(title=f"Matrice de confusion (seuil={thr:.2f})", height=350,
                         margin=dict(l=10, r=10, t=40, b=30))

    return html.Div([
        html.H5([html.I(className="bi bi-gear me-2"), "Performance du modèle"], className="mb-3"),
        kpi_row,
        html.H6("Analyse de coût (contexte UEMOA)", className="mb-2 text-muted"),
        cost_row,
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=shap_fig, config={"displayModeBar": False})),
                             className="shadow"), md=7),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=cm_fig, config={"displayModeBar": False})),
                             className="shadow"), md=5),
        ], className="mb-4"),
        html.Div(id="admin-db-stats"),
        dbc.Button([html.I(className="bi bi-arrow-clockwise me-2"), "Rafraîchir stats BD"],
                   id="btn-refresh-stats", color="outline-primary", className="mt-2",
                   style={"display": "inline-block" if DB_ENABLED else "none"}),
    ])


# ══════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="session", storage_type="session", data={"logged_in": not DB_ENABLED, "role": "admin", "user_id": None, "email": ""}),
    navbar,
    dbc.Container(id="page-content", fluid=True, className="px-4"),
])


# ══════════════════════════════════════════════════════════════
# CALLBACKS — NAVIGATION
# ══════════════════════════════════════════════════════════════

@callback(Output("nav-links", "children"), Input("session", "data"))
def update_nav(session):
    if not session.get("logged_in"):
        return html.Span("Non connecté", className="text-muted small")
    links = [
        dbc.NavLink([html.I(className="bi bi-pencil-square me-1"), "Saisie"], href="/saisie", active="exact", className="text-white"),
        dbc.NavLink([html.I(className="bi bi-search me-1"), "Analyste"], href="/analyste", active="exact", className="text-white"),
    ]
    if session.get("role") == "admin":
        links.append(dbc.NavLink([html.I(className="bi bi-gear me-1"), "Admin"], href="/admin", active="exact", className="text-white"))
    if DB_ENABLED:
        links.append(dbc.Button([html.I(className="bi bi-box-arrow-right me-1"), session.get("email", "")[:20]],
                                id="btn-logout", color="outline-light", size="sm", className="ms-2"))
    return dbc.Nav(links, pills=True)


@callback(Output("page-content", "children"), [Input("url", "pathname"), Input("session", "data")])
def render_page(pathname, session):
    if not session.get("logged_in"):
        return login_page
    if pathname == "/analyste":
        return page_analyste
    if pathname == "/admin" and session.get("role") == "admin":
        return build_admin_page()
    return page_saisie


# ══════════════════════════════════════════════════════════════
# CALLBACKS — AUTH
# ══════════════════════════════════════════════════════════════

@callback(
    [Output("session", "data", allow_duplicate=True), Output("login-error", "children"), Output("url", "pathname", allow_duplicate=True)],
    [Input("btn-login", "n_clicks"), Input("btn-demo", "n_clicks")],
    [State("login-email", "value"), State("login-password", "value"), State("session", "data")],
    prevent_initial_call=True,
)
def handle_login(n_login, n_demo, email, password, session):
    trigger = ctx.triggered_id
    if trigger == "btn-demo":
        return {"logged_in": True, "role": "admin", "user_id": None, "email": "demo"}, "", "/saisie"
    if trigger == "btn-login" and email and password:
        try:
            from database import sign_in
            user = sign_in(email, password)
            return {
                "logged_in": True, "role": user["role"],
                "user_id": user["user_id"], "email": user["email"],
            }, "", "/saisie"
        except Exception as e:
            return no_update, dbc.Alert(f"Erreur : {str(e)[:100]}", color="danger"), no_update
    return no_update, no_update, no_update


@callback(
    [Output("session", "data", allow_duplicate=True), Output("url", "pathname", allow_duplicate=True)],
    Input("btn-logout", "n_clicks"),
    prevent_initial_call=True,
)
def handle_logout(n):
    if n:
        from database import sign_out
        try:
            sign_out()
        except Exception:
            pass
        return {"logged_in": False, "role": None, "user_id": None, "email": ""}, "/"
    return no_update, no_update


# ══════════════════════════════════════════════════════════════
# CALLBACKS — SAISIE MANUELLE
# ══════════════════════════════════════════════════════════════

@callback(
    Output("result-manual", "children"),
    Input("btn-score", "n_clicks"),
    [State("in-amount", "value"), State("in-hour", "value"), State("in-merchant", "value"),
     State("in-distance", "value"), State("in-card", "value"), State("in-age", "value"),
     State("in-txn-hour", "value"), State("in-avg30", "value"), State("in-foreign", "value"),
     State("in-declines", "value"), State("session", "data")],
    prevent_initial_call=True,
)
def score_manual(n, amount, hour, merchant, distance, card, age, txn_h, avg30, foreign, declines, session):
    if n is None:
        return no_update
    try:
        txn = {
            "amount": float(amount or 0), "hour": int(hour or 0),
            "merchant_category": merchant, "distance_from_home_km": float(distance or 0),
            "card_present": int(card), "account_age_days": int(age or 0),
            "txn_last_hour": int(txn_h or 0), "avg_amount_30d": float(avg30 or 1),
            "foreign_country": int(foreign), "num_declines_today": int(declines or 0),
        }
        raw_df = pd.DataFrame([txn])
        X = engineer_features(raw_df, merchant_encoder, is_training=False)
        result = _explain_single(X)

        # Log en BD si possible
        if DB_ENABLED:
            try:
                from database import log_transaction
                summary = _generate_summary(result, mode="auto")
                log_transaction(txn, result, summary, source="manual", user_id=session.get("user_id"))
            except Exception:
                pass

        border = "border-danger" if result["prediction"] == 1 else "border-success"
        return dbc.Card([
            dbc.CardHeader(html.Div([
                html.H5("Résultat", className="d-inline me-3"), score_badge(result["probability"]),
            ])),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=gauge_fig(result["probability"]), config={"displayModeBar": False}), md=4),
                    dbc.Col(dcc.Graph(figure=shap_waterfall_fig(result["shap_values"], "Facteurs de risque (SHAP)"),
                                      config={"displayModeBar": False}), md=8),
                ]),
                copilot_panel(result),
            ]),
        ], className=f"shadow {border}")

    except Exception as e:
        return dbc.Alert(f"Erreur : {str(e)}", color="danger")


# ══════════════════════════════════════════════════════════════
# CALLBACKS — IMPORT CSV
# ══════════════════════════════════════════════════════════════

@callback(
    [Output("result-csv", "children"), Output("csv-status", "children")],
    Input("upload-csv", "contents"), State("upload-csv", "filename"),
    prevent_initial_call=True,
)
def analyze_csv(contents, filename):
    if contents is None:
        return no_update, no_update
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df_raw = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

        missing = set(RAW_FEATURES) - set(df_raw.columns)
        if missing:
            return no_update, dbc.Alert(f"Colonnes manquantes : {missing}", color="danger")

        model = _get_model()
        threshold = _get_threshold()
        X = engineer_features(df_raw, merchant_encoder, is_training=False)
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)

        df_results = df_raw[["amount", "hour", "merchant_category", "distance_from_home_km"]].copy()
        df_results["score"] = np.round(proba, 4)
        df_results["décision"] = np.where(preds == 1, "🔴 FRAUDE", "🟢 LÉGITIME")
        df_results = df_results.sort_values("score", ascending=False)

        n_fraud = (preds == 1).sum()
        summary_row = dbc.Row([
            dbc.Col(kpi_card(f"{len(df_raw)}", "Transactions", "primary"), md=3),
            dbc.Col(kpi_card(f"{n_fraud}", "Fraudes", "danger"), md=3),
            dbc.Col(kpi_card(f"{n_fraud / len(df_raw):.1%}", "Taux fraude", "warning"), md=3),
            dbc.Col(kpi_card(f"{df_raw.loc[preds == 1, 'amount'].sum():,.0f} F", "Montant à risque", "success"), md=3),
        ], className="mb-3")

        table = dash_table.DataTable(
            data=df_results.head(50).to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_results.columns],
            style_cell={"textAlign": "center", "padding": "8px", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
            style_data_conditional=[
                {"if": {"filter_query": '{décision} contains "FRAUDE"'}, "backgroundColor": "#FFEBEE", "fontWeight": "bold"},
            ],
            sort_action="native", page_size=15,
        )

        status = dbc.Alert(f"✅ {filename} — {len(df_raw)} transactions", color="success")
        return dbc.Card([dbc.CardHeader(html.H5(f"Résultats — {filename}")),
                         dbc.CardBody([summary_row, table])], className="shadow"), status

    except Exception as e:
        return no_update, dbc.Alert(f"Erreur : {str(e)}", color="danger")


# ══════════════════════════════════════════════════════════════
# CALLBACKS — ANALYSTE
# ══════════════════════════════════════════════════════════════

@callback(
    Output("analyste-table", "children"),
    Input("btn-load-data", "n_clicks"),
    [State("analyste-source", "value"), State("filter-pred", "value"), State("filter-topn", "value")],
    prevent_initial_call=True,
)
def load_analyste_data(n, source, filter_pred, topn):
    if n is None:
        return no_update

    topn = int(topn or 30)

    if source == "db" and DB_ENABLED:
        try:
            from database import get_recent_transactions
            fraud_only = filter_pred == "fraud"
            records = get_recent_transactions(limit=topn, fraud_only=fraud_only)
            if not records:
                return dbc.Alert("Aucune transaction en base.", color="info")

            df_show = pd.DataFrame(records)
            display_cols = [c for c in ["id", "amount", "hour", "merchant_category", "fraud_probability",
                                         "risk_level", "summary_mode", "scored_at"] if c in df_show.columns]
            return dbc.Card([
                dbc.CardHeader(html.H5(f"{len(df_show)} transactions (historique BD)")),
                dbc.CardBody(dash_table.DataTable(
                    data=df_show[display_cols].to_dict("records"),
                    columns=[{"name": c, "id": c} for c in display_cols],
                    style_cell={"textAlign": "center", "padding": "6px", "fontSize": "12px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                    sort_action="native", page_size=15,
                    row_selectable="single", id="data-table",
                )),
            ], className="shadow mb-4")
        except Exception as e:
            return dbc.Alert(f"Erreur BD : {str(e)}", color="danger")

    # Source = jeu de test (démo)
    model = _get_model()
    threshold = _get_threshold()
    X_t, y_t = _get_test_data()
    proba = model.predict_proba(X_t)[:, 1]
    preds = (proba >= threshold).astype(int)

    df_show = X_t.copy()
    df_show["score"] = np.round(proba, 4)
    df_show["prediction"] = preds
    df_show["réel"] = y_t.values
    df_show["statut"] = np.where(
        (preds == 1) & (y_t.values == 1), "✅ TP",
        np.where((preds == 1) & (y_t.values == 0), "⚠️ FP",
                 np.where((preds == 0) & (y_t.values == 1), "❌ FN", "— TN")))

    if filter_pred == "fraud":
        df_show = df_show[df_show["prediction"] == 1]
    elif filter_pred == "legit":
        df_show = df_show[df_show["prediction"] == 0]
    elif filter_pred == "suspect":
        df_show = df_show[df_show["score"] > 0.5]

    df_show = df_show.sort_values("score", ascending=False).head(topn)
    display_cols = ["amount", "hour", "distance_from_home_km", "amount_ratio",
                    "txn_last_hour", "num_declines_today", "score", "statut"]

    return dbc.Card([
        dbc.CardHeader(html.H5(f"{len(df_show)} transactions (jeu de test)")),
        dbc.CardBody(dash_table.DataTable(
            data=df_show[display_cols].round(3).to_dict("records"),
            columns=[{"name": FEATURE_LABELS_FR.get(c, c), "id": c} for c in display_cols],
            style_cell={"textAlign": "center", "padding": "6px", "fontSize": "12px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
            style_data_conditional=[
                {"if": {"filter_query": '{statut} contains "TP"'}, "backgroundColor": "#E8F5E9"},
                {"if": {"filter_query": '{statut} contains "FP"'}, "backgroundColor": "#FFF3E0"},
                {"if": {"filter_query": '{statut} contains "FN"'}, "backgroundColor": "#FFEBEE"},
            ],
            sort_action="native", page_size=15,
            row_selectable="single", id="data-table",
        )),
    ], className="shadow mb-4")


@callback(
    Output("analyste-detail", "children"),
    Input("data-table", "selected_rows"),
    State("data-table", "data"),
    prevent_initial_call=True,
)
def show_detail(selected, data):
    if not selected:
        return no_update

    row = data[selected[0]]

    if "id" in row and DB_ENABLED:
        try:
            from database import get_transaction_detail
            detail = get_transaction_detail(row["id"])
            if detail:
                score = float(detail.get("fraud_probability", 0))
                return dbc.Card([
                    dbc.CardHeader(html.Div([html.H5("Détail", className="d-inline me-3"), score_badge(score)])),
                    dbc.CardBody([
                        html.P(detail.get("summary_headline", ""), className="fw-bold"),
                        html.P(detail.get("summary_analysis", "")),
                        html.Hr(),
                        html.H6("Décisions analystes :"),
                        html.Ul([
                            html.Li(f"{d['decision']} — {d.get('comment', '')} ({d.get('decided_at', '')[:10]})")
                            for d in detail.get("alert_decisions", [])
                        ]) if detail.get("alert_decisions") else html.P("Aucune décision.", className="text-muted"),
                    ]),
                ], className="shadow")
        except Exception:
            pass

    X_t, _ = _get_test_data()
    idx = X_t[
        (X_t["amount"].round(3) == round(row.get("amount", 0), 3)) &
        (X_t["hour"] == row.get("hour", -1))
    ].index

    if len(idx) == 0:
        return dbc.Alert("Transaction introuvable pour le détail SHAP.", color="warning")

    X_row = X_t.iloc[[idx[0]]]
    result = _explain_single(X_row)

    return dbc.Card([
        dbc.CardHeader(html.Div([html.H5("Détail transaction", className="d-inline me-3"),
                                 score_badge(result["probability"])])),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=gauge_fig(result["probability"]), config={"displayModeBar": False}), md=4),
                dbc.Col(dcc.Graph(figure=shap_waterfall_fig(result["shap_values"], "Explication SHAP"),
                                  config={"displayModeBar": False}), md=8),
            ]),
            copilot_panel(result),
        ]),
    ], className="shadow")


# ══════════════════════════════════════════════════════════════
# CALLBACKS — ADMIN STATS BD
# ══════════════════════════════════════════════════════════════

@callback(
    Output("admin-db-stats", "children"),
    Input("btn-refresh-stats", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_db_stats(n):
    if not DB_ENABLED:
        return no_update
    try:
        from database import get_admin_stats
        stats = get_admin_stats()
        if not stats:
            return dbc.Alert("Aucune donnée en base.", color="info")
        return dbc.Card([
            dbc.CardHeader(html.H5([html.I(className="bi bi-database me-2"), "Statistiques Supabase"])),
            dbc.CardBody(dbc.Row([
                dbc.Col(kpi_card(str(stats.get("total_scored", 0)), "Total scoré", "primary"), md=2),
                dbc.Col(kpi_card(str(stats.get("total_alerts", 0)), "Alertes", "danger"), md=2),
                dbc.Col(kpi_card(str(stats.get("confirmed_frauds", 0)), "Confirmées", "warning"), md=2),
                dbc.Col(kpi_card(str(stats.get("false_positives", 0)), "Faux positifs", "info"), md=2),
                dbc.Col(kpi_card(str(stats.get("pending_review", 0)), "En attente", "secondary"), md=2),
                dbc.Col(kpi_card(f"{stats.get('avg_processing_ms', 0)} ms", "Temps moyen", "dark"), md=2),
            ])),
        ], className="shadow mt-4")
    except Exception as e:
        return dbc.Alert(f"Erreur : {str(e)}", color="danger")


# ══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True, host="0.0.0.0", port=port)
