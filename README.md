# 🛡️ FraudShield

**Plateforme de détection de fraude bancaire en temps réel, calibrée pour le marché UEMOA.**

Scoring ML explicable · Copilote IA analyste · Dashboard opérationnel · API REST

Développé par **KN7** — Abidjan, Côte d'Ivoire

---

## 🚀 Démo en ligne

| Service | Lien | Description |
|---------|------|-------------|
| **Dashboard** | [fraudshield-dashboard-a36x.onrender.com](https://fraudshield-dashboard-a36x.onrender.com) | Interface analyste complète (scoring, SHAP, copilote IA) |
| **API — Documentation** | [fraudshield-api-5lvs.onrender.com/docs](https://fraudshield-api-5lvs.onrender.com/docs) | Documentation interactive des 11 endpoints |
| **API — Health check** | [fraudshield-api-5lvs.onrender.com/health](https://fraudshield-api-5lvs.onrender.com/health) | Statut du service en temps réel |

> **Note :** Les services gratuits s'endorment après 15 min d'inactivité. Le premier chargement peut prendre 30-60 secondes.

---

## 📌 Contexte et positionnement

Les solutions anti-fraude du marché (FICO Falcon, Feedzai, NICE Actimize) ne prennent pas en compte les spécificités du marché bancaire UEMOA : schémas de fraude liés au SIM swap, arnaques mobile money, contexte réglementaire BCEAO/CENTIF/GIABA.

**FraudShield** est une alternative locale, calibrée sur des données représentatives du marché ouest-africain, avec une explicabilité complète de chaque décision pour répondre aux exigences de traçabilité des régulateurs.

---

## ✨ Fonctionnalités

### Scoring ML explicable
- Modèle **LightGBM** optimisé par grid search CV (24 configurations × 3 folds)
- Seuil calibré par **matrice de coûts** adaptée au contexte UEMOA (coût FP : 2 000 FCFA/vérification, coût FN : montant réel de la fraude)
- **Explicabilité SHAP** : chaque score est accompagné des facteurs de risque classés par importance, visualisés en graphique

### Copilote IA analyste
- Résumé opérationnel en français, rédigé en langage métier bancaire
- Mode **LLM** (Groq / GPT-OSS 120B) pour les résumés en langage naturel
- Mode **rule-based** (déterministe, hors-ligne) en fallback automatique — zéro dépendance externe
- Recommandations d'action adaptées au niveau de risque

### Dashboard opérationnel (3 vues)
- **Saisie** : scoring manuel + import CSV avec résultats immédiats
- **Analyste** : historique des transactions, drill-down SHAP, feedback (confirmed_fraud / false_positive / escalated)
- **Admin** : KPIs modèle, matrice de confusion, importance SHAP globale, analyse de coûts, statistiques Supabase

### API REST (11 endpoints)
- `POST /predict` — scoring unitaire avec explicabilité SHAP complète
- `POST /predict/batch` — scoring par lot (JSON)
- `POST /predict/csv` — scoring d'un fichier CSV uploadé
- `GET /model/info` — métriques, seuil, features
- `GET /model/shap` — importance SHAP globale
- `GET /health` — healthcheck avec uptime et compteur de requêtes
- `GET /history` — historique des transactions scorées
- `GET /alerts/pending` — alertes en attente de revue
- `POST /alerts/decide` — enregistrement de la décision analyste
- `GET /stats` — statistiques admin

### Authentification et traçabilité
- Auth **Supabase** avec RBAC (rôles `analyst` / `admin`)
- Row Level Security (RLS) sur toutes les tables
- Journal d'audit complet : chaque transaction scorée, chaque décision analyste, horodatée

---

## 📊 Performance du modèle

| Métrique | Valeur |
|----------|--------|
| PR-AUC (test) | **0.865** |
| ROC-AUC (test) | **0.980** |
| Seuil opérationnel | **0.83** (calibré par coût) |
| Précision au seuil | **100%** (zéro faux positif) |
| Rappel au seuil | **71.4%** |
| F1-score | **0.833** |
| Bénéfice net estimé | **+234 000 FCFA** (sur jeu de test) |

> **Interprétation métier :** sur 100 transactions frauduleuses, le modèle en détecte 71 sans jamais bloquer un client honnête par erreur. Chaque alerte levée mérite d'être examinée.

---

## 🏗️ Architecture

```
┌─────────────────────────┐
│       Client / Prospect │
│       (navigateur web)  │
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐   ┌──────────┐
│Dashboard │   │   API    │
│  (Dash)  │   │(FastAPI) │
│ Render   │   │ Render   │
└────┬─────┘   └────┬─────┘
     │               │
     └───────┬───────┘
             │
     ┌───────┴───────┐
     ▼               ▼
┌──────────┐   ┌──────────┐
│ Supabase │   │   Groq   │
│ (Auth +  │   │(Copilote │
│  BD +    │   │   IA)    │
│  RLS)    │   │          │
└──────────┘   └──────────┘
```

```
fraudshield/
├── artifacts/                  # Modèle + artefacts (générés par train.py)
│   ├── lgbm_model.joblib       # Modèle LightGBM entraîné
│   ├── optimal_threshold.joblib# Seuil calibré par coût
│   ├── merchant_encoder.joblib # Encodeur ordinal catégories
│   ├── metrics.json            # Métriques de performance
│   ├── cost_analysis.json      # Analyse coût-bénéfice
│   ├── shap_global_importance.csv
│   ├── X_test.parquet          # Données test (dashboard analyste)
│   └── y_test.parquet
├── assets/style.css            # CSS dashboard
├── features.py                 # Pipeline features (cœur partagé)
├── train.py                    # Entraînement complet (4 phases)
├── explain.py                  # SHAP + copilote IA (rule-based + LLM)
├── database.py                 # Couche Supabase (auth, CRUD, stats)
├── api.py                      # API FastAPI (11 endpoints)
├── app.py                      # Dashboard Dash (3 vues)
├── schema.sql                  # Schéma BD (3 tables + RLS + vue admin)
├── Dockerfile                  # Multi-cible (api / dash)
├── docker-compose.yml          # Dev local
├── render.yaml                 # Config Render
├── requirements.txt            # Dépendances Python
├── .env.example                # Template variables d'environnement
└── DEPLOY.md                   # Guide de déploiement détaillé
```

---

## 🔧 Stack technique

| Couche | Technologie | Rôle |
|--------|-------------|------|
| ML | LightGBM, scikit-learn | Scoring de fraude |
| Explicabilité | SHAP (TreeExplainer) | Justification de chaque décision |
| Copilote IA | Groq (GPT-OSS 120B, gratuit) | Résumé analyste en langage naturel |
| API | FastAPI, Uvicorn | Scoring temps réel, batch, CSV |
| Dashboard | Dash, Plotly, Bootstrap | Interface analyste opérationnelle |
| Base de données | Supabase (PostgreSQL) | Auth RBAC, journal d'audit, feedback |
| Déploiement | Render, Docker | Hébergement cloud |
| Versioning | Git, GitHub | Gestion du code source |

---

## ⚡ Démarrage rapide (local)

```bash
# 1. Cloner le dépôt
git clone https://github.com/kara9934/fraudshield.git
cd fraudshield

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Remplir les clés Supabase et Groq dans .env

# 5. Entraîner le modèle (si artifacts/ est vide)
python train.py --data data/transactions.csv

# 6. Lancer l'API
uvicorn api:app --reload --port 8000

# 7. Lancer le dashboard (dans un autre terminal)
python app.py
# → Dashboard : http://localhost:8050
# → API docs  : http://localhost:8000/docs
```

---

## 🐳 Docker (local)

```bash
# Lancer les deux services en une commande
docker-compose up --build

# API : http://localhost:8000
# Dashboard : http://localhost:8050
```

---

## 📁 Dataset

| Caractéristique | Valeur |
|----------------|--------|
| Transactions | 60 420 |
| Taux de fraude | 0.70% |
| Features brutes | 10 (montant, heure, catégorie, distance, etc.) |
| Features engineered | 15 (ratios, logs, indicateurs binaires) |
| Split | 80% train / 20% test (stratifié) |

---

## 🛡️ Sécurité et conformité

- Les clés API et secrets sont exclus du dépôt (`.env` dans `.gitignore`)
- Row Level Security (RLS) active sur toutes les tables Supabase
- Rôles séparés : `analyst` (lecture + feedback) / `admin` (accès complet)
- Trigger automatique de création de profil à l'inscription
- Conçu pour répondre aux exigences de traçabilité BCEAO/CENTIF

---

## 📈 Feuille de route

- [x] Modèle LightGBM + calibrage par coût
- [x] Explicabilité SHAP (locale + globale)
- [x] Copilote IA (rule-based + LLM)
- [x] Dashboard 3 vues (saisie, analyste, admin)
- [x] API REST 11 endpoints
- [x] Auth Supabase RBAC + RLS
- [x] Déploiement cloud (Render)
- [ ] Feature store Redis (agrégats temps réel)
- [ ] Connecteur core banking
- [ ] Monitoring de dérive du modèle
- [ ] Pipeline MLOps (réentraînement automatique)

---

## 📄 Licence

Projet propriétaire KN7. Usage commercial sur accord.

---

**Développé par Karaboué Vakabou — KN7 · Abidjan, Côte d'Ivoire**
