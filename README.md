# 🛡️ FraudShield — Détection de fraude bancaire

**Outil de scoring anti-fraude calibré pour le marché UEMOA.**
Développé par BIIS (Bureau Ivoirien d'Ingénierie Statistique).

## Fonctionnalités

- **Modèle LightGBM** tuné (PR-AUC 0.865) avec seuil calibré par matrice de coûts (contexte UEMOA)
- **Explicabilité SHAP** : chaque décision est traçable et justifiable
- **Copilote IA** : résumé analyste en langage naturel (rule-based hors-ligne + Groq/Llama 3.3 70B)
- **Dashboard Dash** : 3 vues (saisie, analyste, admin) avec auth Supabase
- **API FastAPI** : scoring temps réel, batch, CSV, endpoints SHAP et historique
- **Base Supabase** : auth RBAC, journal d'audit, feedback analyste

## Architecture

```
fraudshield/
├── data/transactions.csv       # Dataset (60 420 transactions)
├── artifacts/                  # Modèle + artefacts (générés par train.py)
├── assets/style.css            # CSS dashboard
├── features.py                 # Pipeline features (cœur partagé)
├── train.py                    # Entraînement complet (1 seul script)
├── explain.py                  # SHAP + copilote IA
├── database.py                 # Couche Supabase
├── api.py                      # API FastAPI
├── app.py                      # Dashboard Dash
├── schema.sql                  # Schéma BD Supabase
├── Dockerfile                  # Multi-cible (api / dash)
├── docker-compose.yml          # Dev local
├── render.yaml                 # Config Render
├── requirements.txt            # Dépendances
├── .env.example                # Template variables d'env
└── DEPLOY.md                   # Guide de déploiement
```

17 fichiers. Zéro redondance.

## Démarrage rapide

```bash
# 1. Installer
pip install -r requirements.txt

# 2. Entraîner le modèle
python train.py --data data/transactions.csv

# 3. Lancer l'API
uvicorn api:app --reload --port 8000

# 4. Lancer le dashboard
python app.py
```

## Résultats du modèle

| Métrique | Valeur |
|----------|--------|
| PR-AUC | 0.865 |
| ROC-AUC | 0.980 |
| Seuil opérationnel | 0.83 |
| Précision au seuil | 100% |
| Rappel au seuil | 71.4% |
| Coût FP | 2 000 FCFA/vérification |

## Stack technique

- **ML** : LightGBM, SHAP, scikit-learn
- **API** : FastAPI, Uvicorn
- **Dashboard** : Dash, Plotly, Bootstrap
- **Copilote** : Groq (Llama 3.3 70B, gratuit)
- **Base** : Supabase (PostgreSQL, gratuit)
- **Déploiement** : Docker, Hugging Face Spaces, Render

## Licence

Projet propriétaire BIIS. Usage commercial sur accord.

---
*Développé par Karaboué Vakabou — BIIS · Abidjan, Côte d'Ivoire*
