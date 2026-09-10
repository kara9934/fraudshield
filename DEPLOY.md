# FraudShield — Guide de déploiement

## Architecture de déploiement

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Dashboard Dash  │────▶│   API FastAPI     │────▶│   Supabase       │
│  (Render)        │     │  (HF Spaces)     │     │  (PostgreSQL)    │
│  Port 8050       │     │  Port 7860       │     │  Auth + RLS      │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │  Groq API    │
                        │  Llama 3.3   │
                        │  (gratuit)   │
                        └──────────────┘
```

Tout est gratuit pour la démo.

---

## Étape 1 — Supabase (base de données + auth)

1. Aller sur https://supabase.com → créer un compte → nouveau projet
2. Nom du projet : `fraudshield` — Région : Europe (Ouest)
3. Dans l'éditeur SQL (icône SQL à gauche), coller tout le contenu de `schema.sql` et exécuter
4. Dans Settings → API : copier l'URL du projet, la clé `anon` et la clé `service_role`
5. Dans Authentication → Users : créer le premier utilisateur (email + mot de passe)
6. Dans l'éditeur SQL, exécuter :
   ```sql
   UPDATE public.profiles SET role = 'admin' WHERE email = 'votre@email.com';
   ```

## Étape 2 — Groq (copilote IA)

1. Aller sur https://console.groq.com → créer un compte
2. API Keys → Create API Key → copier la clé
3. Le modèle utilisé est `llama-3.3-70b-versatile` (gratuit, 30 req/min)

## Étape 3 — Entraînement du modèle

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'entraînement (génère artifacts/)
python train.py --data data/transactions.csv
```

Vérifier que `artifacts/` contient : `lgbm_model.joblib`, `optimal_threshold.joblib`,
`merchant_encoder.joblib`, `metrics.json`, `cost_analysis.json`, `shap_global_importance.csv`,
`X_test.parquet`, `y_test.parquet`.

## Étape 4 — Test local

```bash
# Créer le fichier .env à partir du template
cp .env.example .env
# Remplir les valeurs Supabase + Groq

# Lancer l'API
uvicorn api:app --reload --port 8000

# Dans un autre terminal, lancer le dashboard
python app.py

# Tester l'API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount":450,"hour":2,"merchant_category":"electronics","distance_from_home_km":120,"card_present":0,"account_age_days":90,"txn_last_hour":3,"avg_amount_30d":55,"foreign_country":1,"num_declines_today":2}'
```

## Étape 5 — Déployer l'API sur Hugging Face Spaces

1. Aller sur https://huggingface.co → créer un compte → New Space
2. Nom : `fraudshield-api` — SDK : Docker — Visibility : Public
3. Dans les Settings du Space → Variables and Secrets :
   - `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_KEY`, `GROQ_API_KEY`
4. Cloner le repo du Space et copier les fichiers du projet :
   ```bash
   git clone https://huggingface.co/spaces/VOTRE_USER/fraudshield-api
   cd fraudshield-api
   # Copier : features.py, explain.py, database.py, api.py, requirements.txt,
   #          Dockerfile (renommer la cible), artifacts/, assets/
   ```
5. Créer un `Dockerfile` simplifié (cible API) :
   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 7860
   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
   ```
6. Push → le Space se build automatiquement

## Étape 6 — Déployer le Dashboard sur Render

1. Aller sur https://render.com → créer un compte → New Web Service
2. Connecter le repo GitHub (ou Docker)
3. Docker target : `dash`
4. Variables d'environnement : `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_KEY`, `GROQ_API_KEY`, `PORT=8050`
5. Le fichier `render.yaml` configure automatiquement si vous utilisez les Blueprints Render

---

## Mode dégradé (sans comptes externes)

Le système fonctionne sans Supabase ni Groq :
- Sans Supabase : pas d'auth, pas d'historique — scoring local fonctionne
- Sans Groq : copilote en mode rule-based (déterministe, hors-ligne)

## Docker local

```bash
docker-compose up --build
# API sur http://localhost:8000
# Dashboard sur http://localhost:8050
```
