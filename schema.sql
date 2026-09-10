-- ══════════════════════════════════════════════════════════════
-- FraudShield — Schéma de base de données · Supabase (PostgreSQL)
-- Phase 8 : Auth RBAC, journal des transactions, feedback analyste
-- ══════════════════════════════════════════════════════════════

-- ──────────────────────────────────────────────────────────────
-- 1. TABLE DES PROFILS UTILISATEURS (extension de auth.users)
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT NOT NULL,
    full_name TEXT,
    role TEXT NOT NULL DEFAULT 'analyst' CHECK (role IN ('analyst', 'admin')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Trigger : créer automatiquement un profil à l'inscription
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, role)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', ''),
        COALESCE(NEW.raw_user_meta_data->>'role', 'analyst')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ──────────────────────────────────────────────────────────────
-- 2. TABLE DES TRANSACTIONS SCORÉES (journal d'audit)
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.transactions_scored (
    id BIGSERIAL PRIMARY KEY,
    
    -- Données brutes de la transaction
    amount NUMERIC(12, 2) NOT NULL,
    hour SMALLINT NOT NULL,
    merchant_category TEXT NOT NULL,
    distance_from_home_km NUMERIC(8, 2),
    card_present SMALLINT,
    account_age_days INTEGER,
    txn_last_hour SMALLINT,
    avg_amount_30d NUMERIC(12, 2),
    foreign_country SMALLINT,
    num_declines_today SMALLINT,
    
    -- Résultats du modèle
    fraud_probability NUMERIC(6, 4) NOT NULL,
    fraud_prediction SMALLINT NOT NULL,  -- 0 ou 1
    risk_level TEXT NOT NULL,             -- FAIBLE / MOYEN / ÉLEVÉ
    threshold_used NUMERIC(6, 4) NOT NULL,
    
    -- Résumé copilote
    summary_headline TEXT,
    summary_analysis TEXT,
    summary_mode TEXT,                    -- rule-based / llm
    
    -- Top 3 raisons SHAP
    top_reasons JSONB,
    
    -- Métadonnées
    source TEXT DEFAULT 'manual',         -- manual / csv / api
    scored_by UUID REFERENCES public.profiles(id),
    scored_at TIMESTAMPTZ DEFAULT now(),
    
    -- Index pour les recherches fréquentes
    processing_time_ms NUMERIC(8, 2)
);

CREATE INDEX IF NOT EXISTS idx_txn_scored_at ON public.transactions_scored (scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_txn_prediction ON public.transactions_scored (fraud_prediction);
CREATE INDEX IF NOT EXISTS idx_txn_risk ON public.transactions_scored (risk_level);
CREATE INDEX IF NOT EXISTS idx_txn_scored_by ON public.transactions_scored (scored_by);

-- ──────────────────────────────────────────────────────────────
-- 3. TABLE DES DÉCISIONS ANALYSTES (feedback humain)
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.alert_decisions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT REFERENCES public.transactions_scored(id) ON DELETE CASCADE NOT NULL,
    
    -- Décision de l'analyste
    decision TEXT NOT NULL CHECK (decision IN (
        'confirmed_fraud',    -- l'analyste confirme la fraude
        'false_positive',     -- faux positif, transaction légitime
        'escalated',          -- escaladé au supérieur
        'pending'             -- en attente de décision
    )),
    
    comment TEXT,              -- note libre de l'analyste
    
    -- Métadonnées
    decided_by UUID REFERENCES public.profiles(id) NOT NULL,
    decided_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_alert_txn ON public.alert_decisions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_alert_decision ON public.alert_decisions (decision);
CREATE INDEX IF NOT EXISTS idx_alert_decided_at ON public.alert_decisions (decided_at DESC);

-- ──────────────────────────────────────────────────────────────
-- 4. VUE POUR LE DASHBOARD ADMIN (statistiques)
-- ──────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW public.admin_stats AS
SELECT
    COUNT(*) AS total_scored,
    COUNT(*) FILTER (WHERE fraud_prediction = 1) AS total_alerts,
    COUNT(*) FILTER (WHERE fraud_prediction = 0) AS total_legit,
    ROUND(AVG(fraud_probability)::NUMERIC, 4) AS avg_score,
    ROUND(SUM(CASE WHEN fraud_prediction = 1 THEN amount ELSE 0 END), 2) AS total_amount_at_risk,
    ROUND(AVG(processing_time_ms)::NUMERIC, 1) AS avg_processing_ms,
    MIN(scored_at) AS first_scored,
    MAX(scored_at) AS last_scored,
    -- Feedback stats
    (SELECT COUNT(*) FROM public.alert_decisions WHERE decision = 'confirmed_fraud') AS confirmed_frauds,
    (SELECT COUNT(*) FROM public.alert_decisions WHERE decision = 'false_positive') AS false_positives,
    (SELECT COUNT(*) FROM public.alert_decisions WHERE decision = 'pending') AS pending_review
FROM public.transactions_scored;

-- ──────────────────────────────────────────────────────────────
-- 5. ROW LEVEL SECURITY (RBAC)
-- ──────────────────────────────────────────────────────────────

-- Activer RLS sur toutes les tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.transactions_scored ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alert_decisions ENABLE ROW LEVEL SECURITY;

-- Profils : chaque utilisateur voit son propre profil, admin voit tout
CREATE POLICY "Users view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Admins view all profiles"
    ON public.profiles FOR SELECT
    USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'admin')
    );

-- Transactions : tout utilisateur authentifié peut lire et insérer
CREATE POLICY "Authenticated users read transactions"
    ON public.transactions_scored FOR SELECT
    TO authenticated
    USING (true);

CREATE POLICY "Authenticated users insert transactions"
    ON public.transactions_scored FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Décisions : tout analyste/admin peut lire et insérer
CREATE POLICY "Authenticated users read decisions"
    ON public.alert_decisions FOR SELECT
    TO authenticated
    USING (true);

CREATE POLICY "Authenticated users insert decisions"
    ON public.alert_decisions FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = decided_by);

-- Admins peuvent tout modifier
CREATE POLICY "Admins full access transactions"
    ON public.transactions_scored FOR ALL
    USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'admin')
    );

CREATE POLICY "Admins full access decisions"
    ON public.alert_decisions FOR ALL
    USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'admin')
    );

-- ──────────────────────────────────────────────────────────────
-- 6. DONNÉES INITIALES (compte admin)
-- ──────────────────────────────────────────────────────────────
-- À exécuter APRÈS avoir créé le premier utilisateur via l'UI Supabase :
-- UPDATE public.profiles SET role = 'admin' WHERE email = 'karaboue9934@gmail.com';
