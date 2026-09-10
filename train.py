"""
train.py — Pipeline d'entraînement complet · FraudShield
Exécution unique : python train.py [--data chemin/transactions.csv]

Fusionne 5 scripts en 1 :
  Phase A  Préparation des données & split stratifié
  Phase B  Tuning LightGBM (grid search CV, PR-AUC)
  Phase C  Calibrage du seuil par matrice de coûts (contexte UEMOA)
  Phase D  Analyse SHAP & importance globale

Artefacts produits dans artifacts/ :
  lgbm_model.joblib, optimal_threshold.joblib, merchant_encoder.joblib,
  metrics.json, cost_analysis.json, shap_global_importance.csv,
  X_test.parquet, y_test.parquet (pour le dashboard analyste)
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from features import (
    ARTIFACT_DIR,
    FEATURE_NAMES,
    RAW_FEATURES,
    build_merchant_encoder,
    engineer_features,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train")

# ── Configuration ──────────────────────────────────────────────
TEST_SIZE = 0.20
RANDOM_STATE = 42
COST_FP_FIXED_FCFA = 2_000  # Coût vérification par FP


# ══════════════════════════════════════════════════════════════
# PHASE A — PRÉPARATION DES DONNÉES
# ══════════════════════════════════════════════════════════════

def phase_a_prepare(data_path: Path):
    """Charge, split, encode, feature-engineer."""
    log.info("=" * 65)
    log.info("PHASE A — PRÉPARATION DES DONNÉES")
    log.info("=" * 65)

    df = pd.read_csv(data_path)
    log.info("Chargé : %d lignes × %d colonnes (fraude %.2f%%)",
             len(df), df.shape[1], df["is_fraud"].mean() * 100)

    # Split stratifié
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["is_fraud"],
    )
    log.info("Train : %d | Test : %d", len(df_train), len(df_test))

    # Encodeur merchant (fit sur train uniquement)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    encoder = build_merchant_encoder(df_train)
    log.info("Encodeur merchant : %s",
             {cat: f"{df_train[df_train['merchant_category']==cat]['is_fraud'].mean():.3%}"
              for cat in sorted(encoder, key=encoder.get)})

    # Feature engineering
    X_train = engineer_features(df_train, encoder, is_training=True)
    X_test = engineer_features(df_test, encoder, is_training=True)
    y_train = X_train.pop("is_fraud")
    y_test = X_test.pop("is_fraud")

    # Validation
    assert X_train.isnull().sum().sum() == 0, "NaN dans train"
    assert X_test.isnull().sum().sum() == 0, "NaN dans test"
    log.info("Features : %d — aucun NaN/Inf", len(FEATURE_NAMES))

    # Sauvegarder test pour dashboard analyste
    X_test.to_parquet(ARTIFACT_DIR / "X_test.parquet", index=False)
    y_test.to_frame().to_parquet(ARTIFACT_DIR / "y_test.parquet", index=False)

    return X_train, y_train, X_test, y_test


# ══════════════════════════════════════════════════════════════
# PHASE B — TUNING LIGHTGBM
# ══════════════════════════════════════════════════════════════

def phase_b_tune(X_train, y_train, X_test, y_test):
    """Grid search CV + entraînement final + baseline logistique."""
    log.info("\n" + "=" * 65)
    log.info("PHASE B — TUNING LIGHTGBM (grid search CV)")
    log.info("=" * 65)

    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # Grille réduite ciblée (24 combinaisons)
    configs = []
    for n_est in [500, 800]:
        for lr in [0.02, 0.05]:
            for nl, md in [(15, 4), (31, 6), (63, 8)]:
                for spw in [1.0, ratio * 0.5]:
                    configs.append({
                        "n_estimators": n_est, "learning_rate": lr,
                        "num_leaves": nl, "max_depth": md,
                        "scale_pos_weight": spw,
                    })

    log.info("Grille : %d combinaisons × CV 3-folds", len(configs))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    best_score, best_params = -1, None

    for i, p in enumerate(configs):
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            m = lgb.LGBMClassifier(
                objective="binary", boosting_type="gbdt",
                **p, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
                verbosity=-1, n_jobs=-1,
            )
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            prob = m.predict_proba(X_train.iloc[val_idx])[:, 1]
            scores.append(average_precision_score(y_train.iloc[val_idx], prob))
        ms = np.mean(scores)
        if ms > best_score:
            best_score, best_params = ms, p
            log.info("  [%2d/%d] CV PR-AUC=%.4f ← BEST  %s", i + 1, len(configs), ms, p)

    log.info("Meilleur CV PR-AUC : %.4f", best_score)

    # Modèle final sur tout le train
    model = lgb.LGBMClassifier(
        objective="binary", boosting_type="gbdt",
        **best_params, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
        verbosity=-1, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    prauc = average_precision_score(y_test, y_prob)
    rocauc = roc_auc_score(y_test, y_prob)
    log.info("Test  PR-AUC=%.4f  ROC-AUC=%.4f", prauc, rocauc)

    # Baseline logistique
    sc = StandardScaler()
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(sc.fit_transform(X_train), y_train)
    prauc_lr = average_precision_score(y_test, lr.predict_proba(sc.transform(X_test))[:, 1])
    log.info("Baseline LogReg PR-AUC=%.4f — gain LightGBM=%+.4f", prauc_lr, prauc - prauc_lr)

    joblib.dump(model, ARTIFACT_DIR / "lgbm_model.joblib")
    return model, y_prob, best_params, best_score, prauc, rocauc, prauc_lr


# ══════════════════════════════════════════════════════════════
# PHASE C — CALIBRAGE DU SEUIL PAR COÛT
# ══════════════════════════════════════════════════════════════

def phase_c_cost_threshold(X_test, y_test, y_prob, prauc, rocauc, prauc_lr, best_params, cv_score):
    """Balayage de seuils, minimisation du coût total (FP+FN)."""
    log.info("\n" + "=" * 65)
    log.info("PHASE C — CALIBRAGE DU SEUIL PAR COÛT")
    log.info("=" * 65)

    amounts = X_test["amount"].values
    thresholds = np.arange(0.01, 0.99, 0.005)
    results = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        fp_mask = (y_pred == 1) & (y_test.values == 0)
        fn_mask = (y_pred == 0) & (y_test.values == 1)
        tp_mask = (y_pred == 1) & (y_test.values == 1)

        cost_fp = fp_mask.sum() * COST_FP_FIXED_FCFA
        cost_fn = amounts[fn_mask].sum()
        saved = amounts[tp_mask].sum()
        cm = confusion_matrix(y_test, y_pred)
        tn, fp_n, fn_n, tp_n = cm.ravel()
        prec = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0
        rec = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        results.append({
            "threshold": round(thr, 4), "cost_total": int(cost_fp + cost_fn),
            "saved": int(saved), "net_benefit": int(saved - cost_fp),
            "tp": tp_n, "fp": fp_n, "fn": fn_n, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        })

    df_r = pd.DataFrame(results)
    best_idx = df_r["cost_total"].idxmin()
    best = df_r.iloc[best_idx]
    optimal = best["threshold"]

    log.info("Seuil optimal (min coût) : %.4f", optimal)
    log.info("  Précision=%.4f  Rappel=%.4f  F1=%.4f", best["precision"], best["recall"], best["f1"])
    log.info("  Coût total=%s FCFA  Bénéfice net=%s FCFA",
             f"{int(best['cost_total']):,}", f"{int(best['net_benefit']):,}")

    # Classification report au seuil optimal
    y_final = (y_prob >= optimal).astype(int)
    log.info("\n%s", classification_report(y_test, y_final,
             target_names=["Légitime", "Fraude"], digits=4))

    # Sauvegarder
    joblib.dump(optimal, ARTIFACT_DIR / "optimal_threshold.joblib")

    bp = best_params.copy()
    bp["scale_pos_weight"] = round(bp["scale_pos_weight"], 2)

    metrics = {
        "baseline_lr": {"pr_auc": round(prauc_lr, 4)},
        "lightgbm_tuned": {
            "pr_auc": round(prauc, 4), "roc_auc": round(rocauc, 4),
            "best_params": bp, "cv_pr_auc": round(cv_score, 4),
        },
    }

    cost_info = {
        "cost_fp_fixed_fcfa": COST_FP_FIXED_FCFA,
        "cost_fn": "montant réel de la transaction",
        "optimal_threshold": optimal,
        "at_optimal": {
            "cost_total_fcfa": int(best["cost_total"]),
            "saved_fcfa": int(best["saved"]),
            "net_benefit_fcfa": int(best["net_benefit"]),
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "tp": int(best["tp"]), "fp": int(best["fp"]),
            "fn": int(best["fn"]), "tn": int(best["tn"]),
        },
    }

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(ARTIFACT_DIR / "cost_analysis.json", "w") as f:
        json.dump(cost_info, f, indent=2, ensure_ascii=False)

    return optimal


# ══════════════════════════════════════════════════════════════
# PHASE D — ANALYSE SHAP
# ══════════════════════════════════════════════════════════════

def phase_d_shap(model, X_test):
    """Calcule et sauvegarde l'importance SHAP globale."""
    log.info("\n" + "=" * 65)
    log.info("PHASE D — ANALYSE SHAP")
    log.info("=" * 65)

    explainer = shap.TreeExplainer(model)
    # Échantillon pour accélérer (max 2000 lignes)
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    imp = pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    imp["pct"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum() * 100
    imp = imp.round(4).reset_index(drop=True)

    imp.to_csv(ARTIFACT_DIR / "shap_global_importance.csv", index=False)

    log.info("Importance SHAP globale :")
    for _, r in imp.iterrows():
        log.info("  %-28s %5.1f%%", r["feature"], r["pct"])

    return imp


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FraudShield — Entraînement complet")
    parser.add_argument("--data", type=str, default="data/transactions.csv",
                        help="Chemin vers transactions.csv")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {data_path}")

    # Phase A
    X_train, y_train, X_test, y_test = phase_a_prepare(data_path)

    # Phase B
    model, y_prob, best_params, cv_score, prauc, rocauc, prauc_lr = \
        phase_b_tune(X_train, y_train, X_test, y_test)

    # Phase C
    optimal = phase_c_cost_threshold(
        X_test, y_test, y_prob, prauc, rocauc, prauc_lr, best_params, cv_score,
    )

    # Phase D
    phase_d_shap(model, X_test)

    # Résumé final
    log.info("\n" + "=" * 65)
    log.info("✅ ENTRAÎNEMENT TERMINÉ")
    log.info("=" * 65)
    log.info("Artefacts dans %s/", ARTIFACT_DIR)
    for f in sorted(ARTIFACT_DIR.iterdir()):
        log.info("  %s (%s)", f.name, f"{f.stat().st_size / 1024:.0f} KB")
    log.info("Seuil opérationnel : %.4f", optimal)
    log.info("PR-AUC test : %.4f (baseline LR : %.4f)", prauc, prauc_lr)


if __name__ == "__main__":
    main()
