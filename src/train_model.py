"""
train_model.py — Phase 3: Train ML model to predict delayed market reactions
Not Yet Priced In · Team 10

Builds feature matrix from SQLite + ChromaDB RAG features,
trains XGBoost classifier, evaluates on held-out 2023 data.

Usage:
    python src/train_model.py
"""
import sys
import sqlite3
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config

DB_PATH = config.DATA_DIR / "not_yet_priced_in.db"
CHROMA_DIR = config.DATA_DIR / "chroma_db"
MODEL_PATH = config.DATA_DIR / "delayed_reaction_model.pkl"
RESULTS_PATH = config.DATA_DIR / "model_results.json"

log = config.setup_logging("train_model")


# ═════════════════════════════════════════════════════════
# STEP 1: Load structured features from SQLite
# ═════════════════════════════════════════════════════════
def load_structured_features():
    """Load the ML feature matrix from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT 
            f.id, f.accession, f.ticker, f.year, f.broad_category,
            f.clean_text,
            ft.numeric_density, ft.forward_looking_density,
            ft.financial_symbol_density, ft.baseline_importance,
            ft.importance_score, ft.llm_importance_norm,
            ft.grounding_rate,
            r.reaction_class, r.MOS_prospective
        FROM filings f
        JOIN features ft ON f.id = ft.filing_id
        JOIN reactions r ON f.id = r.filing_id
        WHERE r.reaction_class IS NOT NULL 
          AND r.reaction_class != 'Unknown'
          AND f.clean_text IS NOT NULL
          AND f.clean_text != ''
    """, conn)
    conn.close()
    log.info(f"Loaded {len(df)} rows from SQLite")
    return df


# ═════════════════════════════════════════════════════════
# STEP 2: Generate RAG features from ChromaDB
# ═════════════════════════════════════════════════════════
def add_rag_features(df):
    """Query ChromaDB to get similarity-based features for each filing."""
    import chromadb
    from chromadb.utils import embedding_functions

    log.info("Loading ChromaDB for RAG features...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )

    collection = client.get_or_create_collection(
        name="sec_8k_filings",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    log.info(f"ChromaDB collection has {collection.count()} vectors")

    # For each filing, find the 10 most similar filings and compute:
    # - % of similar filings that were "Delayed"
    # - average importance score of similar filings
    # - average MOS of similar filings
    similar_delayed_pcts = []
    similar_avg_importance = []
    similar_avg_mos = []

    batch_size = 20
    accessions = df["accession"].tolist()
    texts = df["clean_text"].tolist()

    for i in range(0, len(df), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_accessions = accessions[i:i + batch_size]

        try:
            results = collection.query(
                query_texts=batch_texts,
                n_results=11,  # 11 because the filing itself might be in results
            )

            for j, (ids_list, metas_list) in enumerate(zip(results["ids"], results["metadatas"])):
                # Filter out self-match
                current_acc = batch_accessions[j]
                filtered_metas = [m for doc_id, m in zip(ids_list, metas_list) if doc_id != current_acc][:10]

                if filtered_metas:
                    delayed_count = sum(1 for m in filtered_metas if m.get("reaction_class") == "Delayed")
                    similar_delayed_pcts.append(delayed_count / len(filtered_metas))

                    imp_scores = [m.get("importance_score", 0) for m in filtered_metas if m.get("importance_score")]
                    similar_avg_importance.append(np.mean(imp_scores) if imp_scores else 0)

                    mos_scores = [m.get("MOS_prospective", 0) for m in filtered_metas if m.get("MOS_prospective")]
                    similar_avg_mos.append(np.mean(mos_scores) if mos_scores else 0)
                else:
                    similar_delayed_pcts.append(0)
                    similar_avg_importance.append(0)
                    similar_avg_mos.append(0)

        except Exception as e:
            log.warning(f"  RAG batch {i // batch_size} failed: {e}")
            for _ in batch_texts:
                similar_delayed_pcts.append(0)
                similar_avg_importance.append(0)
                similar_avg_mos.append(0)

        if (i // batch_size + 1) % 10 == 0:
            log.info(f"  RAG progress: {min(i + batch_size, len(df))}/{len(df)}")

    df["similar_delayed_pct"] = similar_delayed_pcts
    df["similar_avg_importance"] = similar_avg_importance
    df["similar_avg_mos"] = similar_avg_mos

    log.info(f"RAG features added. similar_delayed_pct mean: {df['similar_delayed_pct'].mean():.3f}")
    return df


# ═════════════════════════════════════════════════════════
# STEP 3: Feature engineering
# ═════════════════════════════════════════════════════════
def prepare_features(df):
    """Engineer features and create train/test split."""

    # Binary target: Delayed = 1, everything else = 0
    df["target"] = (df["reaction_class"] == "Delayed").astype(int)

    # One-hot encode broad_category
    category_dummies = pd.get_dummies(df["broad_category"], prefix="cat")

    # Feature columns
    numeric_features = [
        "numeric_density", "forward_looking_density",
        "financial_symbol_density", "baseline_importance",
        "importance_score", "llm_importance_norm",
        "grounding_rate", "MOS_prospective",
        "similar_delayed_pct", "similar_avg_importance", "similar_avg_mos",
    ]

    X = pd.concat([df[numeric_features], category_dummies], axis=1)
    y = df["target"]

    # Fill NaN with 0
    X = X.fillna(0)

    # Time-based split: train on 2021-2022, test on 2023
    train_mask = df["year"].isin([2021, 2022])
    test_mask = df["year"] == 2023

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    log.info(f"Train set: {len(X_train)} rows (2021-2022)")
    log.info(f"  Delayed: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    log.info(f"Test set:  {len(X_test)} rows (2023)")
    log.info(f"  Delayed: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ═════════════════════════════════════════════════════════
# STEP 4: Train XGBoost
# ═════════════════════════════════════════════════════════
def train_model(X_train, y_train):
    """Train ensemble of RandomForest + LightGBM with class imbalance handling."""

    # Calculate class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / n_pos if n_pos > 0 else 1

    log.info(f"Class balance: {n_neg} negative, {n_pos} positive (scale_pos_weight={scale_pos:.1f})")

    # Model 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Model 2: Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    )

    # Ensemble via soft voting
    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
    )

    model.fit(X_train, y_train)
    log.info("Ensemble model trained (RandomForest + GradientBoosting)!")
    return model


# ═════════════════════════════════════════════════════════
# STEP 5: Evaluate
# ═════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, feature_names):
    """Comprehensive evaluation with metrics and feature importance."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    report = classification_report(y_test, y_pred, target_names=["Not Delayed", "Delayed"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    try:
        auc_roc = roc_auc_score(y_test, y_prob)
    except:
        auc_roc = 0.0

    avg_precision = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    log.info("")
    log.info("=" * 60)
    log.info("  MODEL EVALUATION (Test Set: 2023)")
    log.info("=" * 60)
    log.info(f"  AUC-ROC:            {auc_roc:.4f}")
    log.info(f"  Average Precision:  {avg_precision:.4f}")
    log.info(f"  F1 Score (Delayed): {f1:.4f}")
    log.info("")
    log.info(f"  Confusion Matrix:")
    log.info(f"                    Predicted")
    log.info(f"                Not Del.  Delayed")
    log.info(f"  Actual Not Del. {cm[0][0]:>6}   {cm[0][1]:>6}")
    log.info(f"  Actual Delayed  {cm[1][0]:>6}   {cm[1][1]:>6}")
    log.info("")
    log.info(f"  Classification Report:")
    log.info(f"    {'Class':20s} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for cls in ["Not Delayed", "Delayed"]:
        r = report[cls]
        log.info(f"    {cls:20s} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1-score']:>10.3f}")
    log.info(f"    {'Accuracy':20s} {'':>10} {'':>10} {report['accuracy']:>10.3f}")

    # Feature importance (from GradientBoosting component)
    gb_model = model.named_estimators_["gb"]
    importances = gb_model.feature_importances_
    importances = importances / importances.sum()  # normalize
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    log.info("")
    log.info("  Top 15 Features:")
    for feat, imp in feat_imp[:15]:
        bar = "█" * int(imp * 100)
        log.info(f"    {feat:35s} {imp:.4f}  {bar}")

    # Save results
    results = {
        "auc_roc": round(auc_roc, 4),
        "average_precision": round(avg_precision, 4),
        "f1_delayed": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importance": {feat: round(float(imp), 4) for feat, imp in feat_imp},
        "train_size": int(len(y_test) + len(X_test)),
        "test_size": int(len(y_test)),
        "delayed_in_test": int(y_test.sum()),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Results saved → {RESULTS_PATH}")

    return results


# ═════════════════════════════════════════════════════════
# STEP 6: Save model
# ═════════════════════════════════════════════════════════
def save_model(model, feature_names):
    """Save trained model with metadata."""
    payload = {
        "model": model,
        "feature_names": feature_names,
        "model_type": "XGBClassifier",
        "target": "Delayed vs All",
        "train_years": [2021, 2022],
        "test_year": 2023,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    log.info(f"  Model saved → {MODEL_PATH}")
    log.info(f"  Size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")


# ═════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("Phase 3: Training ML Model...")
    log.info("=" * 60)

    # Step 1: Load data
    df = load_structured_features()

    # Step 2: Add RAG features from ChromaDB
    df = add_rag_features(df)

    # Step 3: Prepare features
    X_train, X_test, y_train, y_test, feature_names = prepare_features(df)

    # Step 4: Train
    model = train_model(X_train, y_train)

    # Step 5: Evaluate
    evaluate_model(model, X_test, y_test, feature_names)

    # Step 6: Save
    save_model(model, feature_names)

    log.info("")
    log.info("Phase 3 complete!")
