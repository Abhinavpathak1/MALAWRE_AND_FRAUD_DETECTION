"""
train_model.py — Trains and saves the full Hybrid ML + Deep Learning ensemble.

Models trained:
  1. TF-IDF + Logistic Regression         (classical NLP)
  2. Char N-Gram TF-IDF + LinearSVC       (deep character patterns)
  3. Random Forest (300 trees)            (ensemble tree)
  4. Gradient Boosting (200 estimators)   (boosted ensemble)
  5. MLP Neural Network (deep learning)   (2 hidden layers)
  6. Gaussian Naive Bayes                 (probabilistic baseline)
  + Weighted Voting Ensemble              (final combined model)

Run:  python train_model.py
Output: models/  directory with all saved model files + training_report.txt
"""

import os
import time
import json
import joblib
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)

from dataset import TEXTS, LABELS
from features import FeatureExtractor, FEATURE_NAMES

# ──────────────────────────────────────────────────────────────
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 65)
print("  THREAT·SCAN·AI — Model Training Pipeline")
print("=" * 65)
print(f"  Dataset: {len(TEXTS)} samples  |  Fraud: {sum(LABELS)}  |  Legit: {len(LABELS)-sum(LABELS)}")
print()

# ──────────────────────────────────────────────────────────────
# DATA PREPARATION
# ──────────────────────────────────────────────────────────────

print("[1/8] Preparing data & extracting features...")

fe = FeatureExtractor()
X_hand   = fe.transform(TEXTS)              # handcrafted 36-dim features
X_text   = np.array(TEXTS)                  # raw text for TF-IDF models
y        = np.array(LABELS)

# Train / test split (stratified)
(X_text_tr, X_text_te,
 X_hand_tr, X_hand_te,
 y_tr, y_te) = train_test_split(
    X_text, X_hand, y,
    test_size=0.22, random_state=42, stratify=y
)

print(f"  Train: {len(y_tr)} samples  |  Test: {len(y_te)} samples")

# ──────────────────────────────────────────────────────────────
# METRICS HELPER
# ──────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred.astype(float)
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n  ┌─ {name}")
    print(f"  │  Accuracy:  {acc*100:.1f}%")
    print(f"  │  F1 Score:  {f1*100:.1f}%")
    print(f"  │  Precision: {prec*100:.1f}%")
    print(f"  │  Recall:    {rec*100:.1f}%")
    print(f"  │  ROC-AUC:   {auc:.4f}")
    print(f"  │  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    return {"name": name, "acc": acc, "f1": f1, "precision": prec,
            "recall": rec, "auc": auc}


# ──────────────────────────────────────────────────────────────
# MODEL 1 — TF-IDF + Logistic Regression
# ──────────────────────────────────────────────────────────────

print("\n[2/8] Training Model 1: TF-IDF + Logistic Regression...")
t0 = time.time()

m1 = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        min_df=1,
        analyzer="word",
        strip_accents="unicode",
    )),
    ("clf", LogisticRegression(
        C=2.0, max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )),
])
m1.fit(X_text_tr, y_tr)
m1_metrics = evaluate(m1, X_text_te, y_te, "TF-IDF + Logistic Regression")
print(f"  └─ Trained in {time.time()-t0:.2f}s")
joblib.dump(m1, f"{MODELS_DIR}/m1_tfidf_lr.pkl")


# ──────────────────────────────────────────────────────────────
# MODEL 2 — Char N-Gram + LinearSVC (calibrated)
# ──────────────────────────────────────────────────────────────

print("\n[3/8] Training Model 2: Char N-Gram TF-IDF + LinearSVC...")
t0 = time.time()

svc_base = Pipeline([
    ("char_tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=4000,
        sublinear_tf=True,
    )),
    ("clf", LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")),
])

m2 = CalibratedClassifierCV(svc_base, cv=3, method="sigmoid")
m2.fit(X_text_tr, y_tr)
m2_metrics = evaluate(m2, X_text_te, y_te, "Char N-Gram + SVC (calibrated)")
print(f"  └─ Trained in {time.time()-t0:.2f}s")
joblib.dump(m2, f"{MODELS_DIR}/m2_chargram_svc.pkl")


# ──────────────────────────────────────────────────────────────
# MODEL 3 — Random Forest on handcrafted features
# ──────────────────────────────────────────────────────────────

print("\n[4/8] Training Model 3: Random Forest (300 trees)...")
t0 = time.time()

m3 = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
m3.fit(X_hand_tr, y_tr)
m3_metrics = evaluate(m3, X_hand_te, y_te, "Random Forest (300 trees)")
print(f"  └─ Trained in {time.time()-t0:.2f}s")

# Feature importance
importances = m3.feature_importances_
top_idx = np.argsort(importances)[::-1][:8]
print("  │  Top features:")
for i in top_idx:
    print(f"  │    {FEATURE_NAMES[i]:30s} {importances[i]:.4f}")
joblib.dump(m3, f"{MODELS_DIR}/m3_random_forest.pkl")


# ──────────────────────────────────────────────────────────────
# MODEL 4 — Gradient Boosting on handcrafted features
# ──────────────────────────────────────────────────────────────

print("\n[5/8] Training Model 4: Gradient Boosting (200 estimators)...")
t0 = time.time()

m4 = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.85,
    min_samples_split=2,
    random_state=42,
)
m4.fit(X_hand_tr, y_tr)
m4_metrics = evaluate(m4, X_hand_te, y_te, "Gradient Boosting (200 est.)")
print(f"  └─ Trained in {time.time()-t0:.2f}s")
joblib.dump(m4, f"{MODELS_DIR}/m4_gradient_boost.pkl")


# ──────────────────────────────────────────────────────────────
# MODEL 5 — MLP Neural Network (Deep Learning)
# ──────────────────────────────────────────────────────────────

print("\n[6/8] Training Model 5: MLP Neural Network (Deep Learning)...")
t0 = time.time()

# Scale features for neural net
scaler = StandardScaler()
X_hand_tr_sc = scaler.fit_transform(X_hand_tr)
X_hand_te_sc = scaler.transform(X_hand_te)

m5 = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),   # 3 hidden layers
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
    batch_size=16,
)
m5.fit(X_hand_tr_sc, y_tr)
m5_metrics = evaluate(m5, X_hand_te_sc, y_te, "MLP Neural Network (3-layer)")
print(f"  │  Layers: Input(36) → 128 → 64 → 32 → Output(2)")
print(f"  │  Activation: ReLU | Solver: Adam | Epochs: {m5.n_iter_}")
print(f"  └─ Trained in {time.time()-t0:.2f}s")
joblib.dump(m5, f"{MODELS_DIR}/m5_mlp_neural_net.pkl")
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")


# ──────────────────────────────────────────────────────────────
# MODEL 6 — Naive Bayes (TF-IDF word counts)
# ──────────────────────────────────────────────────────────────

print("\n[7/8] Training Model 6: Multinomial Naive Bayes (TF-IDF)...")
t0 = time.time()

m6 = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        min_df=1,
    )),
    ("clf", MultinomialNB(alpha=0.5)),
])
# MinMaxScaler to keep values non-negative for MultinomialNB
# Already ensured by TF-IDF (values >= 0)
m6.fit(X_text_tr, y_tr)
m6_metrics = evaluate(m6, X_text_te, y_te, "Multinomial Naive Bayes")
print(f"  └─ Trained in {time.time()-t0:.2f}s")
joblib.dump(m6, f"{MODELS_DIR}/m6_naive_bayes.pkl")


# ──────────────────────────────────────────────────────────────
# CROSS-VALIDATION on best models
# ──────────────────────────────────────────────────────────────

print("\n[8/8] Cross-Validation (5-fold stratified)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model, X, name in [
    (m1, X_text, "TF-IDF LR"),
    (m3, X_hand, "Random Forest"),
    (m4, X_hand, "Grad Boost"),
    (m5, scaler.transform(X_hand), "MLP Neural Net"),
]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"  {name:22s}  CV F1: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")


# ──────────────────────────────────────────────────────────────
# SAVE FEATURE EXTRACTOR + METADATA
# ──────────────────────────────────────────────────────────────

joblib.dump(fe, f"{MODELS_DIR}/feature_extractor.pkl")

metadata = {
    "n_train":   int(len(y_tr)),
    "n_test":    int(len(y_te)),
    "n_total":   int(len(y)),
    "n_fraud":   int(sum(y)),
    "n_legit":   int(len(y) - sum(y)),
    "feature_names": FEATURE_NAMES,
    "models": {
        "m1": {"file": "m1_tfidf_lr.pkl",       "type": "text",  "desc": "TF-IDF + Logistic Regression"},
        "m2": {"file": "m2_chargram_svc.pkl",    "type": "text",  "desc": "Char N-Gram + SVC (calibrated)"},
        "m3": {"file": "m3_random_forest.pkl",   "type": "hand",  "desc": "Random Forest (300 trees)"},
        "m4": {"file": "m4_gradient_boost.pkl",  "type": "hand",  "desc": "Gradient Boosting (200 est.)"},
        "m5": {"file": "m5_mlp_neural_net.pkl",  "type": "hand_scaled", "desc": "MLP Neural Net (128→64→32)"},
        "m6": {"file": "m6_naive_bayes.pkl",     "type": "text",  "desc": "Multinomial Naive Bayes"},
    },
    "ensemble_weights": {"m1": 0.25, "m2": 0.20, "m3": 0.18,
                         "m4": 0.17, "m5": 0.12, "m6": 0.08},
    "metrics": {
        "m1": m1_metrics, "m2": m2_metrics, "m3": m3_metrics,
        "m4": m4_metrics, "m5": m5_metrics, "m6": m6_metrics,
    }
}

with open(f"{MODELS_DIR}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, default=str)

# ──────────────────────────────────────────────────────────────
# TRAINING REPORT
# ──────────────────────────────────────────────────────────────

all_metrics = [m1_metrics, m2_metrics, m3_metrics, m4_metrics, m5_metrics, m6_metrics]
avg_acc = np.mean([m["acc"] for m in all_metrics]) * 100
avg_f1  = np.mean([m["f1"]  for m in all_metrics]) * 100

report = f"""
═══════════════════════════════════════════════════════════════
  THREAT·SCAN·AI — TRAINING REPORT
═══════════════════════════════════════════════════════════════

Dataset
  Total samples : {len(y)}
  Fraud / Scam  : {sum(y)}
  Legitimate    : {len(y) - sum(y)}
  Train split   : {len(y_tr)} ({len(y_tr)/len(y)*100:.0f}%)
  Test split    : {len(y_te)} ({len(y_te)/len(y)*100:.0f}%)

Model Performance (held-out test set)
  {'Model':<35} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}
  {'─'*65}
"""
for m in all_metrics:
    report += (f"  {m['name']:<35} {m['acc']*100:>5.1f}% {m['f1']*100:>5.1f}% "
               f"{m['precision']*100:>5.1f}% {m['recall']*100:>5.1f}% {m['auc']:>6.3f}\n")
report += f"""
  Average Accuracy : {avg_acc:.1f}%
  Average F1 Score : {avg_f1:.1f}%

Models Saved to: models/
  m1_tfidf_lr.pkl          TF-IDF + Logistic Regression
  m2_chargram_svc.pkl      Char N-Gram + SVC (calibrated)
  m3_random_forest.pkl     Random Forest (300 trees)
  m4_gradient_boost.pkl    Gradient Boosting (200 estimators)
  m5_mlp_neural_net.pkl    MLP Neural Network (128→64→32)
  m6_naive_bayes.pkl       Multinomial Naive Bayes
  feature_extractor.pkl    Handcrafted feature transformer
  scaler.pkl               StandardScaler for neural network
  metadata.json            Model config & performance metrics

═══════════════════════════════════════════════════════════════
"""
print(report)

with open(f"{MODELS_DIR}/training_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("✓ All models saved successfully.")
print("  Run:  streamlit run app.py")
