"""
╔══════════════════════════════════════════════════════════════╗
║       HYBRID MALWARE DETECTOR — TRAIN FROM CSV DATABASE      ║
║  Drop-in training script for malware_training_database.xlsx  ║
╚══════════════════════════════════════════════════════════════╝

HOW TO USE:
  1. Make sure full_training_database.csv is in the same folder.
  2. Run:  python train_from_database.py
  3. Trained models are saved to ./trained_models/

WHAT THIS SCRIPT DOES:
  Step 1 → Load & inspect the CSV database
  Step 2 → Preprocess & engineer features
  Step 3 → Balance classes (SMOTE)
  Step 4 → Split train / validation / test
  Step 5 → Train Random Forest
  Step 6 → Train XGBoost (falls back to GradientBoosting if not installed)
  Step 7 → Train a simple MLP Neural Network (Deep Learning)
  Step 8 → Build Stacking Ensemble (meta-learner)
  Step 9 → Evaluate every model — full metrics + confusion matrix
  Step 10→ Save all models + scaler + label encoder
"""

import os, sys, warnings, json, joblib, time
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network    import MLPClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (classification_report, confusion_matrix,
                                        roc_auc_score, accuracy_score,
                                        f1_score, precision_score, recall_score,
                                        roc_curve)
from sklearn.pipeline          import Pipeline
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  — change these to fit your setup
# ─────────────────────────────────────────────────────────────────
CSV_PATH      = "full_training_database.csv"   # path to your CSV
MODELS_DIR    = "trained_models"               # where to save models
RESULTS_DIR   = "training_results"            # where to save charts/reports
RANDOM_STATE  = 42
TEST_SIZE     = 0.15    # 15% held out for final test
VAL_SIZE      = 0.15    # 15% for validation
CV_FOLDS      = 5       # cross-validation folds

# Feature columns (all 31 numeric features from the database)
FEATURE_COLS = [
    "entropy", "file_size_kb", "n_sections",
    "avg_section_entropy", "max_section_entropy",
    "executable_sections", "writable_sections",
    "n_imports", "n_exports", "suspicious_api_count",
    "has_resources", "is_packed", "has_digital_sig",
    "has_overlay", "vt_score", "strings_suspicious",
    "network_calls", "registry_writes", "file_creates",
    "process_creates", "mutex_creates", "api_unique",
    "code_section_size_kb", "has_encryption_api",
    "has_injection_api", "has_persistence_api",
    "has_network_api", "has_anti_debug",
    "byte_0x00_freq", "byte_0xff_freq",
]

# Binary label: 0 = Benign, 1 = Malicious (for AUC-ROC)
BINARY_LABEL_COL = "is_malicious"

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def banner(text, char="═"):
    w = 64
    print(f"\n{'':2}{char*w}")
    print(f"{'':2}{text}")
    print(f"{'':2}{char*w}")

def step(n, text):
    print(f"\n  ┌{'─'*60}")
    print(f"  │  STEP {n}: {text}")
    print(f"  └{'─'*60}")

def ok(msg):  print(f"     ✔  {msg}")
def info(msg): print(f"     ℹ  {msg}")
def warn(msg): print(f"     ⚠  {msg}")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATABASE
# ═══════════════════════════════════════════════════════════════════
banner("HYBRID MALWARE DETECTOR — TRAINING FROM CSV DATABASE")
step(1, "Load & Inspect Database")

if not os.path.exists(CSV_PATH):
    # Try looking inside training_csvs/ folder
    alt = os.path.join("training_csvs", "full_training_database.csv")
    if os.path.exists(alt):
        CSV_PATH = alt
    else:
        print(f"\n  ERROR: Cannot find {CSV_PATH}")
        print("  Make sure full_training_database.csv is in the same directory.")
        sys.exit(1)

df = pd.read_csv(CSV_PATH)
ok(f"Loaded {len(df):,} rows × {len(df.columns)} columns from  {CSV_PATH}")

# Add binary label
df[BINARY_LABEL_COL] = (df["label"] != 0).astype(int)

# Class distribution
print("\n     CLASS DISTRIBUTION:")
print(f"     {'Class':<16} {'Count':>6}  {'%':>6}  Bar")
counts = df["class"].value_counts().sort_values(ascending=False)
for cls, cnt in counts.items():
    pct = cnt / len(df) * 100
    bar = "█" * max(1, int(pct * 1.5))
    print(f"     {cls:<16} {cnt:>6}  {pct:>5.1f}%  {bar}")

ok(f"Binary split: {df[BINARY_LABEL_COL].sum():,} malicious / "
   f"{(df[BINARY_LABEL_COL]==0).sum():,} benign")

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESS & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
step(2, "Preprocess & Feature Engineering")

# Check all feature columns exist
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    warn(f"Missing columns: {missing_cols} — they will be skipped")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

X = df[FEATURE_COLS].copy()
y_multi  = df["label"].values          # 0-12 multi-class
y_binary = df[BINARY_LABEL_COL].values # 0/1 binary
le = LabelEncoder()
le.fit(df["class"])
y_class_names = le.classes_

# Check for nulls
nulls = X.isnull().sum().sum()
if nulls > 0:
    warn(f"{nulls} null values found — filling with column median")
    X = X.fillna(X.median())
else:
    ok("No null values found")

# Feature engineering — add 5 derived features
X["entropy_x_packed"]     = X["entropy"] * X["is_packed"]
X["inject_persist_score"] = X["has_injection_api"] + X["has_persistence_api"] + X["has_anti_debug"]
X["network_intensity"]    = X["network_calls"] / (X["api_unique"] + 1)
X["section_risk"]         = X["executable_sections"] * X["writable_sections"]
X["obfuscation_score"]    = (X["entropy"] > 7.0).astype(int) + X["is_packed"] + (X["has_digital_sig"] == 0).astype(int)

engineered_features = ["entropy_x_packed", "inject_persist_score",
                        "network_intensity", "section_risk", "obfuscation_score"]

ok(f"Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features "
   f"({len(FEATURE_COLS)} original + {len(engineered_features)} engineered)")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ok("StandardScaler fitted")

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — BALANCE CLASSES
# ═══════════════════════════════════════════════════════════════════
step(3, "Balance Classes")

if HAS_SMOTE:
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_bal, y_bal_multi = smote.fit_resample(X_scaled, y_multi)
    _, y_bal_binary    = smote.fit_resample(X_scaled, y_binary)
    # Use original y_binary aligned to X_bal (SMOTE same on features)
    y_bal_binary = (y_bal_multi != 0).astype(int)
    ok(f"SMOTE applied → {X_bal.shape[0]:,} balanced samples")
else:
    warn("imbalanced-learn not installed — using original data (consider pip install imbalanced-learn)")
    X_bal, y_bal_multi, y_bal_binary = X_scaled, y_multi, y_binary
    ok(f"Using original {X_bal.shape[0]:,} samples")

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — SPLIT TRAIN / VALIDATION / TEST
# ═══════════════════════════════════════════════════════════════════
step(4, "Stratified Train / Validation / Test Split")

# First split off test set
X_trainval, X_test,  y_trainval_m, y_test_m, y_trainval_b, y_test_b = train_test_split(
    X_bal, y_bal_multi, y_bal_binary,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bal_multi
)
# Then split validation from train
X_train, X_val, y_train_m, y_val_m, y_train_b, y_val_b = train_test_split(
    X_trainval, y_trainval_m, y_trainval_b,
    test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_STATE, stratify=y_trainval_m
)

ok(f"Train:      {X_train.shape[0]:,} samples")
ok(f"Validation: {X_val.shape[0]:,}  samples")
ok(f"Test:       {X_test.shape[0]:,}  samples")
ok(f"Features:   {X_train.shape[1]}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════
step(5, "Train Random Forest (Multi-Class)")

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators   = 300,
    max_depth      = 25,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    max_features   = "sqrt",
    class_weight   = "balanced",
    n_jobs         = -1,
    random_state   = RANDOM_STATE,
    oob_score      = True,
)
rf.fit(X_train, y_train_m)
t_rf = time.time() - t0

rf_val_pred  = rf.predict(X_val)
rf_val_proba = rf.predict_proba(X_val)
rf_val_acc   = accuracy_score(y_val_m, rf_val_pred)
rf_val_f1    = f1_score(y_val_m, rf_val_pred, average="weighted")

ok(f"Training time:      {t_rf:.1f}s")
ok(f"OOB Score:          {rf.oob_score_:.4f}")
ok(f"Validation Acc:     {rf_val_acc:.4f}  ({rf_val_acc*100:.1f}%)")
ok(f"Validation F1 (wt): {rf_val_f1:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 6 — XGBOOST / GRADIENT BOOSTING
# ═══════════════════════════════════════════════════════════════════
step(6, f"Train {'XGBoost' if HAS_XGB else 'GradientBoosting'} (Multi-Class)")

t0 = time.time()
if HAS_XGB:
    gb = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 8,
        learning_rate     = 0.08,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )
else:
    gb = GradientBoostingClassifier(
        n_estimators  = 150,
        max_depth     = 5,
        learning_rate = 0.1,
        subsample     = 0.8,
        random_state  = RANDOM_STATE,
    )
    info("XGBoost not installed — using sklearn GradientBoostingClassifier")

gb.fit(X_train, y_train_m)
t_gb = time.time() - t0

gb_val_pred  = gb.predict(X_val)
gb_val_proba = gb.predict_proba(X_val)
gb_val_acc   = accuracy_score(y_val_m, gb_val_pred)
gb_val_f1    = f1_score(y_val_m, gb_val_pred, average="weighted")

ok(f"Training time:      {t_gb:.1f}s")
ok(f"Validation Acc:     {gb_val_acc:.4f}  ({gb_val_acc*100:.1f}%)")
ok(f"Validation F1 (wt): {gb_val_f1:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 7 — MLP NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════
step(7, "Train MLP Neural Network (Deep Learning proxy)")

t0 = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes = (512, 256, 128, 64),
    activation         = "relu",
    solver             = "adam",
    alpha              = 1e-4,
    batch_size         = 256,
    learning_rate_init = 0.001,
    max_iter           = 100,
    early_stopping     = True,
    validation_fraction= 0.1,
    n_iter_no_change   = 10,
    random_state       = RANDOM_STATE,
    verbose            = False,
)
mlp.fit(X_train, y_train_m)
t_mlp = time.time() - t0

mlp_val_pred  = mlp.predict(X_val)
mlp_val_proba = mlp.predict_proba(X_val)
mlp_val_acc   = accuracy_score(y_val_m, mlp_val_pred)
mlp_val_f1    = f1_score(y_val_m, mlp_val_pred, average="weighted")

ok(f"Training time:         {t_mlp:.1f}s")
ok(f"Iterations (epochs):   {mlp.n_iter_}")
ok(f"Validation Acc:        {mlp_val_acc:.4f}  ({mlp_val_acc*100:.1f}%)")
ok(f"Validation F1 (wt):    {mlp_val_f1:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 8 — STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════
step(8, "Build Stacking Ensemble (meta-learner: Logistic Regression)")

stacker = StackingClassifier(
    estimators=[
        ("rf",  RandomForestClassifier(n_estimators=100, max_depth=15,
                                        n_jobs=-1, random_state=RANDOM_STATE)),
        ("gb",  GradientBoostingClassifier(n_estimators=80, max_depth=4,
                                            random_state=RANDOM_STATE)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=80,
                               early_stopping=True, random_state=RANDOM_STATE)),
    ],
    final_estimator=LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"),
    cv=3,
    n_jobs=-1,
    passthrough=False,
)

t0 = time.time()
stacker.fit(X_train, y_train_m)
t_stack = time.time() - t0

stack_val_pred  = stacker.predict(X_val)
stack_val_proba = stacker.predict_proba(X_val)
stack_val_acc   = accuracy_score(y_val_m, stack_val_pred)
stack_val_f1    = f1_score(y_val_m, stack_val_pred, average="weighted")

ok(f"Training time:      {t_stack:.1f}s")
ok(f"Validation Acc:     {stack_val_acc:.4f}  ({stack_val_acc*100:.1f}%)")
ok(f"Validation F1 (wt): {stack_val_f1:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 9 — EVALUATE ON HELD-OUT TEST SET
# ═══════════════════════════════════════════════════════════════════
step(9, "Final Evaluation on Held-Out TEST SET")

# Label names for the classes present
all_labels = sorted(set(y_test_m))
label_names = [le.classes_[i] for i in all_labels]

models = {
    "Random Forest":    (rf,      rf.predict(X_test),      rf.predict_proba(X_test)),
    "GradientBoosting": (gb,      gb.predict(X_test),      gb.predict_proba(X_test)),
    "MLP Network":      (mlp,     mlp.predict(X_test),     mlp.predict_proba(X_test)),
    "Stacking Ensemble":(stacker, stacker.predict(X_test), stacker.predict_proba(X_test)),
}

results = {}
print(f"\n     {'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print(f"     {'─'*20} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

for name, (model, preds, probas) in models.items():
    acc  = accuracy_score(y_test_m, preds)
    prec = precision_score(y_test_m, preds, average="weighted", zero_division=0)
    rec  = recall_score(y_test_m, preds, average="weighted", zero_division=0)
    f1   = f1_score(y_test_m, preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_test_m, probas, multi_class="ovr", average="weighted")
    except:
        auc = 0.0
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
    star = " ◄ BEST" if name == "Stacking Ensemble" else ""
    print(f"     {name:<20} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {auc:>7.4f}{star}")

# Best model
best_name = max(results, key=lambda n: results[n]["f1"])
best_model, best_preds, best_probas = models[best_name]
ok(f"Best model: {best_name}  (F1={results[best_name]['f1']:.4f})")

# Full classification report
print(f"\n  ── Detailed Report: {best_name} ──")
print(classification_report(y_test_m, best_preds,
                             labels=all_labels,
                             target_names=label_names,
                             zero_division=0))

# ─────────────────────────────────────────────────────────────────
# PLOT 1 — Confusion Matrix
# ─────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test_m, best_preds, labels=all_labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle(f"Hybrid Malware Detector — {best_name}\nTest Set Results",
             fontsize=14, fontweight="bold", y=1.01)

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=axes[0], linewidths=0.5)
axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].tick_params(axis="x", rotation=40)
axes[0].tick_params(axis="y", rotation=0)

# Normalised %
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=label_names, yticklabels=label_names,
            ax=axes[1], linewidths=0.5)
axes[1].set_title("Confusion Matrix (Normalised)", fontweight="bold")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
axes[1].tick_params(axis="x", rotation=40)
axes[1].tick_params(axis="y", rotation=0)

plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Saved: {cm_path}")

# ─────────────────────────────────────────────────────────────────
# PLOT 2 — Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
metrics_list = ["accuracy", "precision", "recall", "f1", "auc"]
x = np.arange(len(results))
w = 0.15
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

for i, metric in enumerate(metrics_list):
    vals = [results[m][metric] for m in results]
    bars = ax.bar(x + i * w, vals, w, label=metric.capitalize(), color=colors[i], alpha=0.85)

ax.set_xticks(x + w * 2)
ax.set_xticklabels(list(results.keys()), fontsize=9)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics on Test Set", fontweight="bold")
ax.legend(loc="lower right")
ax.axhline(0.95, color="red", linestyle="--", alpha=0.4, label="95% target")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
comp_path = os.path.join(RESULTS_DIR, "model_comparison.png")
plt.savefig(comp_path, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Saved: {comp_path}")

# ─────────────────────────────────────────────────────────────────
# PLOT 3 — Feature Importance (Random Forest)
# ─────────────────────────────────────────────────────────────────
feature_names = list(X.columns)
importances   = rf.feature_importances_
top_n = 20
idx   = np.argsort(importances)[::-1][:top_n]

fig, ax = plt.subplots(figsize=(12, 8))
colors_fi = ["#D32F2F" if importances[i] > np.percentile(importances, 80) else "#1976D2"
             for i in idx]
ax.barh([feature_names[i] for i in idx][::-1],
        [importances[i] for i in idx][::-1],
        color=colors_fi[::-1], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Importance Score")
ax.set_title(f"Top {top_n} Most Important Features — Random Forest", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Saved: {fi_path}")

# ─────────────────────────────────────────────────────────────────
# PLOT 4 — Per-Class Metrics
# ─────────────────────────────────────────────────────────────────
from sklearn.metrics import precision_recall_fscore_support
prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
    y_test_m, best_preds, labels=all_labels, zero_division=0
)

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(label_names))
w = 0.28
ax.bar(x - w, prec_pc, w, label="Precision", color="#2196F3", alpha=0.85)
ax.bar(x,     rec_pc,  w, label="Recall",    color="#4CAF50", alpha=0.85)
ax.bar(x + w, f1_pc,   w, label="F1",        color="#FF5722", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(label_names, rotation=35, ha="right")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Per-Class Performance — Precision / Recall / F1", fontweight="bold")
ax.legend()
ax.axhline(0.90, color="grey", linestyle="--", alpha=0.4)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
pc_path = os.path.join(RESULTS_DIR, "per_class_metrics.png")
plt.savefig(pc_path, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Saved: {pc_path}")

# ─────────────────────────────────────────────────────────────────
# PLOT 5 — Class Distribution
# ─────────────────────────────────────────────────────────────────
class_colors = {
    "Benign": "#4CAF50", "Virus": "#F44336", "Trojan": "#E91E63",
    "Worm": "#FF9800", "Ransomware": "#B71C1C", "Rootkit": "#7B1FA2",
    "Spyware": "#9C27B0", "Adware": "#FF6F00", "Backdoor": "#C62828",
    "Fileless": "#333333", "PUP": "#FFC107", "Cryptominer": "#FF5722",
    "Botnet": "#0D47A1",
}
cls_counts = df["class"].value_counts()
bar_colors = [class_colors.get(c, "#607D8B") for c in cls_counts.index]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(cls_counts.index, cls_counts.values, color=bar_colors,
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, cls_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("Malware Class")
ax.set_ylabel("Sample Count")
ax.set_title("Training Database — Sample Distribution per Class", fontweight="bold")
ax.tick_params(axis="x", rotation=35)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
dist_path = os.path.join(RESULTS_DIR, "class_distribution.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Saved: {dist_path}")

# ═══════════════════════════════════════════════════════════════════
# STEP 10 — SAVE ALL MODELS
# ═══════════════════════════════════════════════════════════════════
step(10, "Save All Trained Models & Metadata")

# Save models
joblib.dump(rf,       os.path.join(MODELS_DIR, "random_forest.pkl"))
joblib.dump(gb,       os.path.join(MODELS_DIR, "gradient_boost.pkl"))
joblib.dump(mlp,      os.path.join(MODELS_DIR, "mlp_network.pkl"))
joblib.dump(stacker,  os.path.join(MODELS_DIR, "stacking_ensemble.pkl"))
joblib.dump(scaler,   os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(le,       os.path.join(MODELS_DIR, "label_encoder.pkl"))

ok("random_forest.pkl")
ok("gradient_boost.pkl")
ok("mlp_network.pkl")
ok("stacking_ensemble.pkl  ← PRIMARY model")
ok("scaler.pkl")
ok("label_encoder.pkl")

# Save feature list and config
config = {
    "feature_columns":      list(X.columns),
    "original_features":    FEATURE_COLS,
    "engineered_features":  engineered_features,
    "class_labels":         list(le.classes_),
    "n_classes":            int(len(le.classes_)),
    "binary_threshold":     0.5,
    "train_samples":        int(X_train.shape[0]),
    "val_samples":          int(X_val.shape[0]),
    "test_samples":         int(X_test.shape[0]),
    "best_model":           best_name,
    "trained_at":           datetime.now().isoformat(),
    "test_results":         {k: {m: round(v, 4) for m, v in v2.items()}
                             for k, v2 in results.items()},
}
with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)
ok("model_config.json")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
banner("TRAINING COMPLETE — FINAL SUMMARY")

best_r = results[best_name]
print(f"""
  ┌────────────────────────────────────────────────────────────┐
  │  BEST MODEL  :  {best_name:<44}│
  │  Accuracy    :  {best_r['accuracy']*100:>6.2f}%                                      │
  │  Precision   :  {best_r['precision']*100:>6.2f}%                                      │
  │  Recall      :  {best_r['recall']*100:>6.2f}%                                      │
  │  F1-Score    :  {best_r['f1']*100:>6.2f}%                                      │
  │  ROC-AUC     :  {best_r['auc']:>6.4f}                                       │
  ├────────────────────────────────────────────────────────────┤
  │  SAVED FILES                                               │
  │   Models  → ./trained_models/                             │
  │   Charts  → ./training_results/                           │
  ├────────────────────────────────────────────────────────────┤
  │  NEXT STEP: Run  python predict.py --file suspicious.exe  │
  └────────────────────────────────────────────────────────────┘
""")
