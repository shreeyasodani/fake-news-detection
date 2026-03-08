# ============================================================
# STEP 5 — Evaluate Models (Confusion Matrix, F1, ROC-AUC)
# ============================================================
# Run: python step5_evaluate.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

MODELS_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 5.1  Load test data + trained models ───────────────────
X_test  = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
y_test  = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))

model_files = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Naive Bayes":         "Naive_Bayes.pkl",
    "Linear SVM":          "Linear_SVM.pkl",
}

models = {
    name: joblib.load(os.path.join(MODELS_DIR, fname))
    for name, fname in model_files.items()
}

CLASS_NAMES = ["Fake", "Real"]


# ── 5.2  Print classification reports ─────────────────────
summary_rows = []

print("=" * 58)
for name, model in models.items():
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    # ROC-AUC (LinearSVC uses decision_function, others use predict_proba)
    try:
        scores = model.decision_function(X_test)
    except AttributeError:
        scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)

    summary_rows.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(auc, 4),
    })

    print(f"\n{'─'*58}")
    print(f"  {name.upper()}")
    print(f"{'─'*58}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

summary_df = pd.DataFrame(summary_rows).set_index("Model")
print("\nSUMMARY TABLE")
print(summary_df.to_string())
summary_df.to_csv(os.path.join(OUTPUT_DIR, "05_model_summary.csv"))


# ── 5.3  Confusion matrices (all 3 models) ────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5, linecolor="black",
        annot_kws={"size": 13}
    )
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_confusion_matrices.png"), dpi=150)
plt.show()
print(f"\n[✓] Confusion matrices saved → {OUTPUT_DIR}/05_confusion_matrices.png")


# ── 5.4  ROC Curves ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#3498db", "#e67e22", "#e74c3c"]

for (name, model), color in zip(models.items(), colors):
    try:
        scores = model.decision_function(X_test)
    except AttributeError:
        scores = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.4f})", color=color, lw=2)

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_roc_curves.png"), dpi=150)
plt.show()
print(f"[✓] ROC curves saved → {OUTPUT_DIR}/05_roc_curves.png")
