# ============================================================
# STEP 4 — Train & Compare ML Models
# ============================================================
# Run: python step4_train.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import MultinomialNB
from sklearn.svm             import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

MODELS_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 4.1  Load artefacts from step 3 ───────────────────────
print("Loading data …")
X_train = joblib.load(os.path.join(MODELS_DIR, "X_train.pkl"))
y_train = joblib.load(os.path.join(MODELS_DIR, "y_train.pkl"))


# ── 4.2  Define classifiers ────────────────────────────────
classifiers = {
    "Logistic Regression": LogisticRegression(
        C=5, max_iter=1000, solver="lbfgs", n_jobs=-1
    ),
    "Naive Bayes": MultinomialNB(
        alpha=0.1               # Laplace smoothing
    ),
    "Linear SVM": LinearSVC(
        C=1.0, max_iter=2000, dual=True
    ),
}


# ── 4.3  5-fold cross-validation ───────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n5-FOLD CROSS-VALIDATION (F1 score)")
print("─" * 50)

cv_results = {}
for name, clf in classifiers.items():
    scores = cross_val_score(
        clf, X_train, y_train,
        cv=cv, scoring="f1", n_jobs=-1
    )
    cv_results[name] = scores
    print(f"  {name:<25}  {scores.mean():.4f} ± {scores.std():.4f}")

print()


# ── 4.4  Fit all models on full training set ───────────────
print("Training final models on full training set …")
trained = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    trained[name] = clf
    joblib.dump(clf, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))
    print(f"  [✓] {name} saved")


# ── 4.5  CV comparison bar chart ───────────────────────────
means  = [cv_results[n].mean() for n in classifiers]
stds   = [cv_results[n].std()  for n in classifiers]
labels = list(classifiers.keys())

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(labels, means, xerr=stds, color=["#3498db", "#e67e22", "#e74c3c"],
               capsize=5, edgecolor="black", height=0.5)
ax.set_xlim(0.90, 1.005)
ax.set_xlabel("F1 Score (5-fold CV)")
ax.set_title("Model Comparison — Cross-Validation F1")
for bar, val in zip(bars, means):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_cv_comparison.png"), dpi=150)
plt.show()
print(f"\n[✓] Plot saved → {OUTPUT_DIR}/04_cv_comparison.png")
