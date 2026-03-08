# ============================================================
# STEP 6 — Error Analysis & Feature Inspection
# ============================================================
# Run: python step6_error_analysis.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

MODELS_DIR = "data"          # cleaned.csv lives here
ML_DIR     = "models"
OUTPUT_DIR = "outputs"

# ── 6.1  Reload raw text so we can read misclassified rows ─
df_clean = pd.read_csv(os.path.join(MODELS_DIR, "cleaned.csv"))

X_test  = joblib.load(os.path.join(ML_DIR, "X_test.pkl"))
y_test  = joblib.load(os.path.join(ML_DIR, "y_test.pkl"))
tfidf   = joblib.load(os.path.join(ML_DIR, "tfidf.pkl"))

# We'll analyse the best model — Linear SVM
model   = joblib.load(os.path.join(ML_DIR, "Linear_SVM.pkl"))

y_pred  = model.predict(X_test)
y_test_arr = y_test.reset_index(drop=True)


# ── 6.2  Misclassified articles ────────────────────────────
# Align cleaned text to test indices
df_test = df_clean.loc[y_test.index].reset_index(drop=True)

error_mask  = y_pred != y_test_arr
errors_df   = df_test[error_mask].copy()
errors_df["predicted"] = y_pred[error_mask]
errors_df["actual"]    = y_test_arr[error_mask].values

errors_df["pred_label"] = errors_df["predicted"].map({0: "FAKE", 1: "REAL"})
errors_df["true_label"] = errors_df["actual"].map({0: "FAKE", 1: "REAL"})

print(f"Total misclassified : {error_mask.sum()} / {len(y_test)}")
print(f"  False Positives (Real predicted as Fake) : {((y_pred==0) & (y_test_arr==1)).sum()}")
print(f"  False Negatives (Fake predicted as Real) : {((y_pred==1) & (y_test_arr==0)).sum()}")
print()

# Show 5 sample errors
print("SAMPLE MISCLASSIFIED ARTICLES")
print("─" * 60)
for _, row in errors_df.head(5).iterrows():
    print(f"  True: {row.true_label:<4}  |  Predicted: {row.pred_label}")
    print(f"  Text: {row.clean_text[:140]} …")
    print()

errors_df.to_csv(os.path.join(OUTPUT_DIR, "06_errors.csv"), index=False)
print(f"[✓] Full error list saved → {OUTPUT_DIR}/06_errors.csv")


# ── 6.3  Top predictive words (Logistic Regression) ────────
# LR has .coef_ which is directly interpretable
lr_model = joblib.load(os.path.join(ML_DIR, "Logistic_Regression.pkl"))
feature_names = np.array(tfidf.get_feature_names_out())
coefs         = lr_model.coef_[0]

top_n = 20
top_fake_idx = np.argsort(coefs)[:top_n]          # most negative → Fake
top_real_idx = np.argsort(coefs)[::-1][:top_n]    # most positive → Real

top_fake_words  = feature_names[top_fake_idx]
top_fake_scores = coefs[top_fake_idx]
top_real_words  = feature_names[top_real_idx]
top_real_scores = coefs[top_real_idx]

print("\nTOP 20 WORDS → FAKE")
for w, s in zip(top_fake_words, top_fake_scores):
    print(f"  {w:<30} {s:.4f}")

print("\nTOP 20 WORDS → REAL")
for w, s in zip(top_real_words, top_real_scores):
    print(f"  {w:<30} {s:.4f}")


# ── 6.4  Feature importance bar chart ──────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Logistic Regression — Top Predictive Words", fontsize=13, fontweight="bold")

ax1.barh(top_fake_words[::-1], np.abs(top_fake_scores[::-1]), color="#e74c3c")
ax1.set_title("Top words → FAKE")
ax1.set_xlabel("|Coefficient|")

ax2.barh(top_real_words[::-1], top_real_scores[::-1], color="#2ecc71")
ax2.set_title("Top words → REAL")
ax2.set_xlabel("Coefficient")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_feature_importance.png"), dpi=150)
plt.show()
print(f"\n[✓] Feature importance plot saved → {OUTPUT_DIR}/06_feature_importance.png")


# ── 6.5  Improvement ideas printout ────────────────────────
print("""
╔══════════════════════════════════════════════════════╗
║         IMPROVEMENT IDEAS                           ║
╠══════════════════════════════════════════════════════╣
║  1. Strip source/publisher metadata to avoid        ║
║     data leakage (common in this dataset).          ║
║  2. Try ngram_range=(1,3) and max_features=100k.    ║
║  3. Tune SVM C via GridSearchCV.                    ║
║  4. Fine-tune DistilBERT for +0.3–0.5% F1.         ║
║  5. Use SHAP for deeper explainability.             ║
╚══════════════════════════════════════════════════════╝
""")
