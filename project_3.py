# ============================================================
# STEP 1 — Load & Explore Data
# ============================================================
# Place Fake.csv and True.csv inside the data/ folder before running.
# Run: python step1_load_explore.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1.1  Load raw CSVs ─────────────────────────────────────
fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
true_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

fake_df["label"] = 0   # 0 = Fake
true_df["label"] = 1   # 1 = Real

# Combine & shuffle
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Merge title + body into one column
df["text_full"] = df["title"].fillna("") + " " + df["text"].fillna("")

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Total articles : {len(df):,}")
print(f"Fake           : {(df.label == 0).sum():,}")
print(f"Real           : {(df.label == 1).sum():,}")
print(f"Columns        : {list(df.columns)}")
print(f"Missing values :\n{df.isnull().sum()}\n")


# ── 1.2  Text length distribution ──────────────────────────
df["text_len"] = df["text_full"].str.split().str.len()

print("TEXT LENGTH STATS")
print(df.groupby("label")["text_len"].describe().round(1))
print()


# ── 1.3  Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Exploratory Data Analysis", fontsize=14, fontweight="bold")

# Class distribution
axes[0].bar(["Fake", "Real"],
            [fake_df.shape[0], true_df.shape[0]],
            color=["#e74c3c", "#2ecc71"], edgecolor="black")
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for bar, val in zip(axes[0].patches, [fake_df.shape[0], true_df.shape[0]]):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 100, f"{val:,}",
                 ha="center", fontsize=11)

# Text length histogram
for label, color, name in [(0, "#e74c3c", "Fake"), (1, "#2ecc71", "Real")]:
    axes[1].hist(df[df.label == label]["text_len"],
                 bins=60, alpha=0.6, color=color, label=name)
axes[1].set_title("Article Length (words)")
axes[1].set_xlabel("Word count")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_eda.png"), dpi=150)
plt.show()
print(f"[✓] Plot saved → {OUTPUT_DIR}/01_eda.png")

# ── 1.4  Save combined dataset ─────────────────────────────
df.to_csv(os.path.join(DATA_DIR, "combined.csv"), index=False)
print(f"[✓] Combined dataset saved → {DATA_DIR}/combined.csv")

# ============================================================
# STEP 2 — Text Preprocessing
# ============================================================
# Run: python step2_preprocess.py

import re
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only needed once)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

DATA_DIR = "data"

# ── 2.1  Setup ─────────────────────────────────────────────
stop_words  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()


# ── 2.2  Cleaning function ─────────────────────────────────
def clean_text(text: str) -> str:
    """
    Pipeline:
      - Lowercase
      - Remove URLs
      - Remove HTML tags
      - Keep only alphabetic characters
      - Tokenise
      - Remove stopwords & very short tokens
      - Lemmatize
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                      # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                   # non-alpha
    text = re.sub(r"\s+", " ", text).strip()                 # extra spaces

    tokens = [
        lemmatizer.lemmatize(token)
        for token in text.split()
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(tokens)


# ── 2.3  Apply to dataset ──────────────────────────────────
print("Loading combined.csv …")
df = pd.read_csv(os.path.join(DATA_DIR, "combined.csv"))

print("Cleaning text — this may take ~1 minute …")
df["clean_text"] = df["text_full"].apply(clean_text)

# Sanity check — drop empty rows after cleaning
empty = df["clean_text"].str.strip().eq("").sum()
if empty:
    print(f"  ⚠  Dropping {empty} empty rows after cleaning")
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

print("\nSAMPLE (first 3 rows, truncated to 120 chars):")
for i, row in df.head(3).iterrows():
    label = "FAKE" if row.label == 0 else "REAL"
    print(f"  [{label}] {row.clean_text[:120]} …")

# ── 2.4  Save ──────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "cleaned.csv")
df[["clean_text", "label"]].to_csv(out_path, index=False)
print(f"\n[✓] Cleaned data saved → {out_path}")
print(f"    Rows: {len(df):,}")

# ============================================================
# STEP 3 — Vectorization (TF-IDF) + Train/Test Split
# ============================================================
# Run: python step3_vectorize.py

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 3.1  Load cleaned data ─────────────────────────────────
print("Loading cleaned.csv …")
df = pd.read_csv(os.path.join(DATA_DIR, "cleaned.csv"))
X  = df["clean_text"]
y  = df["label"]

print(f"  Total samples : {len(df):,}")
print(f"  Fake (0)      : {(y == 0).sum():,}")
print(f"  Real (1)      : {(y == 1).sum():,}\n")


# ── 3.2  Stratified train / test split ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # keeps class ratio in both splits
)

print(f"Train size : {len(X_train):,}")
print(f"Test size  : {len(X_test):,}\n")


# ── 3.3  TF-IDF vectorization ──────────────────────────────
#   max_features : vocabulary cap (50k covers most words)
#   ngram_range  : (1,2) = unigrams + bigrams
#   sublinear_tf : log(1 + tf) — dampens very frequent terms
#   min_df       : ignore terms that appear in fewer than 3 docs

tfidf = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    strip_accents="unicode",
    analyzer="word",
)

print("Fitting TF-IDF vectorizer …")
X_train_tfidf = tfidf.fit_transform(X_train)   # fit ONLY on train
X_test_tfidf  = tfidf.transform(X_test)         # transform test

print(f"  Train matrix : {X_train_tfidf.shape}")
print(f"  Test  matrix : {X_test_tfidf.shape}")


# ── 3.4  Persist everything ────────────────────────────────
joblib.dump(tfidf,         os.path.join(MODELS_DIR, "tfidf.pkl"))
joblib.dump(X_train_tfidf, os.path.join(MODELS_DIR, "X_train.pkl"))
joblib.dump(X_test_tfidf,  os.path.join(MODELS_DIR, "X_test.pkl"))
joblib.dump(y_train,       os.path.join(MODELS_DIR, "y_train.pkl"))
joblib.dump(y_test,        os.path.join(MODELS_DIR, "y_test.pkl"))

print("\n[✓] Saved to models/")
print("    tfidf.pkl  |  X_train.pkl  |  X_test.pkl  |  y_train.pkl  |  y_test.pkl")

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

# ============================================================
# PREDICT — Run on any new article
# ============================================================
# Usage:
#   python predict.py
#   python predict.py --text "Your article text here"
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
import argparse
import re
import joblib
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)


def predict(text: str):
    tfidf = joblib.load("models/tfidf.pkl")
    model = joblib.load("models/Linear_SVM.pkl")

    cleaned  = clean_text(text)
    vectored = tfidf.transform([cleaned])
    pred     = model.predict(vectored)[0]

    # Confidence via decision_function distance
    score = model.decision_function(vectored)[0]
    conf  = min(abs(score) / 3.0, 1.0)   # rough normalisation

    label = "REAL ✅" if pred == 1 else "FAKE ❌"
    print(f"\n  Prediction  : {label}")
    print(f"  Confidence  : {conf:.0%}")
    print(f"  Raw score   : {score:.4f}  (positive → Real, negative → Fake)\n")
    return pred, conf


# ── Demo texts ─────────────────────────────────────────────
SAMPLE_FAKE = (
    "BREAKING: Scientists confirm that 5G towers are spreading the virus "
    "and governments are hiding the truth. Share before they delete this!"
)

SAMPLE_REAL = (
    "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
    "as policymakers continue efforts to bring inflation back to the 2% target."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None,
                        help="Article text to classify")
    args = parser.parse_args()

    if args.text:
        predict(args.text)
    else:
        print("=" * 55)
        print("DEMO — SAMPLE FAKE ARTICLE")
        print("=" * 55)
        print(f"Text: {SAMPLE_FAKE[:80]} …")
        predict(SAMPLE_FAKE)

        print("=" * 55)
        print("DEMO — SAMPLE REAL ARTICLE")
        print("=" * 55)
        print(f"Text: {SAMPLE_REAL[:80]} …")
        predict(SAMPLE_REAL)

# ============================================================
# RUN ALL — Execute the full pipeline in sequence
# ============================================================
# Run: python run_all.py

import subprocess
import sys
import time

steps = [
    ("STEP 1 — Load & Explore",       "step1_load_explore.py"),
    ("STEP 2 — Preprocess Text",      "step2_preprocess.py"),
    ("STEP 3 — Vectorize (TF-IDF)",   "step3_vectorize.py"),
    ("STEP 4 — Train Models",         "step4_train.py"),
    ("STEP 5 — Evaluate",             "step5_evaluate.py"),
    ("STEP 6 — Error Analysis",       "step6_error_analysis.py"),
]

print("=" * 55)
print("  FAKE NEWS DETECTION — FULL PIPELINE")
print("=" * 55)

for label, script in steps:
    print(f"\n▶  {label}")
    print("─" * 55)
    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ✗  {script} failed. Fix the error above and re-run.")
        sys.exit(1)
    print(f"  ✓  Done in {elapsed:.1f}s")

print("\n" + "=" * 55)
print("  ALL STEPS COMPLETE")
print("  Outputs saved in:  outputs/")
print("  Models saved in:   models/")
print("=" * 55)