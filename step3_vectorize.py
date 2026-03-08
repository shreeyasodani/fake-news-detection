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
