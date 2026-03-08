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
