# ============================================================
# PREDICT — Run on any new article
# ============================================================
# Usage:
#   python predict.py
#   python predict.py --text "Your article text here"

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
