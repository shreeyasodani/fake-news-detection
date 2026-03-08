# Fake News Detection — NLP Pipeline

Binary text classifier (Fake / Real) using TF-IDF + ML models.
Achieves **~99% F1-score** on the Kaggle Fake & Real News dataset.

---

## Project Structure

```
fake_news_detection/
│
├── data/                        ← put Fake.csv and True.csv here
├── models/                      ← saved models & vectorizer (auto-created)
├── outputs/                     ← plots and reports (auto-created)
│
├── step1_load_explore.py        ← EDA & class distribution
├── step2_preprocess.py          ← text cleaning & lemmatization
├── step3_vectorize.py           ← TF-IDF + train/test split
├── step4_train.py               ← train LR, NB, SVM with CV
├── step5_evaluate.py            ← confusion matrix, ROC, F1
├── step6_error_analysis.py      ← misclassified rows + feature importance
├── predict.py                   ← classify any new article
├── run_all.py                   ← run entire pipeline in one command
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Go to https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
and place **Fake.csv** and **True.csv** inside the `data/` folder.

### 3. Run the full pipeline
```bash
python run_all.py
```

Or run each step individually:
```bash
python step1_load_explore.py
python step2_preprocess.py
python step3_vectorize.py
python step4_train.py
python step5_evaluate.py
python step6_error_analysis.py
```

### 4. Predict on new text
```bash
# Demo mode
python predict.py

# Custom article
python predict.py --text "Paste your article text here"
```

---

## Pipeline Summary

| Step | Script | Output |
|------|--------|--------|
| 1 | step1_load_explore.py | `outputs/01_eda.png` |
| 2 | step2_preprocess.py | `data/cleaned.csv` |
| 3 | step3_vectorize.py | `models/tfidf.pkl`, `X_train.pkl`, etc. |
| 4 | step4_train.py | `models/*.pkl`, `outputs/04_cv_comparison.png` |
| 5 | step5_evaluate.py | `outputs/05_confusion_matrices.png`, `05_roc_curves.png`, `05_model_summary.csv` |
| 6 | step6_error_analysis.py | `outputs/06_errors.csv`, `06_feature_importance.png` |

---

## Model Results (approx.)

| Model | Accuracy | F1 |
|-------|----------|----|
| Logistic Regression | ~98.1% | ~98.0% |
| Naive Bayes | ~95.4% | ~95.3% |
| **Linear SVM** | **~99.2%** | **~99.2%** |

---

## Possible Improvements

- **Tune TF-IDF**: try `ngram_range=(1,3)`, `max_features=100_000`
- **GridSearchCV**: tune SVM `C` parameter (0.1 → 10)
- **DistilBERT**: fine-tune with HuggingFace `Trainer` API for best accuracy
- **SHAP**: explain predictions on individual articles
- **LIAR dataset**: multi-class labels for more nuanced detection
