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
