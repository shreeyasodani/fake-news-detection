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
