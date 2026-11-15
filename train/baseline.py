"""Step 3a: a simple baseline model.

Before training anything fancy, train the simplest reasonable model. It sets the bar:
the fancy model is only worth it if it clearly beats this.

The baseline:
  1. TF-IDF turns each text into numbers: how often each short chunk of characters
     (and each word) appears, weighted so that common chunks count less.
  2. Logistic regression learns one weight per chunk: positive for chunks that show up
     in attacks, negative for chunks that show up in normal text.

Usage:
    .venv/Scripts/python train/baseline.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from kavach.metrics import metrics_by_group

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "baseline.joblib"
LANGS = ["en", "hi", "hinglish", "mr"]


def main() -> None:
    train = pd.read_json(DATA_DIR / "train.jsonl", lines=True)
    val = pd.read_json(DATA_DIR / "val.jsonl", lines=True)

    model = make_pipeline(
        make_union(
            # Character chunks of 2-5 letters: work for any script and survive typos.
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            # Single words and word pairs.
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        ),
        # class_weight="balanced": attacks are rarer, so each one counts for more.
        LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0),
    )
    model.fit(train["text"], train["label"])

    val["baseline"] = model.predict_proba(val["text"])[:, 1]
    print("Validation results:")
    print(pd.DataFrame(metrics_by_group(val, "baseline", groups=LANGS)).to_markdown(index=False))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
