"""Step 1: the gap test.

Runs existing prompt-injection detectors on our hand-written test set and reports,
for each language, how many attacks each detector catches and how many normal
messages it wrongly flags.

Usage:
    .venv/Scripts/python evals/gap_test.py
    .venv/Scripts/python evals/gap_test.py --detectors protectai-v2 promptguard2-86m
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from kavach import load_detector
from kavach.metrics import metrics_by_group

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "gap_test" / "gap_test_v1.json"
RESULTS_DIR = ROOT / "results"

# Short name -> Hugging Face model id. Prompt Guard 2 models are gated (need login).
DETECTORS = {
    "protectai-v2": "protectai/deberta-v3-base-prompt-injection-v2",
    "deepset": "deepset/deberta-v3-base-injection",
    "promptguard2-86m": "meta-llama/Llama-Prompt-Guard-2-86M",
    "promptguard2-22m": "meta-llama/Llama-Prompt-Guard-2-22M",
}

LANGS = ["en", "hi", "hinglish", "mr"]


def load_examples() -> pd.DataFrame:
    """Flatten the 60 parallel items into 240 rows: one per (item, language)."""
    items = json.loads(DATA_PATH.read_text(encoding="utf-8"))["items"]
    rows = []
    for item in items:
        for lang in LANGS:
            rows.append({
                "id": f"{item['id']}-{lang}",
                "item": item["id"],
                "lang": lang,
                "label": item["label"],
                "type": item["type"],
                "text": item[lang],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS),
                        choices=list(DETECTORS))
    args = parser.parse_args()

    df = load_examples()
    print(f"Loaded {len(df)} examples ({df['label'].sum()} attacks, "
          f"{(df['label'] == 0).sum()} normal) in {len(LANGS)} languages.\n")

    ran = []
    for name in args.detectors:
        model_id = DETECTORS[name]
        print(f"Running {name} ({model_id}) ...", flush=True)
        start = time.perf_counter()
        try:
            df[name] = load_detector(model_id).score(df["text"].tolist())
        except Exception as err:  # gated model without login, network error, etc.
            print(f"  skipped: {type(err).__name__}: {str(err).splitlines()[0]}\n")
            continue
        print(f"  done in {time.perf_counter() - start:.1f}s\n")
        ran.append(name)

    if not ran:
        sys.exit("No detector ran.")

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = pd.DataFrame([row for name in ran for row in metrics_by_group(df, name, groups=LANGS)])

    df.to_csv(RESULTS_DIR / "gap_test_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / "gap_test_summary.csv", index=False)
    (RESULTS_DIR / "gap_test_summary.md").write_text(
        summary.to_markdown(index=False) + "\n", encoding="utf-8")

    print(summary.to_markdown(index=False))
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
