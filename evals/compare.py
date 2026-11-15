"""Step 3c: compare detectors on our test split and on the gap test.

  test split: unseen attacks and carriers, made the same way as the training data.
  gap test:   written separately (Step 1), never trained on. The honest score.

Usage:
    .venv/Scripts/python evals/compare.py
"""

import time
from pathlib import Path

import pandas as pd

from gap_test import load_examples as load_gap_test
from kavach import load_detector
from kavach.metrics import metrics_by_group

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
LANGS = ["en", "hi", "hinglish", "mr"]

# Short name -> (model id or path, options for load_detector)
DETECTORS = {
    "protectai-v2": ("protectai/deberta-v3-base-prompt-injection-v2", {}),
    "baseline": (str(ROOT / "models" / "baseline.joblib"), {}),
    "kavach-small": (str(ROOT / "models" / "kavach-small"), {"max_length": 256}),
}


def compact(summary: pd.DataFrame, set_name: str) -> str:
    """One row per detector, one column per language: F1 and false alarm rate."""
    part = summary[summary["set"] == set_name]
    f1 = part.pivot(index="detector", columns="lang", values="f1")[LANGS + ["all"]]
    fa = part.pivot(index="detector", columns="lang", values="false_alarm_rate")[LANGS + ["all"]]
    order = list(DETECTORS)
    return ("F1 (higher is better)\n\n" + f1.loc[order].to_markdown() +
            "\n\nFalse alarm rate (lower is better)\n\n" + fa.loc[order].to_markdown())


def main() -> None:
    sets = {
        "gap test": load_gap_test(),
        "test split": pd.read_json(ROOT / "data" / "processed" / "test.jsonl", lines=True),
    }

    rows = []
    for name, (model, options) in DETECTORS.items():
        print(f"Scoring with {name} ...", flush=True)
        start = time.perf_counter()
        detector = load_detector(model, **options)
        for set_name, df in sets.items():
            df[name] = detector.score(df["text"].tolist())
            rows += [{"set": set_name, **row} for row in metrics_by_group(df, name, groups=LANGS)]
        print(f"  done in {time.perf_counter() - start:.1f}s", flush=True)

    summary = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    sets["gap test"].to_csv(RESULTS_DIR / "comparison_gap_predictions.csv", index=False, encoding="utf-8-sig")
    sets["test split"].to_csv(RESULTS_DIR / "comparison_test_predictions.csv", index=False, encoding="utf-8-sig")

    report = ["# Detector comparison", ""]
    for set_name in sets:
        report += [f"## {set_name}", "", compact(summary, set_name), "",
                   "<details><summary>Full table</summary>", "",
                   summary[summary["set"] == set_name].drop(columns="set").to_markdown(index=False),
                   "", "</details>", ""]
    (RESULTS_DIR / "comparison.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
