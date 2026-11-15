"""Metrics for comparing detectors."""

import pandas as pd
from sklearn.metrics import roc_auc_score

THRESHOLD = 0.5  # score above this = "flagged as attack"


def detection_metrics(labels: pd.Series, scores: pd.Series, threshold: float = THRESHOLD) -> dict:
    flagged = scores >= threshold
    attacks = labels == 1
    normal = ~attacks

    tp = int((flagged & attacks).sum())
    fp = int((flagged & normal).sum())
    recall = tp / attacks.sum() if attacks.any() else float("nan")
    false_alarm_rate = fp / normal.sum() if normal.any() else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both_classes = attacks.any() and normal.any()

    return {
        "attacks_caught": f"{tp}/{int(attacks.sum())}",
        "recall": round(recall, 3),
        "false_alarms": f"{fp}/{int(normal.sum())}",
        "false_alarm_rate": round(false_alarm_rate, 3),
        "f1": round(f1, 3),
        "auroc": round(roc_auc_score(labels, scores), 3) if both_classes else float("nan"),
    }


def metrics_by_group(df: pd.DataFrame, score_col: str, group_col: str = "lang",
                     groups: list[str] | None = None) -> list[dict]:
    """One row of metrics per group (e.g. per language), plus an 'all' row."""
    groups = groups or sorted(df[group_col].unique())
    rows = []
    for group in groups + ["all"]:
        part = df if group == "all" else df[df[group_col] == group]
        rows.append({"detector": score_col, group_col: group,
                     **detection_metrics(part["label"], part[score_col])})
    return rows
