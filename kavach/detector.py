"""Load a prompt-injection detector and score texts with it.

Every detector exposes the same method, `score(texts) -> list[float]`: for each text, the
probability (0 to 1) that it contains a prompt injection.
"""

from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Each model names its "this is an attack" class differently.
POSITIVE_LABELS = {"INJECTION", "LABEL_1", "MALICIOUS", "JAILBREAK", "UNSAFE", "ATTACK"}


class TransformerDetector:
    """A fine-tuned transformer classifier, from the Hugging Face Hub or a local folder.

    Long texts are split into overlapping windows and the text gets its most suspicious
    window's score, so an attack at the end of a long email isn't cut off.
    """

    def __init__(self, model: str, max_length: int = 512, stride: int = 64, batch_size: int = 16):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        self.max_length, self.stride, self.batch_size = max_length, stride, batch_size

        id2label = {int(i): name.upper() for i, name in self.model.config.id2label.items()}
        positive = [i for i, name in id2label.items() if name in POSITIVE_LABELS]
        if not positive:
            raise ValueError(f"Can't tell which class means 'attack': {id2label}")
        self.positive_index = positive[0]

    @torch.no_grad()
    def score(self, texts: list[str]) -> list[float]:
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length, stride=self.stride,
                             return_overflowing_tokens=True)
        owner = enc.pop("overflow_to_sample_mapping")  # window -> index of the text it came from
        # Batch windows of similar length together, so there's less padding to compute.
        order = sorted(range(len(owner)), key=lambda w: len(enc["input_ids"][w]))

        scores = [0.0] * len(texts)
        for start in range(0, len(order), self.batch_size):
            ids = order[start:start + self.batch_size]
            batch = self.tokenizer.pad({k: [v[w] for w in ids] for k, v in enc.items()},
                                       return_tensors="pt")
            probs = torch.softmax(self.model(**batch).logits, dim=-1)[:, self.positive_index]
            for w, p in zip(ids, probs.tolist()):
                scores[owner[w]] = max(scores[owner[w]], p)
        return scores


class BaselineDetector:
    """A scikit-learn pipeline saved with joblib (see train/baseline.py)."""

    def __init__(self, path: str):
        self.pipeline = joblib.load(path)

    def score(self, texts: list[str]) -> list[float]:
        return self.pipeline.predict_proba(texts)[:, 1].tolist()


def load_detector(model: str, **kwargs):
    """`model` is a Hugging Face model id, a local model folder, or a `.joblib` baseline file.
    Extra keyword arguments (e.g. max_length) go to TransformerDetector."""
    if Path(model).suffix == ".joblib":
        return BaselineDetector(model)
    return TransformerDetector(model, **kwargs)
