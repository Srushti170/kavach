"""Step 3b: fine-tune a multilingual transformer.

Fine-tuning = take a model that already understands language (pretrained on huge amounts of
text in about 100 languages) and train it a little more on our examples, so it learns one task:
attack or normal.

Base model: intfloat/multilingual-e5-small (118M parameters, 12 layers). Small enough to train
on a laptop CPU, and pretrained on English, Hindi and Marathi text.

Usage:
    .venv/Scripts/python train/finetune.py
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from kavach.metrics import metrics_by_group

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "models" / "kavach-small"
LOG_PATH = ROOT / "results" / "training_log.csv"

BASE_MODEL = "intfloat/multilingual-e5-small"
EPOCHS = 3             # passes over the training data
BATCH_SIZE = 16        # examples per update
LEARNING_RATE = 5e-5   # how big each update is
WARMUP_FRACTION = 0.1  # start with tiny updates, ramp up over the first 10% of steps
WEIGHT_DECAY = 0.01    # gently pulls weights toward zero, to avoid memorizing
MAX_LENGTH = 256       # tokens per example; 99.97% of our examples are shorter
SEED = 42
LOG_EVERY = 50
LANGS = ["en", "hi", "hinglish", "mr"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(df: pd.DataFrame, tokenizer) -> list[dict]:
    enc = tokenizer(df["text"].tolist(), truncation=True, max_length=MAX_LENGTH)
    return [{"input_ids": ids, "attention_mask": mask, "labels": label}
            for ids, mask, label in zip(enc["input_ids"], enc["attention_mask"], df["label"])]


def length_grouped_batches(features: list[dict], rng: random.Random) -> list[list[int]]:
    """Random batches, but each made of examples of similar length, so less padding is computed."""
    order = list(range(len(features)))
    rng.shuffle(order)
    batches = []
    group = BATCH_SIZE * 50
    for start in range(0, len(order), group):
        chunk = sorted(order[start:start + group], key=lambda i: len(features[i]["input_ids"]))
        batches += [chunk[i:i + BATCH_SIZE] for i in range(0, len(chunk), BATCH_SIZE)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict(model, features: list[dict], collator) -> list[float]:
    model.eval()
    probs = []
    for start in range(0, len(features), 64):
        batch = collator(features[start:start + 64])
        batch.pop("labels")
        probs += torch.softmax(model(**batch).logits, dim=-1)[:, 1].tolist()
    model.train()
    return probs


def main() -> None:
    set_seed(SEED)
    rng = random.Random(SEED)

    train = pd.read_json(DATA_DIR / "train.jsonl", lines=True)
    val = pd.read_json(DATA_DIR / "val.jsonl", lines=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "NORMAL", 1: "INJECTION"},
        label2id={"NORMAL": 0, "INJECTION": 1})
    collator = DataCollatorWithPadding(tokenizer)

    train_features, val_features = encode(train, tokenizer), encode(val, tokenizer)

    # Attacks are about a third of the data, so each attack counts for more in the loss.
    counts = train["label"].value_counts().sort_index()
    class_weights = torch.tensor((len(train) / (2 * counts)).tolist(), dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    steps_per_epoch = -(-len(train_features) // BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRACTION * total_steps), total_steps)

    print(f"Training on {len(train_features)} examples, validating on {len(val_features)}.")
    print(f"{EPOCHS} epochs x {steps_per_epoch} steps = {total_steps} steps.\n", flush=True)

    log, best_score, step, start_time = [], -1.0, 0, time.perf_counter()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        for batch_ids in length_grouped_batches(train_features, rng):
            batch = collator([train_features[i] for i in batch_ids])
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)

            loss.backward()                                          # how should each weight change?
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # cap very large changes
            optimizer.step()                                         # change the weights
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            running_loss += loss.item()
            if step % LOG_EVERY == 0:
                elapsed = time.perf_counter() - start_time
                eta = elapsed / step * (total_steps - step)
                print(f"epoch {epoch} | step {step}/{total_steps} | loss {running_loss / LOG_EVERY:.4f} "
                      f"| {elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left", flush=True)
                running_loss = 0.0

        # Check progress on the validation set, which the model never trains on.
        val["score"] = predict(model, val_features, collator)
        metrics = pd.DataFrame(metrics_by_group(val, "score", groups=LANGS))
        macro_f1 = metrics[metrics["lang"].isin(LANGS)]["f1"].mean()  # every language counts equally
        print(f"\nEpoch {epoch} validation (macro F1 across languages: {macro_f1:.3f}):")
        print(metrics.drop(columns="detector").to_markdown(index=False), "\n", flush=True)
        log += [{"epoch": epoch, **row} for row in metrics.to_dict("records")]

        if macro_f1 > best_score:
            best_score = macro_f1
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"New best model saved to {OUTPUT_DIR}\n", flush=True)

    LOG_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(LOG_PATH, index=False)
    print(f"Done in {(time.perf_counter() - start_time) / 60:.1f} min. Best macro F1: {best_score:.3f}")


if __name__ == "__main__":
    main()
