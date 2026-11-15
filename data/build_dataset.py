"""Step 2: build the training dataset.

Sources
  1. Our seeds (data/seeds/): 40 attack payloads, 30 carrier texts and 52 hard negatives,
     each in English, Hindi, Hinglish and Marathi. They are combined into realistic examples:
     an attack hidden inside an email, the same email with a harmless note instead, and so on.
  2. Public English injection datasets: deepset/prompt-injections, Lakera/gandalf_ignore_instructions.
  3. Public normal text: OPUS-100 (English, Hindi, Marathi) and english-to-hinglish (Hinglish).

Splits
  Seeds are split by id *before* combining, so a payload or carrier in the test split never
  appears in training, not even inside a different email. The gap test (data/gap_test/) is
  never used here: it's the final exam.

Usage:
    .venv/Scripts/python data/build_dataset.py

Output: data/processed/{train,val,test}.jsonl and data/processed/stats.md
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = ROOT / "data" / "seeds"
OUT_DIR = ROOT / "data" / "processed"
GAP_TEST_PATH = ROOT / "data" / "gap_test" / "gap_test_v1.json"

LANGS = ["en", "hi", "hinglish", "mr"]
SPLITS = ["train", "val", "test"]
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
RANDOM_SEED = 42

# How many different carriers each payload / hard negative gets hidden in.
CARRIERS_PER_SNIPPET = {"train": 10, "val": 6, "test": 6}
# Chance that a hidden snippet is in a different language from the text around it.
CROSS_LANGUAGE_RATE = 0.15
# Chance of sprinkling invisible zero-width spaces into a snippet (an evasion trick).
ZERO_WIDTH_RATE = 0.05

# Slot values shared by attacks and normal texts, so an email address, link or amount
# is never a giveaway on its own.
EMAILS = [
    "accounts@finserve.example", "priya.sharma@mail.example", "support@shopkart.example",
    "verify-team@securemail.example", "hr@company.example", "rahul.k@office.example",
    "billing@utility.example", "backup.service@cloudbox.example", "noreply@notify.example",
    "admin@college.example", "help@bankdesk.example", "payments@upidesk.example",
]
URLS = [
    "https://portal.example/login", "https://bit.example/x7Kq2", "https://docs.company.example/share/91",
    "https://kyc-update.example/verify", "https://tickets.railway.example/download",
    "https://shop.example/orders/5521", "https://forms.example/apply",
    "https://cdn.files.example/update.apk", "https://bookings.example/table", "http://192.0.2.15/upload",
]
AMOUNTS = ["₹499", "₹1,250", "₹2,999", "₹5,000", "₹12,400", "₹25,000", "₹49,999", "₹1,00,000",
           "Rs. 750", "Rs 3,200"]

# Ways a snippet can be tucked into a carrier. Used equally for attacks and harmless notes,
# so "hidden in an HTML comment" is never a giveaway on its own either.
WRAPPERS = {
    "plain": "{s}",
    "ps": "P.S. {s}",
    "note": "[Note: {s}]",
    "html_comment": "<!-- {s} -->",
    "hidden_span": '<span style="display:none">{s}</span>',
    "white_text": '<span style="color:#ffffff;font-size:1px">{s}</span>',
    "system_tag": "[SYSTEM] {s}",
    "quote": "> {s}",
}
WRAPPER_WEIGHTS = [0.35, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05]
POSITIONS = ["end", "start", "middle"]
POSITION_WEIGHTS = [0.5, 0.2, 0.3]

# Real emails and pages often end with contact details. Adding them to carriers (for both
# labels) stops the model from learning "contains an email address = attack".
SIGNATURES = ["--\n{EMAIL}", "Contact: {EMAIL}", "Web: {URL}",
              "Mail: {EMAIL} | Phone: +91 22 5550 1234", "Unsubscribe: {URL}"]
SIGNATURE_RATE = 0.4

# Normal sentences taken from public corpora, in total across all splits.
EXTERNAL_BENIGN_COUNTS = {"en": 600, "hi": 900, "mr": 900, "hinglish": 900}

GERMAN_HINTS = re.compile(r"\b(und|der|das|ich|nicht|ist|wie|eine|mit|für|Sie|ein)\b")
SENTENCE_BREAK = re.compile(r"(?<=[.!?।])\s+")


# ---------- helpers ----------

def load_seed(name: str) -> list[dict]:
    return json.loads((SEEDS_DIR / f"{name}.json").read_text(encoding="utf-8"))["items"]


def split_items(items: list, rng: random.Random) -> dict[str, list]:
    """Shuffle and cut a list into train / val / test."""
    items = items[:]
    rng.shuffle(items)
    n_train = round(len(items) * SPLIT_FRACTIONS[0])
    n_val = round(len(items) * SPLIT_FRACTIONS[1])
    return {"train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:]}


def fill_slots(text: str, rng: random.Random) -> str:
    text = re.sub(r"\{EMAIL\}", lambda _: rng.choice(EMAILS), text)
    text = re.sub(r"\{URL\}", lambda _: rng.choice(URLS), text)
    return re.sub(r"\{AMOUNT\}", lambda _: rng.choice(AMOUNTS), text)


def carrier_text(carrier: dict, lang: str, rng: random.Random) -> str:
    text = carrier[lang]
    if rng.random() < SIGNATURE_RATE:
        text += "\n\n" + rng.choice(SIGNATURES)
    return fill_slots(text, rng)


def add_zero_width(text: str, rng: random.Random) -> str:
    return "".join(ch + "​" if ch == " " and rng.random() < 0.3 else ch for ch in text)


def insert_snippet(carrier: str, snippet: str, position: str, rng: random.Random) -> str:
    if position == "start":
        return f"{snippet}\n\n{carrier}"
    if position == "middle":
        sep = "\n" if "\n" in carrier else " "
        parts = carrier.split("\n") if sep == "\n" else SENTENCE_BREAK.split(carrier)
        if len(parts) > 1:
            cut = rng.randint(1, len(parts) - 1)
            return sep.join(parts[:cut] + [snippet] + parts[cut:])
    return f"{carrier}\n\n{snippet}"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("​", "")).strip().lower()


def row(text, label, lang, source, kind, **extra) -> dict:
    return {"text": text, "label": label, "lang": lang, "source": source, "kind": kind, **extra}


# ---------- 1. seed compositions ----------

def compose_seed_examples(split: str, payloads, carriers, hard_negs, rng: random.Random) -> list[dict]:
    """Turn the seeds of one split into examples, for every language."""
    rows = []
    k = min(CARRIERS_PER_SNIPPET[split], len(carriers))

    def embed(snippet_item: dict, carrier: dict, carrier_lang: str, label: int) -> dict:
        snippet_lang = carrier_lang
        if rng.random() < CROSS_LANGUAGE_RATE:
            snippet_lang = rng.choice([l for l in LANGS if l != carrier_lang])
        snippet = fill_slots(snippet_item[snippet_lang], rng)
        if rng.random() < ZERO_WIDTH_RATE:
            snippet = add_zero_width(snippet, rng)
        wrapper = rng.choices(list(WRAPPERS), WRAPPER_WEIGHTS)[0]
        position = rng.choices(POSITIONS, POSITION_WEIGHTS)[0]
        text = insert_snippet(carrier_text(carrier, carrier_lang, rng),
                              WRAPPERS[wrapper].format(s=snippet), position, rng)
        return row(text, label, carrier_lang, "seed", "embedded", snippet_lang=snippet_lang,
                   wrapper=wrapper, position=position, seed_ids=f"{snippet_item['id']}+{carrier['id']}")

    for lang in LANGS:
        for p in payloads:
            rows.append(row(fill_slots(p[lang], rng), 1, lang, "seed", "direct", seed_ids=p["id"]))
            rows.extend(embed(p, c, lang, 1) for c in rng.sample(carriers, k))
        for h in hard_negs:
            rows.append(row(fill_slots(h[lang], rng), 0, lang, "seed", "direct", seed_ids=h["id"]))
            rows.extend(embed(h, c, lang, 0) for c in rng.sample(carriers, k))
        for c in carriers:
            rows.append(row(carrier_text(c, lang, rng), 0, lang, "seed", "carrier_only", seed_ids=c["id"]))
    return rows


# ---------- 2. public injection datasets ----------

def load_public_injections(rng: random.Random) -> dict[str, list[dict]]:
    out = {s: [] for s in SPLITS}

    deepset = load_dataset("deepset/prompt-injections")
    train_rows = list(deepset["train"])
    rng.shuffle(train_rows)
    n_val = round(len(train_rows) * 0.15)
    parts = {"val": train_rows[:n_val], "train": train_rows[n_val:], "test": list(deepset["test"])}
    for split, items in parts.items():
        for r in items:
            lang = "de" if len(GERMAN_HINTS.findall(r["text"])) >= 2 else "en"
            out[split].append(row(r["text"], int(r["label"]), lang, "deepset", "public"))

    gandalf = load_dataset("Lakera/gandalf_ignore_instructions")
    for split, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        out[split].extend(row(r["text"], 1, "en", "gandalf", "public") for r in gandalf[hf_split])
    return out


# ---------- 3. public normal text ----------

def clean_sentences(texts, rng: random.Random, n: int) -> list[str]:
    seen, kept = set(), []
    for t in texts:
        t = (t or "").strip()
        if 20 <= len(t) <= 300 and normalize(t) not in seen:
            seen.add(normalize(t))
            kept.append(t)
    rng.shuffle(kept)
    return kept[:n]


def load_public_benign(rng: random.Random) -> dict[str, list[dict]]:
    pools = {}
    hi_pairs = [r["translation"] for split in ("validation", "test")
                for r in load_dataset("Helsinki-NLP/opus-100", "en-hi", split=split)]
    mr_pairs = [r["translation"] for split in ("validation", "test")
                for r in load_dataset("Helsinki-NLP/opus-100", "en-mr", split=split)]
    half = len(hi_pairs) // 2  # take Hindi and English from different rows, so they aren't translations of each other
    pools["hi"] = ("opus-100", clean_sentences([p["hi"] for p in hi_pairs[:half]], rng, EXTERNAL_BENIGN_COUNTS["hi"]))
    pools["en"] = ("opus-100", clean_sentences([p["en"] for p in hi_pairs[half:]], rng, EXTERNAL_BENIGN_COUNTS["en"]))
    pools["mr"] = ("opus-100", clean_sentences([p["mr"] for p in mr_pairs], rng, EXTERNAL_BENIGN_COUNTS["mr"]))

    path = hf_hub_download("findnitai/english-to-hinglish", "hinglish_upload_v1.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        hinglish = [json.loads(line)["translation"]["hi_ng"] for line in f if line.strip()]
    pools["hinglish"] = ("english-to-hinglish", clean_sentences(hinglish, rng, EXTERNAL_BENIGN_COUNTS["hinglish"]))

    out = {s: [] for s in SPLITS}
    for lang, (source, sentences) in pools.items():
        for split, part in split_items(sentences, rng).items():
            out[split].extend(row(t, 0, lang, source, "public") for t in part)
    return out


# ---------- clean-up and stats ----------

def deduplicate(splits: dict[str, list[dict]], forbidden: set[str]) -> dict[str, int]:
    """Drop gap-test texts and duplicates. A text in test or val is removed from train."""
    removed = Counter()
    seen = set()
    for split in ["test", "val", "train"]:
        kept = []
        for r in splits[split]:
            key = normalize(r["text"])
            if key in forbidden:
                removed["matches gap test"] += 1
            elif key in seen:
                removed["duplicate"] += 1
            else:
                seen.add(key)
                kept.append(r)
        splits[split] = kept
    return dict(removed)


def shortcut_report(df: pd.DataFrame) -> pd.DataFrame:
    """Surface features per label. If one differs a lot between attacks and normal text,
    the model could learn that feature instead of the meaning."""
    features = pd.DataFrame({
        "label": df["label"].map({1: "attack", 0: "normal"}),
        "avg_chars": df["text"].str.len(),
        "has_email_%": df["text"].str.contains("@") * 100,
        "has_link_%": df["text"].str.contains("http") * 100,
        "devanagari_%": df["text"].str.contains("[ऀ-ॿ]") * 100,
        "html_tag_%": df["text"].str.contains("<") * 100,
    })
    return features.groupby("label").mean().round(1)


def write_stats(splits: dict[str, list[dict]], removed: dict) -> str:
    df = pd.concat([pd.DataFrame(rows).assign(split=s) for s, rows in splits.items()])
    by_split = df.pivot_table(index="split", columns="label", values="text", aggfunc="count")
    by_split.columns = ["normal", "attack"]
    by_lang = df.pivot_table(index=["split", "lang"], columns="label", values="text", aggfunc="count", fill_value=0)
    by_lang.columns = ["normal", "attack"]
    by_source = df.pivot_table(index="source", columns="label", values="text", aggfunc="count", fill_value=0)
    by_source.columns = ["normal", "attack"]

    seed_train = df[(df["split"] == "train") & (df["source"] == "seed")]
    lines = [
        "# Dataset stats",
        "",
        f"Total examples: {len(df)}. Removed while cleaning: {removed or 'nothing'}.",
        "",
        "## By split", "", by_split.loc[SPLITS].to_markdown(), "",
        "## By split and language", "", by_lang.to_markdown(), "",
        "## By source", "", by_source.to_markdown(), "",
        "## Shortcut check (seed examples, train split)", "",
        "Surface features should look similar for attacks and normal texts.", "",
        shortcut_report(seed_train).to_markdown(), "",
        "## Licenses", "",
        "- Seeds: written for this project.",
        "- deepset/prompt-injections: Apache-2.0.",
        "- Lakera/gandalf_ignore_instructions: MIT.",
        "- findnitai/english-to-hinglish: Apache-2.0.",
        "- Helsinki-NLP/opus-100: mixed licenses from the OPUS corpora (listed as unknown); "
        "drop `source == \"opus-100\"` rows before redistributing.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    payloads = split_items(load_seed("payloads"), rng)
    carriers = split_items(load_seed("carriers"), rng)
    hard_negs = split_items(load_seed("hard_negatives"), rng)

    splits = {s: compose_seed_examples(s, payloads[s], carriers[s], hard_negs[s], rng) for s in SPLITS}
    print("Seed examples:", {s: len(v) for s, v in splits.items()})

    print("Loading public injection datasets ...")
    for s, rows in load_public_injections(rng).items():
        splits[s].extend(rows)
    print("Loading public normal text ...")
    for s, rows in load_public_benign(rng).items():
        splits[s].extend(rows)

    gap_items = json.loads(GAP_TEST_PATH.read_text(encoding="utf-8"))["items"]
    forbidden = {normalize(item[lang]) for item in gap_items for lang in LANGS}
    removed = deduplicate(splits, forbidden)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        rng.shuffle(splits[s])
        with open(OUT_DIR / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in splits[s]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = write_stats(splits, removed)
    (OUT_DIR / "stats.md").write_text(stats, encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
