#!/usr/bin/env python3
"""
build_items.py  --  ET&S study, step 1

Builds the fixed item sets from the three local datasets.

Reads   : D:\\claude_projects\\ET&S\\02-datasets\\{arc,mmlu,chembench}
Writes  : D:\\claude_projects\\ET&S\\02-datasets\\00-item-sets\\
            items.jsonl          one row per item, all tiers
            item_counts.json     pool / sampled / probe / open counts
            dedup_report.json    what was removed and why

Run:  py build_items.py
"""

import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- config

ROOT = Path(r"D:\claude_projects\ET&S\02-datasets")
OUT = ROOT / "00-item-sets"

SEED = 20260921          # fixed once, never changed
TARGET_PER_TIER = 1000   # cap; pools smaller than this are used whole
N_PROBE = 200            # abstention-probe subset per tier
N_OPEN = 200             # semantic-entropy subset per tier

# MMLU test subsets forming the college tier (chemistry, engineering,
# physical science). Listed explicitly so the scope is auditable.
MMLU_SUBSETS = [
    "college_chemistry",
    "high_school_chemistry",
    "electrical_engineering",
    "college_physics",
    "conceptual_physics",
    "high_school_physics",
]

# ---------------------------------------------------------------- helpers


def norm(text: str) -> str:
    """Normalise for duplicate detection only; never for prompting."""
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


def fingerprint(question: str, options) -> str:
    joined = norm(question) + "||" + "|".join(sorted(norm(o) for o in options))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- loaders


def load_arc():
    """ARC-Challenge, TEST split only. Tier: school."""
    p = ROOT / "arc" / "ARC-Challenge" / "test-00000-of-00001.parquet"
    df = pd.read_parquet(p)
    items = []
    for _, r in df.iterrows():
        texts = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])
        if len(texts) < 3 or r["answerKey"] not in labels:
            continue
        items.append(
            {
                "source": "ARC-Challenge",
                "subject": "science",
                "question": r["question"],
                "options": texts,
                "answer_index": labels.index(r["answerKey"]),
                "src_id": r["id"],
            }
        )
    return items


def load_mmlu():
    """MMLU, TEST split only, chemistry / engineering / physics. Tier: college."""
    items = []
    for sub in MMLU_SUBSETS:
        p = ROOT / "mmlu" / sub / "test-00000-of-00001.parquet"
        if not p.exists():
            print(f"  ! missing MMLU subset: {sub}")
            continue
        df = pd.read_parquet(p)
        for i, r in df.iterrows():
            opts = list(r["choices"])
            ans = int(r["answer"])
            if len(opts) < 3 or not (0 <= ans < len(opts)):
                continue
            items.append(
                {
                    "source": "MMLU",
                    "subject": sub,
                    "question": r["question"],
                    "options": opts,
                    "answer_index": ans,
                    "src_id": f"{sub}:{i}",
                }
            )
    return items


def load_chembench():
    """ChemBench, multiple-choice items only. Tier: expert."""
    items = []
    for d in sorted((ROOT / "chembench").iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        for p in d.glob("*.parquet"):
            df = pd.read_parquet(p)
            for _, r in df.iterrows():
                m = r["metrics"] if "metrics" in r.index else None
                metrics = [] if m is None else [str(x) for x in list(m)]
                if "multiple_choice_grade" not in metrics:
                    continue
                exs = r["examples"] if "examples" in r.index else None
                if exs is None:
                    continue
                for ex in list(exs):
                    raw = ex.get("target_scores") if hasattr(ex, "get") else None
                    if raw is None or not str(raw).strip():
                        continue
                    try:
                        scores = json.loads(raw)
                    except Exception:
                        continue
                    opts = list(scores.keys())
                    correct = [o for o, s in scores.items() if s == 1]
                    if len(opts) < 3 or len(correct) != 1:
                        continue
                    items.append(
                        {
                            "source": "ChemBench",
                            "subject": d.name,
                            "question": ex["input"],
                            "options": opts,
                            "answer_index": opts.index(correct[0]),
                            "src_id": str(r.get("uuid", "")),
                        }
                    )
    return items


# ---------------------------------------------------------------- pipeline


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading ...")
    tiers = {
        "school": load_arc(),
        "college": load_mmlu(),
        "expert": load_chembench(),
    }
    for t, v in tiers.items():
        print(f"  {t:8s} raw pool = {len(v)}")

    # ---- deduplicate, globally, across all three tiers -----------------
    seen = {}
    removed = []
    for tier, items in tiers.items():
        kept = []
        for it in items:
            fp = fingerprint(it["question"], it["options"])
            if fp in seen:
                removed.append(
                    {"tier": tier, "src_id": it["src_id"], "duplicate_of": seen[fp]}
                )
                continue
            seen[fp] = f"{tier}:{it['src_id']}"
            it["fingerprint"] = fp
            kept.append(it)
        tiers[tier] = kept
    print(f"  duplicates removed = {len(removed)}")

    # ---- stratified sample, up to TARGET_PER_TIER ----------------------
    counts = {}
    final = []
    for tier, items in tiers.items():
        pool = len(items)
        if pool <= TARGET_PER_TIER:
            sample = list(items)
        else:
            # proportional by subject, remainder filled at random
            by_sub = {}
            for it in items:
                by_sub.setdefault(it["subject"], []).append(it)
            sample = []
            for sub, group in by_sub.items():
                k = round(TARGET_PER_TIER * len(group) / pool)
                rng.shuffle(group)
                sample.extend(group[:k])
            if len(sample) > TARGET_PER_TIER:
                rng.shuffle(sample)
                sample = sample[:TARGET_PER_TIER]
            elif len(sample) < TARGET_PER_TIER:
                chosen = {id(x) for x in sample}
                rest = [x for x in items if id(x) not in chosen]
                rng.shuffle(rest)
                sample.extend(rest[: TARGET_PER_TIER - len(sample)])

        rng.shuffle(sample)
        probe = {id(x) for x in sample[:N_PROBE]}
        open_ = {id(x) for x in sample[N_PROBE : N_PROBE + N_OPEN]}

        for n, it in enumerate(sample):
            it["tier"] = tier
            it["item_id"] = f"{tier[:3]}-{n:04d}"
            it["in_probe"] = id(it) in probe
            it["in_open"] = id(it) in open_
            final.append(it)

        counts[tier] = {
            "pool_after_dedup": pool,
            "sampled": len(sample),
            "probe": min(N_PROBE, len(sample)),
            "open_ended": min(N_OPEN, max(0, len(sample) - N_PROBE)),
        }
        print(
            f"  {tier:8s} pool={pool:5d}  sampled={len(sample):5d}"
            f"  probe={counts[tier]['probe']}  open={counts[tier]['open_ended']}"
        )

    # ---- write --------------------------------------------------------
    with open(OUT / "items.jsonl", "w", encoding="utf-8") as f:
        for it in final:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    meta = {
        "seed": SEED,
        "target_per_tier": TARGET_PER_TIER,
        "mmlu_subsets": MMLU_SUBSETS,
        "splits_used": {
            "ARC-Challenge": "test",
            "MMLU": "test",
            "ChemBench": "whole benchmark (evaluation-only, no split)",
        },
        "counts": counts,
        "total_items": len(final),
    }
    (OUT / "item_counts.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "dedup_report.json").write_text(
        json.dumps({"removed": len(removed), "detail": removed}, indent=2),
        encoding="utf-8",
    )

    print(f"\nwrote {len(final)} items -> {OUT}")


if __name__ == "__main__":
    main()
