#!/usr/bin/env python3
"""
analyse_entropy.py  --  ET&S study, step 5 (semantic entropy)   [v2]

Implements Farquhar et al. (2024): cluster an item's sampled generations by
BIDIRECTIONAL ENTAILMENT, then take entropy over meaning clusters rather than
over token sequences. Reports AUROC (does entropy predict a wrong answer) and
AURAC (accuracy on the answers the system did not refuse).

v2 fixes two defects in the first implementation, both of which corrupted the
accuracy-dependent outputs of the 2026-09-28 run:

  1. The question was used only as a cache key and was never given to the
     entailment model. Farquhar et al. concatenate the question with each
     answer before running NLI, because entailment between bare answer strings
     is undefined -- "Paris" and "the capital is Paris" do not entail each
     other out of context. Every NLI input is now "<question> <answer>".
  2. Grading is reported at two strictness levels rather than one. The gold
     text is a short multiple-choice option while the model writes a sentence,
     so bidirectional equivalence with the gold string is a harsh test. Strict
     (bidirectional) remains the headline figure; lenient (the answer entails
     the gold) is recorded alongside so the grader's own effect is visible.

The cache from v1 is invalid because its values were computed without the
question, and is ignored: cache entries are versioned.

Performance: every ordered pair an item needs is scored in ONE batched forward
pass, instead of two-pair calls. That is the difference between a 2 h run and
an 8 h one.

  py analyse_entropy.py --check              report backend availability
  py analyse_entropy.py --limit 5            smoke test
  py analyse_entropy.py                      full run
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\claude_projects\ET&S")
RAW = ROOT / "06-results" / "raw"
OUTJ = ROOT / "06-results" / "analysis"
TABLES = ROOT / "08-tables"
CACHE = OUTJ / "entailment_cache_v2.json"
CELLS = OUTJ / "entropy_cells"        # one file per finished cell, so the run
                                      # can be stopped and resumed anywhere

TIERS = ["school", "college", "expert"]
LEVELS = ["fp16", "q8_0", "q4_K_M"]
FAMILIES = ["qwen2.5:3b-instruct", "llama3.2:3b-instruct", "gemma2:2b-instruct"]

NLI_MODEL = "microsoft/deberta-large-mnli"
MAX_LEN = 192
REJECT_POINT = 0.20


# ---------------------------------------------------------------- cache

_cache = {}


def load_cache():
    global _cache
    if CACHE.exists():
        try:
            _cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    print(f"entailment cache: {len(_cache)} decisions")


def save_cache():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(_cache), encoding="utf-8")


def contextualise(question, answer):
    """The string the NLI model actually sees."""
    q = " ".join((question or "").split())
    a = " ".join((answer or "").split())
    return f"{q} {a}".strip()


def ckey(premise, hypothesis):
    return f"{premise[:400]}|||{hypothesis[:400]}"


# ---------------------------------------------------------------- backend


class NLIBackend:
    """DeBERTa-large-MNLI. Labels: 0 contradiction, 1 neutral, 2 entailment."""

    name = "deberta-large-mnli"

    def __init__(self):
        import os
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        torch.set_grad_enabled(False)
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.eval()

    def entails_batch(self, pairs):
        """pairs: [(premise, hypothesis)] -> [bool], one forward pass."""
        if not pairs:
            return []
        enc = self.tok([p for p, _ in pairs], [h for _, h in pairs],
                       return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_LEN)
        logits = self.model(**enc).logits
        return [int(i) == 2 for i in logits.argmax(dim=-1).tolist()]


# ---------------------------------------------------------------- one item


def score_item(question, generations, gold, backend):
    """All entailment decisions this item needs, in one batched pass.

    Returns (clusters, entail_matrix, texts). texts[-1] is the gold answer;
    indices 0..k-1 are the generations, in their sampled order.
    """
    gens = [g for g in (generations or []) if (g or "").strip()]
    if not gens:
        return None
    texts = [contextualise(question, g) for g in gens]
    gold_ctx = contextualise(question, gold) if gold else None
    if gold_ctx:
        texts.append(gold_ctx)

    n = len(texts)
    wanted, need = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            k = ckey(texts[i], texts[j])
            wanted.append((i, j, k))
            if k not in _cache:
                need.append((texts[i], texts[j]))

    if need:
        # deduplicate within the batch, then one forward pass
        uniq = list(dict.fromkeys(need))
        res = backend.entails_batch(uniq)
        for (p, h), v in zip(uniq, res):
            _cache[ckey(p, h)] = bool(v)

    ent = {}
    for i, j, k in wanted:
        ent[(i, j)] = _cache.get(k, False)

    # greedy bidirectional-entailment clustering over the generations only
    ngen = len(gens)
    clusters = []                       # each is a list of generation indices
    for i in range(ngen):
        placed = False
        for cl in clusters:
            rep = cl[0]
            if ent.get((rep, i)) and ent.get((i, rep)):
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters, ent, gens


def grade(clusters, ent, ngen, has_gold):
    """(strict, lenient) correctness of the modal cluster against gold.

    strict  : modal representative and gold entail each other
    lenient : the modal representative entails gold
    """
    if not has_gold or not clusters:
        return None, None
    g = ngen                                   # gold's index in the text list
    rep = max(clusters, key=len)[0]
    fwd = bool(ent.get((rep, g)))
    bwd = bool(ent.get((g, rep)))
    return (fwd and bwd), fwd


# ---------------------------------------------------------------- metrics


def semantic_entropy(clusters):
    """Discrete estimator: entropy over cluster probabilities.

    Token probabilities are not exposed by the runtime, so the discrete
    estimator is used throughout and this is recorded per configuration.
    """
    n = sum(len(c) for c in clusters)
    if n == 0:
        return None
    return -sum((len(c) / n) * math.log(len(c) / n) for c in clusters)


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties at 0.5.

    scores = entropy, labels = 1 when the answer was wrong. Above 0.5 means
    higher entropy goes with being wrong, which is the claim under test.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return round(wins / (len(pos) * len(neg)), 5)


def rejection_curve(entropies, correct):
    """Accuracy on retained items as the highest-entropy items are refused."""
    paired = sorted(zip(entropies, correct), key=lambda x: x[0])
    n = len(paired)
    pts = []
    for i in range(n):
        keep = paired[: n - i]
        if not keep:
            break
        pts.append((round(i / n, 5),
                    round(sum(1 for _, c in keep if c) / len(keep), 5)))
    return pts


def aurac(curve):
    """Area under the rejection-accuracy curve, over the rejection fraction."""
    if len(curve) < 2:
        return None
    a = sum(0.5 * (y0 + y1) * (x1 - x0)
            for (x0, y0), (x1, y1) in zip(curve, curve[1:]))
    span = curve[-1][0] - curve[0][0]
    return round(a / span, 5) if span else None


def accuracy_at_rejection(curve, r):
    if not curve:
        return None
    return min(curve, key=lambda p: abs(p[0] - r))[1]


# ---------------------------------------------------------------- driver


def load_open(model, tier):
    safe = model.replace(":", "_").replace("/", "_")
    p = RAW / f"{safe}__{tier}__open__r1.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error") or not r.get("generations"):
                continue
            out.append(r)
    return out


def questions_by_id():
    p = ROOT / "02-datasets" / "00-item-sets" / "items.jsonl"
    q = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            q[it["item_id"]] = it["question"]
    return q


def run_cell(model, tier, backend, qmap, limit=None):
    rows = load_open(model, tier)
    if limit:
        rows = rows[:limit]
    if not rows:
        return None

    ent_v, strict_v, lenient_v, ncl = [], [], [], []
    for r in rows:
        question = qmap.get(r["item_id"], "")
        gold = r.get("gold_text", "")
        scored = score_item(question, r["generations"], gold, backend)
        if not scored:
            continue
        clusters, ent, gens = scored
        h = semantic_entropy(clusters)
        s, l = grade(clusters, ent, len(gens), bool(gold))
        if h is None or s is None:
            continue
        ent_v.append(h)
        strict_v.append(1 if s else 0)
        lenient_v.append(1 if l else 0)
        ncl.append(len(clusters))

    if len(ent_v) < 5:
        return None

    out = {"model": model, "tier": tier, "n_items": len(ent_v),
           "mean_entropy": round(sum(ent_v) / len(ent_v), 5),
           "mean_clusters": round(sum(ncl) / len(ncl), 3)}

    for label, cor in (("strict", strict_v), ("lenient", lenient_v)):
        curve = rejection_curve(ent_v, cor)
        out[f"accuracy_{label}"] = round(sum(cor) / len(cor), 5)
        out[f"auroc_{label}"] = auroc(ent_v, [1 - c for c in cor])
        out[f"aurac_{label}"] = aurac(curve)
        out[f"acc_at_{int(REJECT_POINT*100)}pct_rejection_{label}"] = \
            accuracy_at_rejection(curve, REJECT_POINT)
        out[f"curve_{label}"] = curve

    # the headline figures analyse_core.py consumes
    out["accuracy_open"] = out["accuracy_strict"]
    out["auroc"] = out["auroc_strict"]
    out["aurac"] = out["aurac_strict"]
    out[f"acc_at_{int(REJECT_POINT*100)}pct_rejection"] = \
        out[f"acc_at_{int(REJECT_POINT*100)}pct_rejection_strict"]
    return out


def check():
    print("backend availability\n")
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        print("  nli     OK — transformers + torch present")
        print(f"          model {NLI_MODEL}")
    except Exception as e:
        print(f"  nli     NOT AVAILABLE — {type(e).__name__}: {e}")
        print("          fix with:  pip install -U transformers torch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, help="items per cell, for a smoke test")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop cleanly after this long; 0 = run to the end")
    ap.add_argument("--redo", action="store_true",
                    help="recompute cells that are already finished")
    a = ap.parse_args()

    if a.check:
        check()
        return

    OUTJ.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    load_cache()
    print("loading deberta-large-mnli ...")
    try:
        backend = NLIBackend()
    except Exception as e:
        sys.exit(f"backend unavailable: {type(e).__name__}: {e}")

    qmap = questions_by_id()
    cells = [(f"{fam}-{lv}", tier)
             for fam in FAMILIES for lv in LEVELS for tier in TIERS]
    results, t0 = [], time.perf_counter()

    budget = a.max_hours * 3600
    durations, stopped = [], False

    for i, (model, tier) in enumerate(cells, 1):
        safe = model.replace(":", "_").replace("/", "_")
        cf = CELLS / f"{safe}__{tier}.json"

        if cf.exists() and not a.redo and not a.limit:
            results.append(json.loads(cf.read_text(encoding="utf-8")))
            print(f"[{i}/{len(cells)}] {model:28s} {tier:8s} done earlier")
            continue

        if budget:
            elapsed = time.perf_counter() - t0
            guard = max(durations) if durations else 600.0
            if elapsed + guard > budget:
                print(f"\nbudget reached ({elapsed/3600:.2f} h of {a.max_hours} h)"
                      " — stopping before the next cell")
                stopped = True
                break

        print(f"[{i}/{len(cells)}] {model:28s} {tier:8s} ", end="", flush=True)
        t_cell = time.perf_counter()
        r = run_cell(model, tier, backend, qmap, a.limit)
        if r is None:
            print("no data")
            continue
        durations.append(time.perf_counter() - t_cell)
        results.append(r)
        if not a.limit:
            cf.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"n={r['n_items']:3d}  H={r['mean_entropy']:.3f}  "
              f"acc(strict)={r['accuracy_strict']:.3f}  "
              f"acc(lenient)={r['accuracy_lenient']:.3f}  "
              f"AUROC={r['auroc_strict']}  AURAC={r['aurac_strict']}  "
              f"[{durations[-1]/60:.1f} min]")
        save_cache()

    save_cache()
    mins = (time.perf_counter() - t0) / 60
    (OUTJ / "entropy_results.json").write_text(json.dumps({
        "version": 2,
        "backend": backend.name,
        "nli_input": "question concatenated with answer, per Farquhar et al.",
        "estimator": "discrete (token probabilities not exposed)",
        "grading": {"strict": "modal cluster and gold entail each other",
                    "lenient": "modal cluster entails gold"},
        "headline": "strict",
        "reject_point": REJECT_POINT,
        "minutes": round(mins, 2),
        "cells": results}, indent=2), encoding="utf-8")

    TABLES.mkdir(parents=True, exist_ok=True)
    cols = ["model", "tier", "n_items", "mean_entropy", "mean_clusters",
            "accuracy_strict", "accuracy_lenient",
            "auroc_strict", "auroc_lenient",
            "aurac_strict", "aurac_lenient",
            f"acc_at_{int(REJECT_POINT*100)}pct_rejection_strict"]
    with open(TABLES / "table6_entropy.md", "w", encoding="utf-8") as f:
        f.write("# Table 6. Semantic entropy, AUROC and AURAC on the "
                "open-ended set\n\n")
        f.write("Strict grading is the headline; lenient is reported so the "
                "grader's own effect is visible.\n\n")
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in results:
            f.write("| " + " | ".join(
                "—" if r.get(c) is None else str(r.get(c)) for c in cols) + " |\n")

    print(f"\n{len(results)}/27 cells complete, {mins:.1f} min this session")
    print(f"wrote {OUTJ / 'entropy_results.json'}")
    print(f"wrote {TABLES / 'table6_entropy.md'}")
    if len(results) < 27:
        print(f"\n{27 - len(results)} cells left — rerun the same command to "
              "carry on from here")
    else:
        print("\nnow rerun:  py analyse_core.py    (fills criterion 4)")


if __name__ == "__main__":
    main()
