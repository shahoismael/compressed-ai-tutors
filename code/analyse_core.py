#!/usr/bin/env python3
"""
analyse_core.py  --  ET&S study, step 4

Turns the raw campaign into the paper's numbers, for RQ1-RQ4 except semantic
entropy (that needs the entailment pass and lives in analyse_entropy.py).

Produces
  06-results/analysis/core_results.json   every computed quantity
  08-tables/table1_accuracy.md/.csv       accuracy + McNemar vs FP16
  08-tables/table2_loyalty.md/.csv        answer fidelity
  08-tables/table3_hallucination.md/.csv  probe abstention and confident error
  08-tables/table4_deployability.md/.csv  the pre-registered rule, per cell
  08-tables/table5_determinism.md/.csv    repeat-1 vs repeat-2 agreement

No scipy dependency: McNemar is the exact binomial test, computed directly.

Run:  py analyse_core.py
"""

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\claude_projects\ET&S")
RAW = ROOT / "06-results" / "raw"
ENERGY = ROOT / "06-results" / "energy"
OUTJ = ROOT / "06-results" / "analysis"
TABLES = ROOT / "08-tables"

TIERS = ["school", "college", "expert"]
LEVELS = ["fp16", "q8_0", "q4_K_M"]
BASELINE = "fp16"
FAMILIES = ["qwen2.5:3b-instruct", "llama3.2:3b-instruct", "gemma2:2b-instruct"]

BOOT = 10000
SEED = 20260921

# pre-registered deployability rule (methodology section 9)
ACC_TOLERANCE_PP = 3.0
ENERGY_CEILING = 0.60


# ---------------------------------------------------------------- loading


def cell_file(model, tier, mode, repeat):
    safe = model.replace(":", "_").replace("/", "_")
    return RAW / f"{safe}__{tier}__{mode}__r{repeat}.jsonl"


def load_cell(model, tier, mode, repeat=1):
    """item_id -> record. Rows carrying an error are dropped and counted."""
    p = cell_file(model, tier, mode, repeat)
    if not p.exists():
        return {}, 0
    out, errs = {}, 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                errs += 1
                continue
            if r.get("error"):
                errs += 1
                continue
            out[r["item_id"]] = r
    return out, errs


def load_energy(model, tier, mode):
    safe = model.replace(":", "_").replace("/", "_")
    p = ENERGY / f"{safe}__{tier}__{mode}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    return rec.get("energy")


def model_name(family, level):
    return f"{family}-{level}"


# ---------------------------------------------------------------- statistics


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar. b, c are the discordant counts.

    Returns (p, n_discordant). An exact test is used throughout rather than the
    chi-square approximation, because several cells have few discordant pairs
    and the approximation is unreliable there.
    """
    n = b + c
    if n == 0:
        return 1.0, 0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), n


def holm(pairs):
    """pairs: [(key, p)]. Returns {key: adjusted p}, Holm-Bonferroni."""
    ordered = sorted(pairs, key=lambda kv: kv[1])
    m = len(ordered)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # enforce monotonicity
        out[k] = running
    return out


def boot_ci(values, stat=None, n=BOOT, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI of a statistic over a list of 0/1 or floats."""
    if not values:
        return (None, None)
    stat = stat or (lambda v: sum(v) / len(v))
    rng = random.Random(seed)
    k = len(values)
    reps = []
    for _ in range(n):
        reps.append(stat([values[rng.randrange(k)] for _ in range(k)]))
    reps.sort()
    lo = reps[int(alpha / 2 * n)]
    hi = reps[int((1 - alpha / 2) * n) - 1]
    return (round(lo, 5), round(hi, 5))


def paired_diff_ci(pairs, n=BOOT, seed=SEED, alpha=0.05):
    """CI on the paired difference (comp - base), resampling item pairs."""
    if not pairs:
        return (None, None)
    rng = random.Random(seed)
    k = len(pairs)
    reps = []
    for _ in range(n):
        s = 0.0
        for _ in range(k):
            a, b = pairs[rng.randrange(k)]
            s += a - b
        reps.append(s / k)
    reps.sort()
    return (round(reps[int(alpha / 2 * n)], 5),
            round(reps[int((1 - alpha / 2) * n) - 1], 5))


# ---------------------------------------------------------------- RQ1 + RQ2


def accuracy_and_fidelity():
    """Per family x tier: accuracy, McNemar vs FP16, loyalty, parse rate."""
    rows, raw_p = [], []
    for family in FAMILIES:
        for tier in TIERS:
            base, base_err = load_cell(model_name(family, BASELINE), tier, "mcq")
            if not base:
                continue
            base_acc = sum(1 for r in base.values() if r["correct"]) / len(base)

            for level in LEVELS:
                comp, comp_err = load_cell(model_name(family, level), tier, "mcq")
                if not comp:
                    continue
                shared = sorted(set(base) & set(comp))
                n = len(shared)
                acc = sum(1 for r in comp.values() if r["correct"]) / len(comp)
                parsed = sum(1 for r in comp.values() if r.get("parsed")) / len(comp)

                # McNemar on the shared items
                b = sum(1 for i in shared
                        if base[i]["correct"] and not comp[i]["correct"])
                c = sum(1 for i in shared
                        if comp[i]["correct"] and not base[i]["correct"])
                p, ndisc = mcnemar_exact(b, c)

                # loyalty: same label as baseline, right or wrong
                loyal = [1 if comp[i]["pred_index"] == base[i]["pred_index"] else 0
                         for i in shared]
                loyalty = sum(loyal) / n if n else None

                pairs = [(1.0 if comp[i]["correct"] else 0.0,
                          1.0 if base[i]["correct"] else 0.0) for i in shared]

                row = {
                    "family": family, "tier": tier, "level": level,
                    "n_items": len(comp), "n_shared": n,
                    "accuracy": round(acc, 5),
                    "accuracy_ci": boot_ci([1 if r["correct"] else 0
                                            for r in comp.values()]),
                    "parse_rate": round(parsed, 5),
                    "baseline_accuracy": round(base_acc, 5),
                    "delta_pp": round((acc - base_acc) * 100, 3),
                    "delta_ci_pp": tuple(None if v is None else round(v * 100, 3)
                                         for v in paired_diff_ci(pairs)),
                    "mcnemar_b": b, "mcnemar_c": c, "n_discordant": ndisc,
                    "p_raw": None if level == BASELINE else round(p, 6),
                    "loyalty": None if level == BASELINE else round(loyalty, 5),
                    "loyalty_ci": None if level == BASELINE else boot_ci(loyal),
                    "dropped_rows": comp_err,
                }
                rows.append(row)
                if level != BASELINE:
                    raw_p.append(((family, tier, level), p))

    # Holm within each tier family of comparisons
    by_tier = defaultdict(list)
    for key, p in raw_p:
        by_tier[key[1]].append((key, p))
    adjusted = {}
    for tier, pairs in by_tier.items():
        adjusted.update(holm(pairs))
    for r in rows:
        k = (r["family"], r["tier"], r["level"])
        r["p_holm"] = None if k not in adjusted else round(adjusted[k], 6)
        r["significant_05"] = (r["p_holm"] is not None and r["p_holm"] < 0.05)
    return rows


# ---------------------------------------------------------------- RQ3 (probe)


def hallucination():
    """Abstention-permitted probe: how often the model picks a distractor.

    The correct option was removed, so every non-abstention is a confident
    wrong answer. Abstention is the correct behaviour on every probe item.
    """
    rows, raw_p = [], []
    for family in FAMILIES:
        for tier in TIERS:
            base, _ = load_cell(model_name(family, BASELINE), tier, "probe")
            if not base:
                continue
            base_rate = sum(1 for r in base.values()
                            if r["hallucinated"]) / len(base)

            for level in LEVELS:
                comp, errs = load_cell(model_name(family, level), tier, "probe")
                if not comp:
                    continue
                shared = sorted(set(base) & set(comp))
                n = len(comp)
                hall = [1 if r["hallucinated"] else 0 for r in comp.values()]
                abst = [1 if r["abstained"] else 0 for r in comp.values()]
                rate = sum(hall) / n

                b = sum(1 for i in shared
                        if base[i]["hallucinated"] and not comp[i]["hallucinated"])
                c = sum(1 for i in shared
                        if comp[i]["hallucinated"] and not base[i]["hallucinated"])
                p, ndisc = mcnemar_exact(b, c)

                pairs = [(1.0 if comp[i]["hallucinated"] else 0.0,
                          1.0 if base[i]["hallucinated"] else 0.0) for i in shared]
                ci = boot_ci(hall)

                rows.append({
                    "family": family, "tier": tier, "level": level,
                    "n_items": n,
                    "hallucination_rate": round(rate, 5),
                    "hallucination_ci": ci,
                    "abstention_rate": round(sum(abst) / n, 5),
                    "baseline_rate": round(base_rate, 5),
                    "delta_pp": round((rate - base_rate) * 100, 3),
                    "delta_ci_pp": tuple(None if v is None else round(v * 100, 3)
                                         for v in paired_diff_ci(pairs)),
                    "p_raw": None if level == BASELINE else round(p, 6),
                    "n_discordant": ndisc,
                    # criterion 2: upper bound of the CI must not exceed the
                    # baseline point estimate
                    "criterion2_pass": (None if level == BASELINE
                                        else (ci[1] is not None
                                              and ci[1] <= base_rate)),
                    "dropped_rows": errs,
                })
                if level != BASELINE:
                    raw_p.append(((family, tier, level), p))

    by_tier = defaultdict(list)
    for key, p in raw_p:
        by_tier[key[1]].append((key, p))
    adjusted = {}
    for tier, pairs in by_tier.items():
        adjusted.update(holm(pairs))
    for r in rows:
        k = (r["family"], r["tier"], r["level"])
        r["p_holm"] = None if k not in adjusted else round(adjusted[k], 6)
    return rows


# ---------------------------------------------------------------- RQ4 + rule


def energy_and_rule(acc_rows, hall_rows):
    """Net energy ratios and the pre-registered deployability rule.

    Criterion 4 (AURAC at 20% rejection) needs semantic entropy and is left
    unresolved here; analyse_entropy.py fills it. A configuration is reported
    as PASS only when all four criteria are decided and met.
    """
    acc = {(r["family"], r["tier"], r["level"]): r for r in acc_rows}
    hal = {(r["family"], r["tier"], r["level"]): r for r in hall_rows}

    # criterion 4 arrives from analyse_entropy.py when it has been run
    ent = {}
    ep = OUTJ / "entropy_results.json"
    if ep.exists():
        blob = json.loads(ep.read_text(encoding="utf-8"))
        key = f"acc_at_{int(blob.get('reject_point', 0.20) * 100)}pct_rejection"
        for c in blob.get("cells", []):
            ent[(c["model"], c["tier"])] = c.get(key)
        print(f"         criterion 4 from {ep.name} "
              f"({blob.get('backend')}, {len(ent)} cells)")

    rows = []
    for family in FAMILIES:
        for tier in TIERS:
            base_e = {m: load_energy(model_name(family, BASELINE), tier, m)
                      for m in ("mcq", "probe", "open")}
            for level in LEVELS:
                e = {m: load_energy(model_name(family, level), tier, m)
                     for m in ("mcq", "probe", "open")}
                ratios = {}
                for m in ("mcq", "probe", "open"):
                    if e[m] and base_e[m] and base_e[m]["j_per_item"]:
                        ratios[m] = round(e[m]["j_per_item"]
                                          / base_e[m]["j_per_item"], 4)
                    else:
                        ratios[m] = None

                k = (family, tier, level)
                a, h = acc.get(k), hal.get(k)

                c1 = None if a is None else (a["delta_pp"] >= -ACC_TOLERANCE_PP)
                c2 = None if h is None else h["criterion2_pass"]
                c3 = (None if ratios["mcq"] is None
                      else ratios["mcq"] <= ENERGY_CEILING)
                a4 = ent.get((model_name(family, level), tier))
                b4 = ent.get((model_name(family, BASELINE), tier))
                c4 = (None if (a4 is None or b4 is None)
                      else (a4 >= b4))         # AURAC at the rejection point

                decided = [x for x in (c1, c2, c3, c4) if x is not None]
                if level == BASELINE:
                    # the baseline is the reference the rule is measured
                    # against, not a candidate to be judged by it
                    verdict = "baseline"
                else:
                    verdict = ("PASS" if (len(decided) == 4 and all(decided))
                               else "FAIL" if any(x is False for x in decided)
                               else "UNDECIDED")

                rows.append({
                    "family": family, "tier": tier, "level": level,
                    "j_per_item_mcq": None if not e["mcq"] else e["mcq"]["j_per_item"],
                    "j_per_item_probe": None if not e["probe"] else e["probe"]["j_per_item"],
                    "j_per_item_open": None if not e["open"] else e["open"]["j_per_item"],
                    "j_per_token_mcq": None if not e["mcq"] else e["mcq"].get("j_per_token"),
                    "ratio_mcq": ratios["mcq"],
                    "ratio_probe": ratios["probe"],
                    "ratio_open": ratios["open"],
                    "criterion1_accuracy": c1,
                    "criterion2_hallucination": c2,
                    "criterion3_energy": c3,
                    "criterion4_aurac": c4,
                    "verdict": verdict,
                    # the analysis plan treats a sub-5% energy gap as not established
                    "energy_difference_established": (
                        None if ratios["mcq"] is None
                        else abs(1 - ratios["mcq"]) >= 0.05),
                })
    return rows


# ---------------------------------------------------------------- determinism


def determinism():
    """Repeat 1 vs repeat 2 on the shared 500-item prefix.

    Greedy decoding should be deterministic. Any disagreement is a
    reproducibility finding and is reported as one, not averaged away.
    """
    rows = []
    for family in FAMILIES:
        for level in LEVELS:
            for tier in TIERS:
                for mode in ("mcq", "probe"):
                    m = model_name(family, level)
                    r1, _ = load_cell(m, tier, mode, 1)
                    r2, _ = load_cell(m, tier, mode, 2)
                    shared = sorted(set(r1) & set(r2))
                    if not shared:
                        continue
                    same = sum(1 for i in shared
                               if r1[i].get("pred_index") == r2[i].get("pred_index"))
                    rows.append({
                        "model": m, "tier": tier, "mode": mode,
                        "n_compared": len(shared),
                        "agreement": round(same / len(shared), 5),
                        "n_disagreements": len(shared) - same,
                    })
    return rows


# ---------------------------------------------------------------- output


def write_table(name, rows, columns, title):
    TABLES.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(TABLES / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c)) for c in columns})
    with open(TABLES / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "---|" * len(columns) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in columns) + " |\n")


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main():
    OUTJ.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    print("RQ1/RQ2  accuracy, McNemar, loyalty ...")
    acc = accuracy_and_fidelity()
    print("RQ3      probe hallucination ...")
    hal = hallucination()
    print("RQ4      energy and the deployability rule ...")
    ene = energy_and_rule(acc, hal)
    print("         determinism check ...")
    det = determinism()

    write_table(
        "table1_accuracy", acc,
        ["family", "tier", "level", "n_items", "accuracy", "accuracy_ci",
         "delta_pp", "delta_ci_pp", "mcnemar_b", "mcnemar_c", "n_discordant",
         "p_holm", "significant_05", "parse_rate"],
        "Table 1. Accuracy by model family, difficulty tier and compression level")

    write_table(
        "table2_loyalty", [r for r in acc if r["level"] != BASELINE],
        ["family", "tier", "level", "n_shared", "loyalty", "loyalty_ci"],
        "Table 2. Answer fidelity (loyalty) to the FP16 baseline")

    write_table(
        "table3_hallucination", hal,
        ["family", "tier", "level", "n_items", "hallucination_rate",
         "hallucination_ci", "abstention_rate", "delta_pp", "delta_ci_pp",
         "p_holm", "criterion2_pass"],
        "Table 3. Confident error and abstention on the probe set")

    write_table(
        "table4_deployability", ene,
        ["family", "tier", "level", "j_per_item_mcq", "j_per_item_open",
         "ratio_mcq", "ratio_open", "energy_difference_established",
         "criterion1_accuracy", "criterion2_hallucination",
         "criterion3_energy", "criterion4_aurac", "verdict"],
        "Table 4. Net energy and the pre-registered deployability rule")

    write_table(
        "table5_determinism", det,
        ["model", "tier", "mode", "n_compared", "agreement", "n_disagreements"],
        "Table 5. Greedy-decoding agreement between repeat 1 and repeat 2")

    out = {
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "seed": SEED, "bootstrap_resamples": BOOT,
        "rule": {"accuracy_tolerance_pp": ACC_TOLERANCE_PP,
                 "energy_ceiling": ENERGY_CEILING,
                 "criterion4_source": "analyse_entropy.py (pending)"},
        "accuracy_fidelity": acc,
        "hallucination": hal,
        "energy_rule": ene,
        "determinism": det,
    }
    (OUTJ / "core_results.json").write_text(json.dumps(out, indent=2),
                                            encoding="utf-8")

    # ---- console summary -------------------------------------------------
    print(f"\nwrote {OUTJ / 'core_results.json'}")
    print(f"wrote 5 tables -> {TABLES}\n")

    dis = [d for d in det if d["n_disagreements"]]
    print(f"determinism : {len(det)-len(dis)}/{len(det)} cells fully reproducible")
    if dis:
        for d in dis[:6]:
            print(f"   {d['model']:28s} {d['tier']:8s} {d['mode']:5s} "
                  f"{d['n_disagreements']} disagreements")

    print("\naccuracy vs FP16 (pp, negative = worse):")
    for r in acc:
        if r["level"] == BASELINE:
            continue
        sig = "*" if r["significant_05"] else " "
        print(f"   {r['family']:22s} {r['tier']:8s} {r['level']:8s} "
              f"{r['delta_pp']:+7.2f} pp {sig}  loyalty {r['loyalty']:.3f}")

    print("\ndeployability verdicts:")
    tally = defaultdict(int)
    for r in ene:
        if r["level"] == BASELINE:
            continue
        tally[r["verdict"]] += 1
    for k, v in sorted(tally.items()):
        print(f"   {k:10s} {v}")
    if not (OUTJ / "entropy_results.json").exists():
        print("\ncriterion 4 (AURAC) is undecided until analyse_entropy.py runs.")
    else:
        c4_fail = [r for r in ene if r["level"] != BASELINE
                   and r["criterion4_aurac"] is False]
        print(f"\ncriterion 4 (AURAC) decided for all cells; "
              f"{len(c4_fail)} configurations fail on it")


if __name__ == "__main__":
    main()
