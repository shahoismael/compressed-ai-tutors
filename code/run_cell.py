#!/usr/bin/env python3
"""
run_cell.py  --  ET&S study, step 2

Runs ONE cell: one model x one tier x one mode x one repeat.
Resumable: results are appended after every item, so a crash costs one item.

Modes
  mcq    forced choice, greedy, deterministic   -> correctness + fidelity
  probe  correct option removed, abstention allowed -> hallucination
  open   no options, 10 samples at T=1.0        -> semantic entropy

Examples
  py run_cell.py --model gemma2:2b-instruct-q4_K_M --tier school --mode mcq --repeat 1
  py run_cell.py --model gemma2:2b-instruct-q4_K_M --tier school --mode mcq --repeat 1 --limit 50
  py run_cell.py --list-cells
"""

import argparse
import json
import platform
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- config

ROOT = Path(r"D:\claude_projects\ET&S")
ITEMS = ROOT / "02-datasets" / "00-item-sets" / "items.jsonl"
RAW = ROOT / "06-results" / "raw"
LOGS = ROOT / "06-results" / "logs"

OLLAMA = "http://127.0.0.1:11434"
SEED = 20260921

MODELS = [
    "qwen2.5:3b-instruct-fp16",
    "qwen2.5:3b-instruct-q8_0",
    "qwen2.5:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-fp16",
    "llama3.2:3b-instruct-q8_0",
    "llama3.2:3b-instruct-q4_K_M",
    "gemma2:2b-instruct-fp16",
    "gemma2:2b-instruct-q8_0",
    "gemma2:2b-instruct-q4_K_M",
]
TIERS = ["school", "college", "expert"]
LETTERS = "ABCDEFGHIJ"

WARMUP = 3          # items discarded before timing starts
OPEN_SAMPLES = 5    # generations per open-ended item (was 10)
OPEN_PER_TIER = 100  # open-ended items actually run per tier (was 200)
REPEAT_SUBSET = 500  # repeats 2+ run on this many items, not the full set
N_REPEATS = 2        # repeats on the deterministic axes (was 3)

# ---------------------------------------------------------------- ollama


def ollama(model, messages, temperature, num_predict, seed=SEED):
    """One chat call. Returns (text, metrics_dict)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            "top_p": 1.0 if temperature > 0 else 1.0,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall = time.perf_counter() - t0

    return body.get("message", {}).get("content", ""), {
        "wall_s": round(wall, 4),
        "eval_count": body.get("eval_count"),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "total_duration_ns": body.get("total_duration"),
        "load_duration_ns": body.get("load_duration"),
    }


# ---------------------------------------------------------------- prompts


def mcq_prompt(item):
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(item["options"]))
    return [
        {
            "role": "system",
            "content": "Answer the multiple-choice question. "
            "Reply with the single letter of the correct option and nothing else.",
        },
        {"role": "user", "content": f"{item['question']}\n\n{opts}\n\nAnswer:"},
    ]


def probe_prompt(item):
    """Correct option removed; abstention explicitly permitted."""
    kept = [o for i, o in enumerate(item["options"]) if i != item["answer_index"]]
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(kept))
    return (
        [
            {
                "role": "system",
                "content": "Answer the multiple-choice question. "
                "If none of the options is correct, or you do not know, "
                'reply exactly "IDK". Otherwise reply with the single letter '
                "of the correct option and nothing else.",
            },
            {"role": "user", "content": f"{item['question']}\n\n{opts}\n\nAnswer:"},
        ],
        kept,
    )


def open_prompt(item):
    return [
        {
            "role": "system",
            "content": "Answer the question in one short sentence.",
        },
        {"role": "user", "content": item["question"]},
    ]


def parse_letter(text, n_options):
    """First standalone letter within range, else None."""
    m = re.search(r"\b([A-J])\b", text.strip().upper())
    if not m:
        m = re.match(r"\s*([A-J])", text.strip().upper())
    if not m:
        return None
    idx = LETTERS.index(m.group(1))
    return idx if idx < n_options else None


# ---------------------------------------------------------------- runner


def load_items(tier, mode, repeat):
    """Item list for a cell.

    repeat 1  -> the full set for that mode (the primary measurement)
    repeat 2+ -> a fixed prefix of REPEAT_SUBSET items (the determinism check)

    The prefix is deterministic because items.jsonl order is fixed by the
    build seed, so every model sees exactly the same subset.
    """
    items = []
    with open(ITEMS, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            if it["tier"] != tier:
                continue
            if mode == "probe" and not it["in_probe"]:
                continue
            if mode == "open" and not it["in_open"]:
                continue
            items.append(it)

    if mode == "open":
        items = items[:OPEN_PER_TIER]
    elif repeat > 1:
        items = items[:REPEAT_SUBSET]
    return items


def cell_name(model, tier, mode, repeat):
    safe = model.replace(":", "_").replace("/", "_")
    return f"{safe}__{tier}__{mode}__r{repeat}"


def run(model, tier, mode, repeat, limit=None):
    RAW.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    name = cell_name(model, tier, mode, repeat)
    out_path = RAW / f"{name}.jsonl"
    log_path = LOGS / f"{name}.json"

    items = load_items(tier, mode, repeat)
    if limit:
        items = items[:limit]

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["item_id"])
                except Exception:
                    pass
    todo = [it for it in items if it["item_id"] not in done]

    print(f"cell   : {name}")
    print(f"items  : {len(items)}   done: {len(done)}   todo: {len(todo)}")
    if not todo:
        print("nothing to do")
        return

    # warm-up, discarded
    print(f"warm-up: {WARMUP} items ...", end=" ", flush=True)
    for it in todo[:WARMUP]:
        try:
            ollama(model, mcq_prompt(it), 0.0, 8)
        except Exception:
            pass
    print("done")

    t_start = time.perf_counter()
    n = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for it in todo:
            rec = {
                "item_id": it["item_id"],
                "tier": tier,
                "mode": mode,
                "model": model,
                "repeat": repeat,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            try:
                if mode == "mcq":
                    txt, m = ollama(model, mcq_prompt(it), 0.0, 8)
                    pred = parse_letter(txt, len(it["options"]))
                    rec.update(
                        raw=txt.strip()[:200],
                        pred_index=pred,
                        gold_index=it["answer_index"],
                        correct=(pred == it["answer_index"]),
                        parsed=(pred is not None),
                        **m,
                    )

                elif mode == "probe":
                    msgs, kept = probe_prompt(it)
                    txt, m = ollama(model, msgs, 0.0, 8)
                    up = txt.strip().upper()
                    abstained = "IDK" in up or "DON'T KNOW" in up or "DO NOT KNOW" in up
                    pred = None if abstained else parse_letter(txt, len(kept))
                    rec.update(
                        raw=txt.strip()[:200],
                        abstained=abstained,
                        pred_index=pred,
                        hallucinated=(not abstained and pred is not None),
                        n_options_shown=len(kept),
                        **m,
                    )

                elif mode == "open":
                    gens, walls = [], []
                    for k in range(OPEN_SAMPLES):
                        txt, m = ollama(
                            model, open_prompt(it), 1.0, 96, seed=SEED + k
                        )
                        gens.append(txt.strip())
                        walls.append(m["wall_s"])
                    rec.update(
                        generations=gens,
                        gold_text=it["options"][it["answer_index"]],
                        wall_s=round(sum(walls), 4),
                        n_samples=OPEN_SAMPLES,
                    )
                else:
                    raise ValueError(mode)

                rec["error"] = None

            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            if n % 25 == 0 or n == len(todo):
                el = time.perf_counter() - t_start
                rate = el / n
                eta = rate * (len(todo) - n)
                print(
                    f"  {n}/{len(todo)}  {rate:.2f} s/item  eta {eta/60:.1f} min",
                    flush=True,
                )

    elapsed = time.perf_counter() - t_start
    log = {
        "cell": name,
        "model": model,
        "tier": tier,
        "mode": mode,
        "repeat": repeat,
        "seed": SEED,
        "temperature": 0.0 if mode != "open" else 1.0,
        "open_samples": OPEN_SAMPLES if mode == "open" else None,
        "open_per_tier": OPEN_PER_TIER if mode == "open" else None,
        "repeat_subset_size": REPEAT_SUBSET if (repeat > 1 and mode != "open") else None,
        "is_full_set": (repeat == 1 or mode == "open"),
        "warmup_items": WARMUP,
        "items_total": len(items),
        "items_run_this_session": n,
        "elapsed_s": round(elapsed, 2),
        "s_per_item": round(elapsed / max(n, 1), 3),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "ollama_url": OLLAMA,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\ndone in {elapsed/60:.1f} min  ({elapsed/max(n,1):.2f} s/item)")
    print(f"raw -> {out_path}")
    print(f"log -> {log_path}")


# ---------------------------------------------------------------- cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tier", choices=TIERS)
    ap.add_argument("--mode", choices=["mcq", "probe", "open"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--list-cells", action="store_true")
    a = ap.parse_args()

    if a.list_cells:
        n = 0
        for m in MODELS:
            for t in TIERS:
                for mode, reps in (("mcq", N_REPEATS), ("probe", N_REPEATS), ("open", 1)):
                    for r in range(1, reps + 1):
                        print(
                            f"py run_cell.py --model {m} --tier {t} "
                            f"--mode {mode} --repeat {r}"
                        )
                        n += 1
        print(f"\n{n} cells total")
        return

    if not (a.model and a.tier and a.mode):
        ap.error("--model, --tier and --mode are required")

    try:
        urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10)
    except Exception as e:
        sys.exit(f"cannot reach ollama at {OLLAMA}: {e}")

    run(a.model, a.tier, a.mode, a.repeat, a.limit)


if __name__ == "__main__":
    main()
