#!/usr/bin/env python3
"""
run_energy.py  --  ET&S study, step 3 (energy)

Re-runs a fixed subset of the already-measured items and records an exact
wall-clock window for every cell. HWiNFO logs package power to CSV in the
background; hwinfo_align.py then integrates the CSV over each window to give
a MEASURED joules-per-item figure. No TDP estimates anywhere.

Item ids are a prefix of the same items.jsonl, so these rows join directly
onto the accuracy and hallucination results in 06-results/raw.

BEFORE RUNNING
  1. HWiNFO -> Sensors
  2. bottom toolbar -> Logging Start -> save as
       D:\\claude_projects\\ET&S\\06-results\\energy\\hwinfo.csv
  3. Settings -> Logging -> interval 1000 ms (or 500)
  4. leave HWiNFO logging for the WHOLE session, stop it only at the end

  py run_energy.py --status        progress only
  py run_energy.py                 run the next 15 cells, then stop
  py run_energy.py --max-cells 5   shorter chunk

Safe to Ctrl+C between cells; finished cells are skipped on restart.
"""

import argparse
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone

import run_cell as rc

ORDER_SEED = 20260921
ENERGY_N = {"mcq": 100, "probe": 60, "open": 20}   # items per cell, per mode
IDLE_SECONDS = 120        # idle baseline, start and end of every session
COOLDOWN_S = 60           # between cells, so thermals settle
WARMUP = 3                # items run before the measured window opens

OUT = rc.ROOT / "06-results" / "energy"
RAWE = OUT / "raw"


def now():
    return time.time()


def cells():
    cs = [
        {"model": m, "tier": t, "mode": mode}
        for m in rc.MODELS
        for t in rc.TIERS
        for mode in ("mcq", "probe", "open")
    ]
    random.Random(ORDER_SEED).shuffle(cs)
    return cs


def cell_name(c):
    safe = c["model"].replace(":", "_").replace("/", "_")
    return f"{safe}__{c['tier']}__{c['mode']}"


def items_for(c):
    return rc.load_items(c["tier"], c["mode"], 1)[: ENERGY_N[c["mode"]]]


def done(c):
    return (OUT / f"{cell_name(c)}.json").exists()


def idle_window(seconds, label):
    print(f"{label} idle baseline: {seconds}s — leave the machine alone ...")
    t0 = now()
    time.sleep(seconds)
    t1 = now()
    return {"label": label, "t_start_epoch": t0, "t_end_epoch": t1,
            "seconds": seconds,
            "t_start_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
            "t_end_utc": datetime.fromtimestamp(t1, timezone.utc).isoformat()}


def one_item(model, mode, it):
    """Run a single item; returns (record_fields, tokens_generated)."""
    if mode == "mcq":
        txt, m = rc.ollama(model, rc.mcq_prompt(it), 0.0, 8)
        pred = rc.parse_letter(txt, len(it["options"]))
        return ({"pred_index": pred, "correct": pred == it["answer_index"], **m},
                m.get("eval_count") or 0)

    if mode == "probe":
        msgs, kept = rc.probe_prompt(it)
        txt, m = rc.ollama(model, msgs, 0.0, 8)
        up = txt.strip().upper()
        abst = "IDK" in up or "DON'T KNOW" in up or "DO NOT KNOW" in up
        pred = None if abst else rc.parse_letter(txt, len(kept))
        return ({"abstained": abst,
                 "hallucinated": (not abst and pred is not None), **m},
                m.get("eval_count") or 0)

    walls, toks = [], 0
    for k in range(rc.OPEN_SAMPLES):
        _, m = rc.ollama(model, rc.open_prompt(it), 1.0, 96, seed=rc.SEED + k)
        walls.append(m["wall_s"])
        toks += m.get("eval_count") or 0
    return ({"wall_s": round(sum(walls), 4), "n_samples": rc.OPEN_SAMPLES}, toks)


def run_one(c):
    name = cell_name(c)
    items = items_for(c)
    print(f"cell   : {name}")
    print(f"items  : {len(items)}")

    # warm-up: model load and cache fill stay OUTSIDE the measured window
    for it in items[:WARMUP]:
        try:
            rc.ollama(c["model"], rc.mcq_prompt(it), 0.0, 8)
        except Exception:
            pass
    time.sleep(2)  # let the load transient settle before the window opens

    t_start = now()
    tokens, errors = 0, 0
    with open(RAWE / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for n, it in enumerate(items, 1):
            rec = {"item_id": it["item_id"], "tier": c["tier"],
                   "mode": c["mode"], "model": c["model"]}
            try:
                body, tk = one_item(c["model"], c["mode"], it)
                rec.update(body)
                tokens += tk
                rec["error"] = None
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                errors += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(items):
                print(f"  {n}/{len(items)}  "
                      f"{(now()-t_start)/n:.2f} s/item", flush=True)
    t_end = now()
    return name, items, tokens, errors, t_start, t_end


def write_cell(c, name, items, tokens, errors, t_start, t_end):
    """Per-cell record. Energy fields stay null until hwinfo_align.py runs."""
    n = max(len(items), 1)
    rec = {
        "cell": name,
        "model": c["model"],
        "tier": c["tier"],
        "mode": c["mode"],
        "items": len(items),
        "errors": errors,
        "tokens_generated": tokens,
        "t_start_epoch": t_start,
        "t_end_epoch": t_end,
        "t_start_utc": datetime.fromtimestamp(t_start, timezone.utc).isoformat(),
        "t_end_utc": datetime.fromtimestamp(t_end, timezone.utc).isoformat(),
        "active_s": round(t_end - t_start, 3),
        "s_per_item": round((t_end - t_start) / n, 4),
        "open_samples": rc.OPEN_SAMPLES if c["mode"] == "open" else None,
        "seed": rc.SEED,
        "warmup_items": WARMUP,
        # filled by hwinfo_align.py from the HWiNFO CSV
        "energy": None,
    }
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def status():
    cs = cells()
    fin = [c for c in cs if done(c)]
    left = [c for c in cs if not done(c)]
    print(f"energy cells  {len(fin)}/{len(cs)} complete")
    aligned = 0
    for c in fin:
        try:
            if json.loads((OUT / f"{cell_name(c)}.json").read_text(encoding="utf-8"))["energy"]:
                aligned += 1
        except Exception:
            pass
    print(f"energy joined {aligned}/{len(fin)}  (run hwinfo_align.py for the rest)")
    for c in left[:8]:
        print(f"  {c['model']:28s} {c['tier']:8s} {c['mode']}")
    if len(left) > 8:
        print(f"  ... {len(left) - 8} more")
    return left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-hours", type=float, default=5.5,
                    help="wall-clock budget for this session (default 5.5, "
                         "sized for a 00:00-06:00 overnight window)")
    ap.add_argument("--max-cells", type=int, default=0, help="0 = no cell cap")
    ap.add_argument("--idle-seconds", type=int, default=IDLE_SECONDS)
    a = ap.parse_args()

    if a.status:
        status()
        return

    try:
        import urllib.request
        urllib.request.urlopen(f"{rc.OLLAMA}/api/tags", timeout=10)
    except Exception as e:
        sys.exit(f"cannot reach ollama at {rc.OLLAMA}: {e}")

    OUT.mkdir(parents=True, exist_ok=True)
    RAWE.mkdir(parents=True, exist_ok=True)

    todo = [c for c in cells() if not done(c)]
    if not todo:
        print("all energy cells complete")
        status()
        return

    budget = a.max_hours * 3600
    print(f"{len(todo)} energy cells outstanding")
    print(f"budget  : {a.max_hours} h — stops cleanly before overrunning")
    print("HWiNFO  : logging must already be running to "
          "06-results/energy/hwinfo.csv\n")

    t0 = time.perf_counter()
    idle_start = idle_window(a.idle_seconds, "start")

    ran, durations = 0, []
    for c in todo:
        if a.max_cells and ran >= a.max_cells:
            print(f"\nreached --max-cells {a.max_cells}, stopping cleanly")
            break
        elapsed = time.perf_counter() - t0
        guard = max(durations) if durations else 900.0     # 15 min for cell one
        if elapsed + guard + a.idle_seconds > budget:
            print(f"\nbudget reached ({elapsed/3600:.2f} h of {a.max_hours} h) — "
                  "stopping before the next cell")
            break

        print("=" * 68)
        print(f"[{ran+1}]  {c['model']}  {c['tier']}  {c['mode']}   "
              f"elapsed {elapsed/3600:.2f} h")
        try:
            name, items, tokens, errors, ts, te = run_one(c)
        except KeyboardInterrupt:
            print("\ninterrupted — this cell is discarded, rerun to redo it")
            break
        except Exception:
            print("CELL FAILED, continuing:")
            traceback.print_exc()
            continue

        rec = write_cell(c, name, items, tokens, errors, ts, te)
        durations.append(rec["active_s"] + COOLDOWN_S)
        print(f"  {rec['active_s']/60:.1f} min   {rec['s_per_item']} s/item   "
              f"{tokens} tokens")
        ran += 1
        if time.perf_counter() - t0 + COOLDOWN_S < budget:
            print(f"  cooldown {COOLDOWN_S}s ...")
            time.sleep(COOLDOWN_S)

    idle_end = idle_window(a.idle_seconds, "end")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (OUT / f"session__{stamp}.json").write_text(
        json.dumps({"idle_start": idle_start, "idle_end": idle_end,
                    "cells_run": ran,
                    "session_hours": round((time.perf_counter() - t0) / 3600, 3)},
                   indent=2),
        encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"ran {ran} cells in {(time.perf_counter()-t0)/3600:.2f} h")
    print("stop HWiNFO logging, then run:  py hwinfo_align.py")
    status()


if __name__ == "__main__":
    main()
