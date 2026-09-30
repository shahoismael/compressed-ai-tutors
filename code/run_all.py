#!/usr/bin/env python3
"""
run_all.py  --  ET&S study, campaign driver

Walks every cell in a fixed, randomised order and runs the ones not yet
finished. Safe to stop with Ctrl+C and restart: each cell resumes from its
own last completed item.

  py run_all.py                 run the next 10 cells, then stop
  py run_all.py --status        show progress, run nothing
  py run_all.py --only mcq      restrict to one mode
  py run_all.py --max-cells 5   stop after five cells (overnight chunks)
"""

import argparse
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import run_cell as rc

ORDER_SEED = 20260921  # cell order fixed once, so the sequence is reproducible
STATE = rc.ROOT / "06-results" / "campaign_state.json"


def all_cells():
    cells = []
    for m in rc.MODELS:
        for t in rc.TIERS:
            for mode, reps in (("mcq", rc.N_REPEATS), ("probe", rc.N_REPEATS), ("open", 1)):
                for r in range(1, reps + 1):
                    cells.append({"model": m, "tier": t, "mode": mode, "repeat": r})
    # Randomised execution order decorrelates thermal drift from condition.
    random.Random(ORDER_SEED).shuffle(cells)
    return cells


def expected_count(cell):
    return len(rc.load_items(cell["tier"], cell["mode"], cell["repeat"]))


def done_count(cell):
    p = rc.RAW / (rc.cell_name(**cell) + ".jsonl")
    if not p.exists():
        return 0
    n = 0
    with open(p, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def status():
    cells = all_cells()
    tot = fin = items_done = items_tot = 0
    pending = []
    for c in cells:
        exp = expected_count(c)
        got = done_count(c)
        tot += 1
        items_tot += exp
        items_done += min(got, exp)
        if got >= exp:
            fin += 1
        else:
            pending.append((c, got, exp))
    print(f"cells   {fin}/{tot} complete")
    print(f"items   {items_done}/{items_tot}  ({100*items_done/max(items_tot,1):.1f}%)")
    if pending:
        print(f"\nnext up ({min(8,len(pending))} of {len(pending)}):")
        for c, got, exp in pending[:8]:
            print(f"  {c['model']:28s} {c['tier']:8s} {c['mode']:5s} r{c['repeat']}  {got}/{exp}")
    return pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", choices=["mcq", "probe", "open"])
    ap.add_argument("--max-cells", type=int, default=10)
    a = ap.parse_args()

    if a.status:
        status()
        return

    try:
        import urllib.request

        urllib.request.urlopen(f"{rc.OLLAMA}/api/tags", timeout=10)
    except Exception as e:
        sys.exit(f"cannot reach ollama at {rc.OLLAMA}: {e}")

    cells = all_cells()
    if a.only:
        cells = [c for c in cells if c["mode"] == a.only]

    todo = [c for c in cells if done_count(c) < expected_count(c)]
    print(f"{len(todo)} cells outstanding of {len(cells)}\n")

    ran = 0
    t0 = time.perf_counter()
    for c in todo:
        if a.max_cells and ran >= a.max_cells:
            print(f"\nreached --max-cells {a.max_cells}, stopping cleanly")
            break
        print("=" * 68)
        print(f"[{ran+1}/{len(todo)}]  {c['model']}  {c['tier']}  {c['mode']}  r{c['repeat']}")
        try:
            rc.run(c["model"], c["tier"], c["mode"], c["repeat"])
        except KeyboardInterrupt:
            print("\ninterrupted — progress for this cell is saved, rerun to resume")
            break
        except Exception:
            print("CELL FAILED, continuing:")
            traceback.print_exc()
        ran += 1

        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            json.dumps(
                {
                    "last_cell": c,
                    "cells_run_this_session": ran,
                    "elapsed_h": round((time.perf_counter() - t0) / 3600, 2),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("\n" + "=" * 68)
    print(f"ran {ran} cells in {(time.perf_counter()-t0)/3600:.2f} h")
    status()


if __name__ == "__main__":
    main()
