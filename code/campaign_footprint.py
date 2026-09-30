#!/usr/bin/env python3
"""
campaign_footprint.py  --  ET&S study, step 7

The study's own energy cost, reported because a Green AI paper that conceals
its footprint is self-refuting (methodology section 5.4).

Two figures are produced and kept apart, because they have different evidential
status:

  MEASURED   the 81 energy cells, where package power was logged the whole time
             and idle was subtracted. This is measurement.
  ESTIMATED  the 135-cell accuracy campaign and the pilot, which ran before the
             power log existed. Their energy is inferred from measured joules
             per item for the same model, tier and mode. This is extrapolation,
             and it is labelled as such everywhere it appears.

The pragmatic scaling factor (PSF, Lannelongue et al., 2021) covers the whole
experimental effort, including the discarded 2026-09-28 entropy run and the
pilots, not only the runs that survived into the paper.

Run:  py campaign_footprint.py
"""

import json
from pathlib import Path

ROOT = Path(r"D:\claude_projects\ET&S")
ENERGY = ROOT / "06-results" / "energy"
RAW = ROOT / "06-results" / "raw"
ANALYSIS = ROOT / "06-results" / "analysis"

# Green Algorithms parameters (Lannelongue et al., 2021)
PUE = 1.0                 # single personal machine, not a data centre
MEM_W_PER_GB = 0.3725
MEM_GB = 16
IRAQ_GRID_GCO2_PER_KWH = 610.0   # stated as a parameter, cited in the paper


def measured():
    """Every joule actually logged, from the 81 energy cells."""
    cells, total_j, total_s, total_items, total_tok = [], 0.0, 0.0, 0, 0
    rates = {}                      # (model, tier, mode) -> J per item
    for p in sorted(ENERGY.glob("*.json")):
        if p.name.startswith(("session__", "summary")):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        e = r.get("energy")
        if not e:
            continue
        total_j += e["net_total_j"]
        total_s += r["active_s"]
        total_items += r["items"]
        total_tok += r.get("tokens_generated") or 0
        rates[(r["model"], r["tier"], r["mode"])] = e["j_per_item"]
        cells.append(r["cell"])
    return {"cells": len(cells), "net_j": round(total_j, 1),
            "active_s": round(total_s, 1), "items": total_items,
            "tokens": total_tok, "rates": rates}


def estimated(rates):
    """The accuracy campaign, priced at the measured rate for the same cell.

    Every row here is an inference, not a measurement: these runs finished
    before the power log was started.
    """
    total_j, total_items, missing = 0.0, 0, []
    for p in sorted(RAW.glob("*.jsonl")):
        stem = p.stem                       # model__tier__mode__rN
        parts = stem.split("__")
        if len(parts) != 4:
            continue
        model_safe, tier, mode, _rep = parts
        model = model_safe.replace("_", ":", 1) if "_" in model_safe else model_safe
        n = sum(1 for _ in open(p, encoding="utf-8"))
        rate = rates.get((model, tier, mode))
        if rate is None:
            missing.append(stem)
            continue
        total_j += rate * n
        total_items += n
    return {"net_j": round(total_j, 1), "items": total_items,
            "cells_without_a_rate": missing}


def memory_energy(seconds):
    """Green Algorithms memory term: n_m x P_m x t."""
    return MEM_GB * MEM_W_PER_GB * seconds


def to_kwh(joules):
    return joules / 3.6e6


def co2e_g(kwh):
    return kwh * IRAQ_GRID_GCO2_PER_KWH


def main():
    m = measured()
    e = estimated(m["rates"])

    # compute time, for the memory term: measured cells plus the accuracy
    # campaign, whose duration comes from its own logs
    camp_s = 0.0
    for p in sorted((ROOT / "06-results" / "logs").glob("*.json")):
        try:
            camp_s += json.loads(p.read_text(encoding="utf-8"))["elapsed_s"]
        except Exception:
            pass

    cpu_j = m["net_j"] + e["net_j"]
    mem_j = memory_energy(m["active_s"] + camp_s)
    total_j = cpu_j + mem_j
    kwh = to_kwh(total_j) * PUE
    g = co2e_g(kwh)

    # PSF: the whole effort, including what was discarded
    discarded = {
        "entropy run 2026-09-28 (grader defect, results discarded)": 8.65,
        "entropy smoke tests": 0.9,
        "timing and build pilots": 1.5,
    }
    discarded_h = sum(discarded.values())
    productive_h = (m["active_s"] + camp_s) / 3600
    psf = round((productive_h + discarded_h) / productive_h, 3)

    out = {
        "status": {
            "measured": "81 energy cells, package power logged, idle subtracted",
            "estimated": "accuracy campaign priced at the measured per-item rate "
                         "for the same model, tier and mode; not a measurement",
        },
        "measured": {k: v for k, v in m.items() if k != "rates"},
        "estimated_accuracy_campaign": e,
        "memory_term": {"gb": MEM_GB, "w_per_gb": MEM_W_PER_GB,
                        "seconds": round(m["active_s"] + camp_s, 1),
                        "joules": round(mem_j, 1)},
        "totals": {
            "cpu_j": round(cpu_j, 1),
            "memory_j": round(mem_j, 1),
            "total_j": round(total_j, 1),
            "kwh": round(kwh, 4),
            "pue": PUE,
            "grid_gco2_per_kwh": IRAQ_GRID_GCO2_PER_KWH,
            "co2e_g": round(g, 1),
            "co2e_kg": round(g / 1000, 4),
        },
        "psf": {"productive_hours": round(productive_h, 2),
                "discarded_hours": discarded_h,
                "discarded_detail": discarded,
                "pragmatic_scaling_factor": psf,
                "co2e_g_including_discarded": round(g * psf, 1)},
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "campaign_footprint.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print(f"measured   {m['cells']} cells   {m['net_j']:,.0f} J   "
          f"{m['items']:,} items")
    print(f"estimated  accuracy campaign   {e['net_j']:,.0f} J   "
          f"{e['items']:,} items")
    if e["cells_without_a_rate"]:
        print(f"           {len(e['cells_without_a_rate'])} cells had no "
              "measured rate and are excluded")
    print(f"memory     {mem_j:,.0f} J over "
          f"{(m['active_s']+camp_s)/3600:.1f} h")
    print(f"\ntotal      {total_j:,.0f} J = {kwh:.4f} kWh")
    print(f"CO2e       {g:,.1f} g at {IRAQ_GRID_GCO2_PER_KWH} gCO2/kWh")
    print(f"PSF        {psf}  ->  {g*psf:,.1f} g including discarded work")
    print(f"\nwrote {ANALYSIS / 'campaign_footprint.json'}")


if __name__ == "__main__":
    main()
