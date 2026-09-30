#!/usr/bin/env python3
"""
make_schematic.py  --  ET&S study, Figure 1 (study design)

Left-to-right flow: sources -> fixed item set -> conditions -> measurement ->
the pre-registered rule. Counts are read from item_counts.json, so the figure
cannot drift from the manuscript.

Vector PDF plus a 600 dpi PNG, because ET&S accepts PNG or JPEG only.

Run:  py make_schematic.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle

ROOT = Path(r"D:\claude_projects\ET&S")
COUNTS = ROOT / "02-datasets" / "00-item-sets" / "item_counts.json"
FIGS = ROOT / "07-figs"

MM = 1 / 25.4
FULL = 165 * MM

# open-ended items actually RUN per tier (200 were marked, 100 were run —
# methodology section 13). The figure must state what was run.
OPEN_RUN = 100

GREY = "#4D4D4D"        # outlines and text
OURS = "#0072B2"        # the study's own contribution
BASE = "#E69F00"        # the comparison baseline
FILL_OURS = "#E3F0F8"
FILL_BASE = "#FDF2E0"
FILL_N = "#F2F2F2"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ---- grid: every box sits on it, every gap is equal ------------------------
COL_W, ROW_H = 30.0, 9.5
GAP_X, GAP_Y = 8.0, 4.0
COLS = [0, COL_W + GAP_X, 2 * (COL_W + GAP_X), 3 * (COL_W + GAP_X)]


def box(ax, x, y, w, h, label, sub=None, edge=GREY, fill="white", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                linewidth=lw, edgecolor=edge, facecolor=fill,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2 + (1.1 if sub else 0), label,
            ha="center", va="center", fontsize=7, color=GREY, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.5, sub, ha="center", va="center",
                fontsize=6, color=GREY, zorder=3)


def arrow(ax, x0, y0, x1, y1, label=None, dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=GREY, linewidth=0.8,
                                linestyle="--" if dashed else "-",
                                shrinkA=0, shrinkB=0,
                                mutation_scale=7), zorder=1)
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.0, label, ha="center",
                va="bottom", fontsize=5.5, color=GREY, zorder=3)


def band(ax, x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FAFAFA",
                           edgecolor="none", zorder=0))
    ax.text(x + w / 2, y + h + 1.0, title, ha="center", va="bottom",
            fontsize=7, color=GREY, fontweight="bold", zorder=3)


def main():
    c = json.loads(COUNTS.read_text(encoding="utf-8"))
    tiers = c["counts"]
    total = c["total_items"]
    dedup = ROOT / "02-datasets" / "00-item-sets" / "dedup_report.json"
    removed = json.loads(dedup.read_text(encoding="utf-8"))["removed"] \
        if dedup.exists() else 0

    fig, ax = plt.subplots(figsize=(FULL, 78 * MM))
    ax.set_xlim(-2, COLS[3] + COL_W + 2)
    ax.set_ylim(-8, 40)
    ax.axis("off")

    rows = [26.0, 26.0 - (ROW_H + GAP_Y), 26.0 - 2 * (ROW_H + GAP_Y)]
    src = [("ARC-Challenge", "school", "school"),
           ("MMLU chem/eng", "college", "college"),
           ("ChemBench", "expert", "expert")]

    band(ax, COLS[0] - 1.5, rows[2] - 2, COL_W + 3,
         rows[0] - rows[2] + ROW_H + 4, "Sources")
    band(ax, COLS[1] - 1.5, rows[2] - 2, COL_W + 3,
         rows[0] - rows[2] + ROW_H + 4, "Fixed item set")

    # ---- column 1: sources, with the pool each was drawn from --------------
    for i, (name, tier, key) in enumerate(src):
        pool = tiers[key]["pool_after_dedup"]
        box(ax, COLS[0], rows[i], COL_W, ROW_H, name,
            f"test split, pool n = {pool:,}", fill=FILL_N)

    # ---- column 2: what was kept, and what was dropped ---------------------
    for i, (_, tier, key) in enumerate(src):
        t = tiers[key]
        box(ax, COLS[1], rows[i], COL_W, ROW_H, tier.capitalize() + " tier",
            f"n = {t['sampled']:,}   {t['probe']} probe · {OPEN_RUN} open",
            edge=OURS, fill=FILL_OURS)
        arrow(ax, COLS[0] + COL_W, rows[i] + ROW_H / 2,
              COLS[1], rows[i] + ROW_H / 2)


    # ---- column 3: the conditions, stacked as alternatives ----------------
    band(ax, COLS[2] - 1.5, rows[2] - 2, COL_W + 3,
         rows[0] - rows[2] + ROW_H + 4, "Conditions")
    cond = [("FP16", "baseline", BASE, FILL_BASE),
            ("q8_0", "8-bit", OURS, FILL_OURS),
            ("q4_K_M", "4-bit", OURS, FILL_OURS)]
    for i, (name, sub, edge, fill) in enumerate(cond):
        box(ax, COLS[2], rows[i], COL_W, ROW_H, name, sub,
            edge=edge, fill=fill)
    arrow(ax, COLS[1] + COL_W, rows[1] + ROW_H / 2,
          COLS[2], rows[1] + ROW_H / 2)

    # ---- column 4: what was measured on those items -----------------------
    band(ax, COLS[3] - 1.5, rows[2] - 2, COL_W + 3,
         rows[0] - rows[2] + ROW_H + 4, "Measured")
    meas = [("Correctness", "and fidelity to FP16"),
            ("Confident error", "probe + semantic entropy"),
            ("Net energy", "package power, idle subtracted")]
    for i, (name, sub) in enumerate(meas):
        box(ax, COLS[3], rows[i], COL_W, ROW_H, name, sub,
            edge=OURS, fill=FILL_OURS)
    # one arrow, not three: every axis is measured on every condition, so
    # three arrows would cross and would also misstate the design
    arrow(ax, COLS[2] + COL_W, rows[1] + ROW_H / 2,
          COLS[3], rows[1] + ROW_H / 2)

    # ---- notes, each on its own line under its own column -----------------
    ax.text(COLS[1] + COL_W / 2, rows[2] - 3.0,
            f"{removed} duplicates removed · {total:,} items fixed once",
            ha="center", va="top", fontsize=6, color=GREY)
    ax.text(COLS[2] + COL_W / 2 + (COL_W + GAP_X) / 2, rows[2] - 3.0,
            "3 families × 3 levels × 3 tiers = 27 configurations",
            ha="center", va="top", fontsize=6, color=GREY)

    # ---- the rule, spanning the foot of the figure ------------------------
    ry = rows[2] - 13.5
    box(ax, COLS[1], ry, COLS[3] + COL_W - COLS[1], ROW_H - 1.5,
        "Pre-registered deployability rule, fixed before any model ran",
        "accuracy within 3 pp  ·  hallucination not higher  ·  "
        "net energy ≤ 60% of baseline  ·  AURAC at 20% rejection ≥ baseline",
        edge=OURS, fill="white", lw=1.0)
    arrow(ax, COLS[3] + COL_W / 2, rows[2] - 6.0,
          COLS[3] + COL_W / 2, ry + ROW_H - 1.5)

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig1_design.pdf")
    fig.savefig(FIGS / "fig1_design.png")
    plt.close(fig)
    print(f"wrote fig1_design.pdf / .png -> {FIGS}")


if __name__ == "__main__":
    main()
