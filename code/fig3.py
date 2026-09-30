#!/usr/bin/env python3
"""
fig3.py  --  ET&S manuscript, Figure 3

Inference energy on the multiple-choice task (a-c) and the same energy
relative to each family's FP16 baseline (d-f), by difficulty tier.

Every value is read from 06-results/analysis/core_results.json. Nothing is
recomputed here, so the figure cannot disagree with the energy table.

Output: 07-figs/fig3_energy.{pdf,png} + fig3_caption.md
Canvas is exactly 165 x 100 mm, the ET&S body-text width: place at 100%.

Run:  py fig3.py
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(r"D:\claude_projects\ET&S")
ANALYSIS = ROOT / "06-results" / "analysis"
FIGS = ROOT / "07-figs"

MM = 1 / 25.4
FULL = 165 * MM

TIERS = ["school", "college", "expert"]
LEVELS = ["fp16", "q8_0", "q4_K_M"]
LEVEL_LABEL = {"fp16": "FP16", "q8_0": "q8_0", "q4_K_M": "q4_K_M"}
FAMILIES = ["qwen2.5:3b-instruct", "llama3.2:3b-instruct", "gemma2:2b-instruct"]
SHORT = {FAMILIES[0]: "Qwen2.5 3B", FAMILIES[1]: "Llama3.2 3B", FAMILIES[2]: "Gemma2 2B"}
COLOUR = {FAMILIES[0]: "#0072B2", FAMILIES[1]: "#D55E00", FAMILIES[2]: "#009E73"}
MARKER = {FAMILIES[0]: "o", FAMILIES[1]: "s", FAMILIES[2]: "^"}
DASH = {FAMILIES[0]: "-", FAMILIES[1]: (0, (5, 2)), FAMILIES[2]: (0, (1, 1.6))}
DODGE = {FAMILIES[0]: -0.13, FAMILIES[1]: 0.0, FAMILIES[2]: 0.13}
CEILING = 0.60

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": None,
    "savefig.pad_inches": 0.0, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

rows = json.loads((ANALYSIS / "core_results.json").read_text())["energy_rule"]
idx = {(r["family"], r["tier"], r["level"]): r for r in rows}


def panel_tag(ax, t):
    ax.text(-0.13, 1.04, t, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


x = list(range(len(LEVELS)))
fig, axes = plt.subplots(2, 3, figsize=(FULL, 100 * MM), sharex=True,
                         sharey="row", layout="constrained")
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.05, hspace=0.08)

for j, tier in enumerate(TIERS):
    ax_j, ax_r = axes[0][j], axes[1][j]
    ax_r.axhline(CEILING, color="#555555", linewidth=0.7,
                 linestyle=(0, (4, 2)), zorder=1)

    for fam in FAMILIES:
        dx = DODGE[fam]
        jj = [idx[(fam, tier, lv)]["j_per_item_mcq"] for lv in LEVELS]
        rr = [idx[(fam, tier, lv)]["ratio_mcq"] for lv in LEVELS]
        ax_j.plot([i + dx for i in x], jj, color=COLOUR[fam],
                  marker=MARKER[fam], linestyle=DASH[fam], label=SHORT[fam],
                  clip_on=False, zorder=3)
        # open marker = above the ceiling, filled = at or below it, so
        # pass/fail on criterion 3 is unambiguous even at the boundary
        ax_r.plot([i + dx for i in x], rr, color=COLOUR[fam],
                  marker=MARKER[fam], linestyle=DASH[fam], clip_on=False,
                  zorder=3, markerfacecolor="white", markeredgewidth=0.9)
        px = [i + dx for i, lv in enumerate(LEVELS)
              if idx[(fam, tier, lv)]["criterion3_energy"]]
        py = [idx[(fam, tier, lv)]["ratio_mcq"] for lv in LEVELS
              if idx[(fam, tier, lv)]["criterion3_energy"]]
        if px:
            ax_r.plot(px, py, color=COLOUR[fam], marker=MARKER[fam],
                      markerfacecolor=COLOUR[fam], linestyle="none",
                      zorder=4, clip_on=False)

    ax_j.set_title(tier.capitalize(), pad=4)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS])
    ax_r.set_xlim(-0.35, len(LEVELS) - 0.65)
    panel_tag(ax_j, f"({chr(97 + j)})")
    panel_tag(ax_r, f"({chr(100 + j)})")

axes[0][0].set_ylabel("Energy per item (J)")
axes[0][0].set_ylim(20, 61)
axes[0][0].set_yticks([20, 30, 40, 50, 60])
axes[1][0].set_ylabel("Energy relative to FP16")
axes[1][0].set_ylim(0.50, 1.07)
axes[1][0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

axes[1][0].text(-0.28, CEILING + 0.012, "pre-registered ceiling",
                fontsize=6.5, color="#555555", va="bottom", ha="left")

h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3, handlelength=2.2,
           columnspacing=1.8, borderaxespad=0.0)
fig.supxlabel("Compression level", fontsize=8)
fig.savefig(FIGS / "fig3_energy.pdf")
fig.savefig(FIGS / "fig3_energy.png")
print("ok")

caption = """**Figure 3.** Inference energy per multiple-choice item (a–c) and the same values relative to each family's FP16 baseline (d–f), by difficulty tier. Each point is one measured, idle-corrected run, so no interval is shown. The dashed line in (d–f) is the pre-registered ceiling of 0.60: filled markers meet it, open markers do not, and FP16 is 1.000 by construction. Five of the nine q4_K_M configurations clear the ceiling. Both axes are truncated. Colour, marker and line style each identify the model family.
"""
(FIGS / "fig3_caption.md").write_text(caption, encoding="utf-8")
print("caption written")

