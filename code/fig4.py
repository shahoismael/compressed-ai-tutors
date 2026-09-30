#!/usr/bin/env python3
"""
fig4.py  --  ET&S manuscript, Figure 4

Accuracy-rejection curves from semantic entropy, by model family (rows) and
difficulty tier (columns).

Every value is read from 06-results/analysis/entropy_results.json. Nothing is
recomputed here, so the figure cannot disagree with the entropy table.

Output: 07-figs/fig4_entropy_rejection.{pdf,png} + fig4_caption.md
Canvas is exactly 165 x 140 mm, the ET&S body-text width: place at 100%.

Run:  py fig4.py
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
FAMILIES = ["qwen2.5:3b-instruct", "llama3.2:3b-instruct", "gemma2:2b-instruct"]
SHORT = {FAMILIES[0]: "Qwen2.5 3B", FAMILIES[1]: "Llama3.2 3B", FAMILIES[2]: "Gemma2 2B"}
LEVELS = ["fp16", "q8_0", "q4_K_M"]
LEVEL_LABEL = {"fp16": "FP16", "q8_0": "q8_0", "q4_K_M": "q4_K_M"}

# In this figure the panel carries the model family, so colour and line style
# are free to carry the compression level. A deliberately different palette
# from Figures 1-3 signals the switch.
LCOLOUR = {"fp16": "#000000", "q8_0": "#E69F00", "q4_K_M": "#CC79A7"}
LDASH = {"fp16": "-", "q8_0": (0, (5, 2)), "q4_K_M": (0, (1, 1.6))}

XMAX = 0.80          # beyond this fewer than 20 items remain
REJECT = 0.20        # pre-registered operating point

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": None,
    "savefig.pad_inches": 0.0, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

data = json.loads((ANALYSIS / "entropy_results.json").read_text())
cells = {}
for c in data["cells"]:
    model = c["model"]
    fam, lvl = model.rsplit("-", 1)
    cells[(fam, c["tier"], lvl)] = c


def panel_tag(ax, t):
    ax.text(-0.07, 1.03, t, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


fig, axes = plt.subplots(3, 3, figsize=(FULL, 140 * MM), sharex=True,
                         sharey=True, layout="constrained")
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.05, hspace=0.06)

for i, fam in enumerate(FAMILIES):
    for j, tier in enumerate(TIERS):
        ax = axes[i][j]
        ax.axvline(REJECT, color="#BBBBBB", linewidth=0.5, zorder=1)
        for lvl in LEVELS:
            c = cells[(fam, tier, lvl)]
            pts = [(x, y) for x, y in c["curve_strict"] if x <= XMAX]
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=LCOLOUR[lvl], linestyle=LDASH[lvl],
                    label=LEVEL_LABEL[lvl], zorder=3)
        if i == 0:
            ax.set_title(tier.capitalize(), pad=4)
        if j == 0:
            # row label inside the panel: the left margin carries the
            # quantity, and the upper-left of these panels is empty
            ax.text(0.035, 0.94, SHORT[fam], transform=ax.transAxes,
                    fontsize=8, va="top", ha="left")
        panel_tag(ax, f"({chr(97 + i * 3 + j)})")

axes[0][0].set_xlim(0, XMAX)
axes[0][0].set_xticks([0, 0.2, 0.4, 0.6, 0.8])
axes[0][0].set_ylim(0.04, 0.63)
axes[0][0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h[:3], l[:3], loc="outside upper center", ncol=3, handlelength=2.4,
           columnspacing=1.8, borderaxespad=0.0)
fig.supxlabel("Fraction of items rejected (highest semantic entropy first)",
              fontsize=8)
fig.supylabel("Accuracy of retained answers", fontsize=8)
fig.savefig(FIGS / "fig4_entropy_rejection.pdf")
fig.savefig(FIGS / "fig4_entropy_rejection.png")
print("ok")

caption = """**Figure 4.** Accuracy of retained answers as the highest-entropy answers are rejected, by model family (rows) and difficulty tier (columns). A rising curve means semantic entropy ranks wrong answers above right ones. The vertical line marks the pre-registered 0.20 rejection fraction. Curves stop at 80% rejection, beyond which fewer than twenty items remain; the summary statistics use the full curve. Accuracy is the strict grading (n = 100 per configuration per tier); the lenient grading is in Table A3. Colour and line style both identify the compression level here, whereas in Figures 1–3 colour identifies the model family.
"""
(FIGS / "fig4_caption.md").write_text(caption, encoding="utf-8")
print("caption written")

