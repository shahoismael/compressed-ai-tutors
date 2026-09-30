#!/usr/bin/env python3
"""
fig1.py  --  ET&S manuscript, Figure 1

Accuracy (a-c) and answer fidelity (d-f) across compression levels, by tier.

Every value is read from 06-results/analysis/core_results.json. Nothing is
recomputed here, so the figure cannot disagree with Table 2.

Output: 07-figs/fig1_accuracy_fidelity.{pdf,png} + fig1_caption.md

Run:  py fig1.py
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
FULL = 165 * MM              # ET&S body-text width

TIERS = ["school", "college", "expert"]
LEVELS = ["fp16", "q8_0", "q4_K_M"]
LEVEL_LABEL = {"fp16": "FP16", "q8_0": "q8_0", "q4_K_M": "q4_K_M"}
FAMILIES = ["qwen2.5:3b-instruct", "llama3.2:3b-instruct", "gemma2:2b-instruct"]
SHORT = {FAMILIES[0]: "Qwen2.5 3B",
         FAMILIES[1]: "Llama3.2 3B",
         FAMILIES[2]: "Gemma2 2B"}

# Okabe-Ito. Colour is never the only channel: marker and line style repeat it.
COLOUR = {FAMILIES[0]: "#0072B2", FAMILIES[1]: "#D55E00", FAMILIES[2]: "#009E73"}
MARKER = {FAMILIES[0]: "o", FAMILIES[1]: "s", FAMILIES[2]: "^"}
DASH = {FAMILIES[0]: "-", FAMILIES[1]: (0, (5, 2)), FAMILIES[2]: (0, (1, 1.6))}


mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    # NOT "tight": a trimmed bbox shrinks the canvas below the journal
    # width, which forces a rescale in Word and breaks the point sizes below
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load():
    p = ANALYSIS / "core_results.json"
    if not p.exists():
        raise SystemExit(f"missing {p} — run analyse_core.py first")
    rows = json.loads(p.read_text(encoding="utf-8"))["accuracy_fidelity"]
    return {(r["family"], r["tier"], r["level"]): r for r in rows}


def panel_tag(ax, text):
    ax.text(-0.13, 1.04, text, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def main():
    idx = load()
    x = list(range(len(LEVELS)))

    # constrained layout + an untrimmed bbox: the saved canvas is exactly
    # 165 x 100 mm, so the figure is placed at 100% and never rescaled
    fig, axes = plt.subplots(
        2, 3, figsize=(FULL, 100 * MM), sharex=True, sharey="row",
        layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01,
                                wspace=0.05, hspace=0.08)

    for j, tier in enumerate(TIERS):
        ax_a, ax_l = axes[0][j], axes[1][j]

        for fam in FAMILIES:
            acc, lo, hi, loy = [], [], [], []
            for lv in LEVELS:
                r = idx.get((fam, tier, lv))
                if r is None:
                    acc.append(None); lo.append(0); hi.append(0)
                    loy.append(None)
                    continue
                a = r["accuracy"] * 100
                ci = r["accuracy_ci"] or [r["accuracy"], r["accuracy"]]
                acc.append(a)
                lo.append(a - ci[0] * 100)
                hi.append(ci[1] * 100 - a)
                # a model is loyal to itself by definition; plotting 1.0 at the
                # baseline keeps the panel readable instead of leaving a gap
                loy.append(1.0 if lv == "fp16" else r["loyalty"])

            ax_a.errorbar(x, acc, yerr=[lo, hi], color=COLOUR[fam],
                          marker=MARKER[fam], linestyle=DASH[fam],
                          capsize=2, elinewidth=0.6, capthick=0.6,
                          label=SHORT[fam], clip_on=False, zorder=3)

            ax_l.plot(x, loy, color=COLOUR[fam], marker=MARKER[fam],
                      linestyle=DASH[fam], clip_on=False, zorder=3,
                      markerfacecolor=COLOUR[fam])
            # open marker at the definitional point, so it reads as a reference
            ax_l.plot([0], [1.0], color=COLOUR[fam], marker=MARKER[fam],
                      markerfacecolor="white", markeredgewidth=0.9,
                      linestyle="none", zorder=4, clip_on=False)

        ax_a.set_title(tier.capitalize(), pad=4)
        ax_l.set_xticks(x)
        ax_l.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS])
        ax_l.set_xlim(-0.25, len(LEVELS) - 0.75)
        panel_tag(ax_a, f"({chr(97 + j)})")
        panel_tag(ax_l, f"({chr(100 + j)})")


    axes[0][0].set_ylabel("Accuracy (%)")
    axes[1][0].set_ylabel("Loyalty to FP16")
    # lowest observed loyalty is 0.797; the axis stops just below it rather
    # than leaving a band of empty panel under the data
    axes[1][0].set_ylim(0.785, 1.012)
    axes[1][0].set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])

    # one legend for the whole figure, placed where no data sits
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][0].legend(handles, labels, loc="lower left", handlelength=2.2,
                      borderaxespad=0.3, labelspacing=0.35)

    fig.supxlabel("Compression level", fontsize=8)
    fig.savefig(FIGS / "fig1_accuracy_fidelity.pdf")
    fig.savefig(FIGS / "fig1_accuracy_fidelity.png")
    plt.close(fig)

    caption = """**Figure 1.** Accuracy (a–c) and fidelity to the FP16 baseline (d–f) by compression level and difficulty tier. Points are per-configuration values on the fixed item set (school n = 1,000; college n = 922; expert n = 1,000); error bars are 95% percentile bootstrap intervals over items. Loyalty is the proportion of items on which the compressed model returns the same label as its FP16 baseline, correct or not; the open marker at FP16 is definitional. The loyalty axis is truncated. Colour, marker and line style each identify the model family, so the figure carries in greyscale.
"""
    (FIGS / "fig1_caption.md").write_text(caption, encoding="utf-8")

    print("wrote fig1_accuracy_fidelity.pdf")
    print("wrote fig1_accuracy_fidelity.png  (600 dpi)")
    print("wrote fig1_caption.md")
    print(f"\n-> {FIGS}")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    main()
