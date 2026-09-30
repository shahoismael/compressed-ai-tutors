#!/usr/bin/env python3
"""
fig2.py  --  ET&S manuscript, Figure 2

Confident error on the abstention probe (a-c) and its change from the FP16
baseline (d-f), by difficulty tier.

Every value is read from 06-results/analysis/core_results.json. Nothing is
recomputed here, so the figure cannot disagree with the hallucination table.

Output: 07-figs/fig2_confident_error.{pdf,png} + fig2_caption.md
Canvas is exactly 165 x 100 mm, the ET&S body-text width: place at 100%.

Run:  py fig2.py
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

# Okabe-Ito, identical assignment to Figure 1 so the key carries across
# figures. Colour is never the only channel.
COLOUR = {FAMILIES[0]: "#0072B2", FAMILIES[1]: "#D55E00", FAMILIES[2]: "#009E73"}
MARKER = {FAMILIES[0]: "o", FAMILIES[1]: "s", FAMILIES[2]: "^"}
DASH = {FAMILIES[0]: "-", FAMILIES[1]: (0, (5, 2)), FAMILIES[2]: (0, (1, 1.6))}

# horizontal dodge: without it the three families' intervals collapse onto one
# another in the crowded upper band and along the zero line
DODGE = {FAMILIES[0]: -0.13, FAMILIES[1]: 0.0, FAMILIES[2]: 0.13}

ALPHA = 0.05

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    # NOT "tight": a trimmed bbox shrinks the canvas below the journal width,
    # which forces a rescale in Word and breaks the point sizes below
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
        raise SystemExit(f"missing {p} - run analyse_core.py first")
    rows = json.loads(p.read_text(encoding="utf-8"))["hallucination"]
    return {(r["family"], r["tier"], r["level"]): r for r in rows}


def panel_tag(ax, text):
    ax.text(-0.13, 1.04, text, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def main():
    idx = load()
    x = list(range(len(LEVELS)))

    # constrained layout + an untrimmed bbox: the saved canvas is exactly
    # 165 x 100 mm, so the figure is placed at 100% and never rescaled
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 100 * MM), sharex=True,
                             sharey="row", layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01,
                                wspace=0.05, hspace=0.08)

    for j, tier in enumerate(TIERS):
        ax_r, ax_d = axes[0][j], axes[1][j]
        ax_d.axhline(0, color="#BBBBBB", linewidth=0.5, zorder=1)

        for fam in FAMILIES:
            dx = DODGE[fam]
            rate, rlo, rhi = [], [], []
            dxs, dvals, dlo, dhi, dsig = [], [], [], [], []

            for i, lv in enumerate(LEVELS):
                r = idx[(fam, tier, lv)]
                v = r["hallucination_rate"] * 100
                ci = r["hallucination_ci"]
                rate.append(v)
                rlo.append(v - ci[0] * 100)
                rhi.append(ci[1] * 100 - v)

                # FP16 is the reference: its difference from itself is zero by
                # construction. Plotting it keeps the line continuous and both
                # rows on the same x-scale.
                d = r["delta_pp"]
                dci = r["delta_ci_pp"]
                dxs.append(i + dx)
                dvals.append(d)
                dlo.append(d - dci[0])
                dhi.append(dci[1] - d)
                p = r.get("p_holm")
                dsig.append(p is not None and p < ALPHA)

            ax_r.errorbar([i + dx for i in x], rate, yerr=[rlo, rhi],
                          color=COLOUR[fam], marker=MARKER[fam],
                          linestyle=DASH[fam], capsize=2, elinewidth=0.6,
                          capthick=0.6, label=SHORT[fam], clip_on=False,
                          zorder=3)

            ax_d.errorbar(dxs, dvals, yerr=[dlo, dhi], color=COLOUR[fam],
                          marker=MARKER[fam], linestyle=DASH[fam], capsize=2,
                          elinewidth=0.6, capthick=0.6, clip_on=False,
                          zorder=3, markerfacecolor="white",
                          markeredgewidth=0.9)
            # filled marker = survives Holm correction within its tier
            sx = [a for a, s in zip(dxs, dsig) if s]
            sy = [a for a, s in zip(dvals, dsig) if s]
            if sx:
                ax_d.plot(sx, sy, color=COLOUR[fam], marker=MARKER[fam],
                          markerfacecolor=COLOUR[fam], linestyle="none",
                          zorder=4, clip_on=False)

        ax_r.set_title(tier.capitalize(), pad=4)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS])
        ax_d.set_xlim(-0.35, len(LEVELS) - 0.65)
        panel_tag(ax_r, f"({chr(97 + j)})")
        panel_tag(ax_d, f"({chr(100 + j)})")

    axes[0][0].set_ylabel("Confident error (%)")
    # lowest interval bound is 46.5 %; the axis stops just below it
    axes[0][0].set_ylim(44, 101)
    axes[0][0].set_yticks([50, 60, 70, 80, 90, 100])

    axes[1][0].set_ylabel("Change from FP16 (pp)")
    axes[1][0].set_ylim(-22, 13)
    axes[1][0].set_yticks([-20, -10, 0, 10])

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               handlelength=2.2, columnspacing=1.8, borderaxespad=0.0)

    fig.supxlabel("Compression level", fontsize=8)
    fig.savefig(FIGS / "fig2_confident_error.pdf")
    fig.savefig(FIGS / "fig2_confident_error.png")
    plt.close(fig)

    caption = """**Figure 2.** Confident error on the abstention probe (a–c) and its change from the FP16 baseline (d–f), by difficulty tier. The correct option was removed and an explicit refusal offered, so choosing any remaining option is a confident error (n = 200 per configuration per tier). Error bars are 95% percentile bootstrap intervals, over items in (a–c) and over item pairs in (d–f), where the grey line marks no change. Filled markers in (d–f) are significant at α = .05 after Holm correction within tier; open markers, including the FP16 reference, are not. Points are dodged and the error axis truncated. Colour, marker and line style each identify the model family.
"""
    (FIGS / "fig2_caption.md").write_text(caption, encoding="utf-8")

    print("wrote fig2_confident_error.pdf  (165 x 100 mm, place at 100%)")
    print("wrote fig2_confident_error.png  (600 dpi)")
    print("wrote fig2_caption.md")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    main()
