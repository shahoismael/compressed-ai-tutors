#!/usr/bin/env python3
"""
make_figures.py  --  ET&S study, step 6 (figures)

Builds the manuscript's four figures from the analysis outputs. Nothing is
recomputed here: every value is read from core_results.json or
entropy_results.json, so a figure can never disagree with a table.

  fig1  accuracy and answer fidelity against compression, by tier
  fig2  confident error and abstention on the probe set, by tier
  fig3  rejection-accuracy curves (the AURAC surface), by tier
  fig4  the trade-off surface: net energy against accuracy loss, by tier

Output: 07-figs/fig<N>_<slug>.{png,pdf}
PNG at 600 dpi because ET&S accepts JPEG/PNG only; PDF is kept as the vector
archive copy.

Run:  py make_figures.py
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
FULL = 165 * MM          # ET&S body text width
TIERS = ["school", "college", "expert"]
LEVELS = ["fp16", "q8_0", "q4_K_M"]
LEVEL_LABEL = {"fp16": "FP16", "q8_0": "q8_0", "q4_K_M": "q4_K_M"}
FAMILIES = ["qwen2.5:3b-instruct", "llama3.2:3b-instruct", "gemma2:2b-instruct"]
SHORT = {"qwen2.5:3b-instruct": "Qwen2.5 3B",
         "llama3.2:3b-instruct": "Llama3.2 3B",
         "gemma2:2b-instruct": "Gemma2 2B"}

# Okabe-Ito, colourblind-safe; marker and line style carry the same information
COLOUR = {FAMILIES[0]: "#0072B2", FAMILIES[1]: "#D55E00", FAMILIES[2]: "#009E73"}
MARKER = {FAMILIES[0]: "o", FAMILIES[1]: "s", FAMILIES[2]: "^"}
DASH = {FAMILIES[0]: "-", FAMILIES[1]: "--", FAMILIES[2]: ":"}


mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
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
    "lines.markersize": 3.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load():
    core = json.loads((ANALYSIS / "core_results.json").read_text(encoding="utf-8"))
    ep = ANALYSIS / "entropy_results.json"
    ent = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else {"cells": []}
    return core, ent


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png")
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def panel_label(ax, text):
    ax.text(-0.14, 1.06, text, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left")


def fig1_accuracy_fidelity(core):
    """Accuracy (top) and loyalty (bottom) across compression, one column per tier."""
    rows = core["accuracy_fidelity"]
    idx = {(r["family"], r["tier"], r["level"]): r for r in rows}

    fig, axes = plt.subplots(2, 3, figsize=(FULL, 95 * MM), sharex=True,
                             sharey="row")
    x = range(len(LEVELS))

    for j, tier in enumerate(TIERS):
        ax_a, ax_l = axes[0][j], axes[1][j]
        for fam in FAMILIES:
            acc, lo, hi, loy = [], [], [], []
            for lv in LEVELS:
                r = idx.get((fam, tier, lv))
                if not r:
                    continue
                acc.append(r["accuracy"] * 100)
                ci = r["accuracy_ci"] or [None, None]
                lo.append((r["accuracy"] - (ci[0] or r["accuracy"])) * 100)
                hi.append(((ci[1] or r["accuracy"]) - r["accuracy"]) * 100)
                loy.append(None if r["loyalty"] is None else r["loyalty"])
            ax_a.errorbar(x, acc, yerr=[lo, hi], color=COLOUR[fam],
                          marker=MARKER[fam], linestyle=DASH[fam],
                          capsize=2, elinewidth=0.6, label=SHORT[fam])
            pts = [(k, v) for k, v in zip(x, loy) if v is not None]
            ax_l.plot([p[0] for p in pts], [p[1] for p in pts],
                      color=COLOUR[fam], marker=MARKER[fam],
                      linestyle=DASH[fam])

        ax_a.set_title(tier.capitalize())
        ax_l.set_xticks(list(x))
        ax_l.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS])
        ax_l.axhline(1.0, color="#999999", linewidth=0.5, zorder=0)
        if j == 0:
            ax_a.set_ylabel("Accuracy (%)")
            ax_l.set_ylabel("Loyalty to FP16")
        panel_label(ax_a, f"({chr(97+j)})")
        panel_label(ax_l, f"({chr(100+j)})")

    axes[0][0].legend(loc="lower left", handlelength=1.8)
    fig.supxlabel("Compression level", fontsize=8, y=0.01)
    fig.tight_layout()
    save(fig, "fig1_accuracy_fidelity")


def fig2_hallucination(core):
    """Confident error on the probe set, with abstention shown as open markers."""
    rows = core["hallucination"]
    idx = {(r["family"], r["tier"], r["level"]): r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 58 * MM), sharey=True)
    x = range(len(LEVELS))

    for j, tier in enumerate(TIERS):
        ax = axes[j]
        for fam in FAMILIES:
            rate, lo, hi, abst = [], [], [], []
            for lv in LEVELS:
                r = idx.get((fam, tier, lv))
                if not r:
                    continue
                v = r["hallucination_rate"] * 100
                ci = r["hallucination_ci"] or [None, None]
                rate.append(v)
                lo.append(v - (ci[0] or r["hallucination_rate"]) * 100)
                hi.append((ci[1] or r["hallucination_rate"]) * 100 - v)
                abst.append(r["abstention_rate"] * 100)
            ax.errorbar(x, rate, yerr=[lo, hi], color=COLOUR[fam],
                        marker=MARKER[fam], linestyle=DASH[fam],
                        capsize=2, elinewidth=0.6, label=SHORT[fam])
            ax.plot(x, abst, color=COLOUR[fam], marker=MARKER[fam],
                    linestyle="none", markerfacecolor="white",
                    markeredgewidth=0.8)

        ax.set_title(tier.capitalize())
        ax.set_xticks(list(x))
        ax.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS])
        ax.set_ylim(0, 100)
        if j == 0:
            ax.set_ylabel("Probe items (%)")
        panel_label(ax, f"({chr(97+j)})")

    axes[0].legend(loc="center left", handlelength=1.8)
    fig.supxlabel("Compression level", fontsize=8, y=0.02)
    fig.tight_layout()
    save(fig, "fig2_hallucination")


def fig3_rejection(ent):
    """Accuracy on retained answers as the highest-entropy answers are refused."""
    cells = {(c["model"], c["tier"]): c for c in ent.get("cells", [])}
    if not cells:
        print("  fig3 skipped: no entropy results")
        return

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 58 * MM), sharey=True)
    for j, tier in enumerate(TIERS):
        ax = axes[j]
        for fam in FAMILIES:
            for lv, alpha in zip(LEVELS, (1.0, 0.65, 0.4)):
                c = cells.get((f"{fam}-{lv}", tier))
                if not c or not c.get("curve_strict"):
                    continue
                xs = [p[0] * 100 for p in c["curve_strict"]]
                ys = [p[1] * 100 for p in c["curve_strict"]]
                ax.plot(xs, ys, color=COLOUR[fam], alpha=alpha,
                        linestyle=DASH[fam], linewidth=0.9)
        ax.axvline(20, color="#999999", linewidth=0.5, zorder=0)
        ax.set_title(tier.capitalize())
        ax.set_xlim(0, 90)
        if j == 0:
            ax.set_ylabel("Accuracy on retained (%)")
        panel_label(ax, f"({chr(97+j)})")

    handles = [plt.Line2D([], [], color=COLOUR[f], linestyle=DASH[f],
                          label=SHORT[f]) for f in FAMILIES]
    handles += [plt.Line2D([], [], color="#444444", alpha=a, label=LEVEL_LABEL[l])
                for l, a in zip(LEVELS, (1.0, 0.65, 0.4))]
    axes[0].legend(handles=handles, loc="upper left", handlelength=1.8, ncol=1)
    fig.supxlabel("Answers refused (%)", fontsize=8, y=0.02)
    fig.tight_layout()
    save(fig, "fig3_rejection_accuracy")


def fig4_tradeoff(core):
    """Net energy against accuracy loss. The deployability box is drawn, so a
    reader can see that nothing lands inside it."""
    acc = {(r["family"], r["tier"], r["level"]): r
           for r in core["accuracy_fidelity"]}
    ene = {(r["family"], r["tier"], r["level"]): r for r in core["energy_rule"]}
    tol = core["rule"]["accuracy_tolerance_pp"]
    ceil = core["rule"]["energy_ceiling"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 60 * MM), sharey=True)
    for j, tier in enumerate(TIERS):
        ax = axes[j]
        # the region the pre-registered rule admits on these two axes
        ax.axvspan(-tol, 0.0, ymin=0, ymax=1, color="#EEEEEE", zorder=0)
        ax.axhline(ceil, color="#999999", linewidth=0.5, zorder=1)

        for fam in FAMILIES:
            for lv in LEVELS:
                a, e = acc.get((fam, tier, lv)), ene.get((fam, tier, lv))
                if not a or not e or e["ratio_mcq"] is None:
                    continue
                # compression level is carried by marker size and fill, not by
                # a text label: at this point density the labels collided
                style = {"fp16": dict(markerfacecolor="white", markersize=4.5),
                         "q8_0": dict(markerfacecolor=COLOUR[fam],
                                      markersize=3.2, alpha=0.55),
                         "q4_K_M": dict(markerfacecolor=COLOUR[fam],
                                        markersize=5.5)}[lv]
                ax.plot(a["delta_pp"], e["ratio_mcq"] * 100,
                        marker=MARKER[fam], color=COLOUR[fam],
                        markeredgewidth=0.8, linestyle="none", **style)

        ax.set_title(tier.capitalize())
        ax.set_xlim(-4, 1.5)
        ax.set_ylim(40, 110)
        if j == 0:
            ax.set_ylabel("Net energy, % of FP16")
        panel_label(ax, f"({chr(97+j)})")

    handles = [plt.Line2D([], [], color=COLOUR[f], marker=MARKER[f],
                          linestyle="none", label=SHORT[f]) for f in FAMILIES]
    lvl = [plt.Line2D([], [], color="#444444", marker="o", linestyle="none",
                      markerfacecolor="white", markersize=4.5, label="FP16"),
           plt.Line2D([], [], color="#444444", marker="o", linestyle="none",
                      markerfacecolor="#444444", markersize=3.2, alpha=0.55,
                      label="q8_0"),
           plt.Line2D([], [], color="#444444", marker="o", linestyle="none",
                      markerfacecolor="#444444", markersize=5.5,
                      label="q4_K_M")]
    axes[0].legend(handles=handles, loc="lower left", handlelength=1.2)
    axes[1].legend(handles=lvl, loc="lower left", handlelength=1.2)
    fig.supxlabel("Accuracy change from FP16 (percentage points)",
                  fontsize=8, y=0.01)
    fig.tight_layout()
    save(fig, "fig4_tradeoff")


CAPTIONS = """# Figure captions

**Figure 1.** Accuracy (a–c) and answer fidelity (d–f) across compression
levels, by difficulty tier. Points are per-configuration values on the fixed
item set; error bars are 95% percentile bootstrap intervals over items
(10,000 resamples). Loyalty is the proportion of items on which the compressed
model returns the same label as its FP16 baseline, correct or not; the grey
line marks perfect agreement.

**Figure 2.** Confident error on the abstention-permitted probe set, by tier.
Filled markers are the hallucination rate, the proportion of probe items on
which the model selected a distractor after the correct option had been
removed; error bars are 95% percentile bootstrap intervals over items. Open
markers are the abstention rate on the same items. Abstention is the correct
response on every probe item.

**Figure 3.** Accuracy on retained answers as the highest-entropy answers are
refused, by tier. Each line is one configuration; colour and line style
identify the model family and opacity the compression level. The grey vertical
line marks the 20% rejection threshold at which the pre-registered rule is
evaluated. Grading is strict: the modal meaning cluster and the reference
answer must entail each other.

**Figure 4.** The trade-off surface: net energy per assessed item, as a
percentage of the FP16 baseline, against the change in accuracy from that
baseline, by tier. Open markers are the FP16 baselines themselves. The shaded
band and the horizontal line mark the two pre-registered thresholds on these
axes — accuracy within 3 percentage points, energy at or below 60% of
baseline. A configuration must fall inside the shaded band and below the line
to satisfy them, and must also satisfy the hallucination and AURAC criteria,
which this figure does not show.
"""


def main():
    core, ent = load()
    print("building figures ...")
    fig1_accuracy_fidelity(core)
    fig2_hallucination(core)
    fig3_rejection(ent)
    fig4_tradeoff(core)
    FIGS.mkdir(parents=True, exist_ok=True)
    (FIGS / "figure-captions.md").write_text(CAPTIONS, encoding="utf-8")
    print(f"  wrote figure-captions.md")
    print(f"\nall figures -> {FIGS}")


if __name__ == "__main__":
    main()
