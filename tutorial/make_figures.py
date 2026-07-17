"""Generate the tutorial's figures. Requires the optional `figures` extra:

    pip install -e ".[figures]" && python tutorial/make_figures.py

Every figure here is built from arithmetic or from the synthetic demo cohort —
no patient data is read, and nothing here needs the restricted dataset. Figures
land in tutorial/assets/ (tutorial/figures/ is swallowed by .gitignore's
`figures/` rule, which is deliberate — that rule exists to keep real figures out).

Aesthetic target: a journal methods figure, not a slide. Helvetica, one accent
colour on a grey scale, thin reference lines, no in-axes narrative titles — the
prose around each figure is its caption. Keep the ink for the data.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medaudit.audits import probe
import make_demo  # noqa: E402  — reuses the exact cohorts §2 reports

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")

# One accent on a grey scale. Blue = data that clears the bar; grey = data that
# does not (and, for reference lines, structure the eye should not dwell on).
INK = "#1a1a1a"      # axes, text
DATA = "#1f4e79"     # deep, low-chroma blue — the only saturated ink
MUTED = "#9297a0"    # de-emphasised data (below the decision line)
LINE = "#b9bcc2"     # chance / neutral reference
RULE = "#6f7681"     # decision line — grey, dashed, present but quiet

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#4a4a4a",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def _spare(ax):
    """Two spines only. The journal-figure default."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def ppv(sens, spec, prev):
    return sens * prev / (sens * prev + (1 - spec) * (1 - prev))


def fig_prevalence():
    """§5: the same model, read at three prevalences. Pure arithmetic.

    The narrative (one model / three clinics / 85% false alarms at 1%, and the
    RARE25 0.035 line) lives in the caption and body text — the figure just
    carries the curves, the three read-points, and a quiet reference line.
    """
    prev = np.logspace(-3, 0.0, 400)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(prev * 100, [ppv(0.90, 0.95, p) for p in prev], "-",
            color=DATA, lw=1.8, label="specificity 0.95")
    ax.plot(prev * 100, [ppv(0.90, 0.99, p) for p in prev], "--",
            color=DATA, lw=1.1, dashes=(5, 3), label="specificity 0.99")

    # three read-points on the 0.95 curve; plain labels, no bold, no arrows
    # 50% sits high-right; label it below-and-right where the curve only rises
    # away — below-left would let the descending curve clip the text.
    reads = {0.01: (10, -3, "left"), 0.10: (9, -11, "left"), 0.50: (13, -15, "left")}
    for p, (dx, dy, ha) in reads.items():
        v = ppv(0.90, 0.95, p)
        ax.plot(p * 100, v, "o", color=DATA, ms=4.5, zorder=4)
        ax.annotate(f"{p:.0%}  PPV {v:.2f}", (p * 100, v),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, color=INK, ha=ha)

    # RARE25 reference line — quiet; the body text explains it
    ax.axhline(0.035, color=RULE, lw=0.9, ls=(0, (1, 2)))
    ax.annotate("RARE25 winning entry: PPV 0.035", xy=(100, 0.035),
                xytext=(0, 4), textcoords="offset points",
                fontsize=7, color=RULE, ha="right")

    ax.set_xscale("log")
    ax.set_xlabel("prevalence (%, log scale)")
    ax.set_ylabel("positive predictive value")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0.09, 110)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8,
              borderaxespad=0.2)
    _spare(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prevalence.png"))
    plt.close(fig)
    print("  wrote assets/prevalence.png")


def fig_probe():
    """§2: the minimal pair. Same headline height, opposite verdict — the
    within-class bars are the whole argument, so draw them. Panels (a)/(b);
    blue clears the decision line, grey does not."""
    a, b = make_demo.case_a_class_collinear(), make_demo.case_b_encoded()
    labels = ["overall", "within\nbenign", "within\nmalignant"]

    def rows(rep):
        return [rep["overall"], rep["within_class"]["benign"],
                rep["within_class"]["malignant"]]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    panels = ((axes[0], a, "a", "features encode only the class"),
              (axes[1], b, "b", "features also encode mode"))
    for ax, rep, letter, descr in panels:
        x = np.arange(3)
        ax.axhline(0.60, color=RULE, lw=0.9, ls=(0, (5, 3)), zorder=0)
        ax.axhline(0.50, color=LINE, lw=0.9, zorder=0)
        for xi, r in zip(x, rows(rep)):
            p, (clo, chi) = r["auroc"], r["ci"]
            c = DATA if clo > 0.60 else MUTED           # below the line = not evidence
            ax.errorbar([xi], [p], yerr=[[p - clo], [chi - p]], fmt="o", ms=5,
                        capsize=2.5, elinewidth=1.0, capthick=1.0,
                        color=c, ecolor=c, mfc=c, mew=0, zorder=3)
            ax.annotate(f"{p:.2f}", (xi, p), textcoords="offset points",
                        xytext=(9, -2.5), fontsize=7.5, color=c)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.55, 2.7)
        ax.set_ylim(0.35, 1.05)
        # journal-style panel letter, bold, top-left outside the plot body
        ax.text(-0.02, 1.06, f"({letter})", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="bottom", ha="left", color=INK)
        ax.set_title(descr, loc="left", x=0.08, pad=6, color=INK, fontsize=8)
        _spare(ax)
    axes[0].set_ylabel("probe AUROC (95% CI)")
    # Name the two shared reference lines once. Panel (b)'s lower band is empty
    # (every point there sits above 0.71), so long labels can go there without
    # crossing any error bar — in panel (a) they would run through the benign /
    # malignant whiskers, which both reach ~0.62.
    axes[1].annotate("0.60 decision line", (-0.45, 0.618), ha="left",
                     fontsize=6.8, color=RULE)
    axes[1].annotate("chance", (-0.45, 0.515), ha="left",
                     fontsize=6.8, color="#9a9a9a")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "probe.png"))
    plt.close(fig)
    print("  wrote assets/probe.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_prevalence()
    fig_probe()
    print("figures in", OUT)
