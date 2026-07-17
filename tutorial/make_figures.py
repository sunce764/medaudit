"""Generate the tutorial's figures. Requires the optional `figures` extra:

    pip install -e ".[figures]" && python tutorial/make_figures.py

Every figure here is built from arithmetic or from the synthetic demo cohort —
no patient data is read, and nothing here needs the restricted dataset. Figures
land in tutorial/assets/ (tutorial/figures/ is swallowed by .gitignore's
`figures/` rule, which is deliberate — that rule exists to keep real figures out).
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
INK, ACCENT, WARN = "#222222", "#1f6feb", "#d1495b"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def ppv(sens, spec, prev):
    return sens * prev / (sens * prev + (1 - spec) * (1 - prev))


def fig_prevalence():
    """§5: the same model, read at three prevalences. Pure arithmetic."""
    prev = np.logspace(-3, -0.0, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    for spec, style in ((0.95, "-"), (0.99, "--")):
        ax.plot(prev * 100, [ppv(0.90, spec, p) for p in prev], style,
                color=ACCENT, lw=2 if spec == 0.95 else 1.2,
                label=f"sensitivity 0.90, specificity {spec}")
    offsets = {0.50: (-42, 8), 0.10: (8, 10), 0.01: (10, 14)}
    for p, txt in ((0.50, "50%"), (0.10, "10%"), (0.01, "1%")):
        v = ppv(0.90, 0.95, p)
        ax.plot(p * 100, v, "o", color=ACCENT, ms=6, zorder=4)
        ax.annotate(f"{txt} → PPV {v:.2f}", (p * 100, v), textcoords="offset points",
                    xytext=offsets[p], fontsize=8.5, color=INK, weight="bold")
    ax.axhline(0.035, color=WARN, lw=1.2, ls=":")
    ax.annotate("a real leaderboard lives down here:\nRARE25's winning entry scored "
                "PPV@90%recall = 0.035\nat ~1% prevalence — with AUROC 0.92 [15]",
                xy=(3.0, 0.035), xytext=(3.0, 0.30), fontsize=7.5, color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1))
    ax.set_xscale("log")
    ax.set_xlabel("prevalence (%, log scale)", fontsize=9, color=INK)
    ax.set_ylabel("positive predictive value", fontsize=9, color=INK)
    ax.set_title("The model never changes. Only the base rate does.",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(-0.03, 1.03)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prevalence.png"), dpi=200)
    print("  wrote assets/prevalence.png")


def fig_probe():
    """§2: the minimal pair. Same headline height, opposite verdict — the
    within-class bars are the whole argument, so draw them."""
    a, b = make_demo.case_a_class_collinear(), make_demo.case_b_encoded()
    labels = ["overall", "within\nbenign", "within\nmalignant"]

    def rows(rep):
        out = [rep["overall"]]
        for k in ("benign", "malignant"):
            out.append(rep["within_class"][k])
        return out

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for ax, rep, title in ((axes[0], a, "A · features encode only the CLASS"),
                           (axes[1], b, "B · features also encode MODE")):
        rs = rows(rep)
        x = np.arange(3)
        ax.axhline(0.60, color=WARN, lw=1, ls="--", zorder=0)
        ax.axhline(0.50, color="#cccccc", lw=1, zorder=0)
        # colour by verdict: a point below the decision line is not evidence
        for xi, r in zip(x, rs):
            p, (clo, chi) = r["auroc"], r["ci"]
            c = ACCENT if clo > 0.60 else "#9aa4b2"
            ax.errorbar([xi], [p], yerr=[[p - clo], [chi - p]], fmt="o", ms=7,
                        capsize=4, lw=1.6, color=c, ecolor=c, mfc=c, mew=0, zorder=3)
            ax.annotate(f"{p:.2f}", (xi, p), textcoords="offset points",
                        xytext=(10, -3), fontsize=8.5, color=c, weight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.55, 2.75); ax.set_ylim(0.35, 1.06)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel("probe AUROC (95% CI)", fontsize=9, color=INK)
    # labels on the right panel's empty upper-left, where nothing collides
    axes[1].annotate("0.60 decision line", (-0.45, 0.615), fontsize=7.5, color=WARN)
    axes[1].annotate("chance", (-0.45, 0.515), fontsize=7.5, color="#999999")
    fig.suptitle("Both headlines look alarming. Only B is a shortcut — "
                 "the within-class points say which.",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "probe.png"), dpi=200)
    print("  wrote assets/probe.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_prevalence()
    fig_probe()
    print("figures in", OUT)
