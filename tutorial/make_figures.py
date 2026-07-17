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
from medaudit.metrics import auroc
import make_demo  # noqa: E402  — reuses the exact cohorts §2 reports

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")

# One accent on a grey scale. Blue = data that clears the bar; grey = data that
# does not (and, for reference lines, structure the eye should not dwell on).
INK = "#1a1a1a"      # axes, text
DATA = "#1f4e79"     # deep, low-chroma blue — the only saturated ink
EXT = "#c8792b"      # a second, colour-blind-safe hue for the "external" series
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _roc(scores, y):
    """FPR, TPR sweep for a ROC curve (plotting only; the AUROC number comes
    from medaudit.metrics so the figure and the toolkit agree)."""
    order = np.argsort(-scores, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.concatenate([[0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0], fp / max(fp[-1], 1)])
    return fpr, tpr


def _reliability(scores, y, n_bins=10):
    """Classic reliability points: mean predicted prob vs observed frequency per
    equal-width bin. (metrics.reliability_curve is top-class confidence, which is
    not what this diagram wants for a binary positive-probability curve.)"""
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if m.sum() >= 15:                       # skip underpowered bins
            xs.append(scores[m].mean())
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def fig_calibration():
    """§4: the same model read in two hospitals — discrimination transfers
    (ROC curves nearly superimpose) but calibration does not (reliability curves
    come apart). Illustrative synthetic cohort: no patient data, known ground
    truth. The model score is unchanged across sites; the external site's lower
    prevalence makes that same score over-confident."""
    rng = np.random.default_rng(20260718)
    n, a = 6000, 1.55
    zi = rng.normal(0, 1, n)                     # internal
    si = _sigmoid(a * zi)                        # calibrated model score
    yi = rng.binomial(1, si)
    ze = rng.normal(0, 1, n)                     # external
    se = _sigmoid(a * ze)                        # SAME model score mapping
    ye = rng.binomial(1, _sigmoid(a * ze - 1.1))  # true prob lower -> score over-confident

    ai, ae = auroc(si, yi), auroc(se, ye)

    fig, (axr, axc) = plt.subplots(1, 2, figsize=(6.6, 3.2))

    # (a) ROC — nearly superimposed
    axr.plot([0, 1], [0, 1], color=LINE, lw=0.9, zorder=0)
    for s, y, c, ls, lab, av in ((si, yi, DATA, "-", "internal", ai),
                                 (se, ye, EXT, "--", "external", ae)):
        fpr, tpr = _roc(s, y)
        axr.plot(fpr, tpr, ls, color=c, lw=1.7, label=f"{lab}  (AUROC {av:.2f})")
    axr.text(-0.02, 1.06, "(a)", transform=axr.transAxes, fontsize=9.5,
             fontweight="bold", va="bottom", color=INK)
    axr.set_title("discrimination transfers", loc="left", x=0.08, pad=6,
                  fontsize=8, color=INK)
    axr.set_xlabel("false-positive rate")
    axr.set_ylabel("true-positive rate")
    axr.set_xlim(-0.02, 1.02); axr.set_ylim(-0.02, 1.02)
    axr.legend(frameon=False, loc="lower right", handlelength=1.8)
    _spare(axr)

    # (b) reliability — comes apart
    axc.plot([0, 1], [0, 1], color=LINE, lw=0.9, zorder=0)
    for s, y, c, mk, lab in ((si, yi, DATA, "o", "internal"),
                             (se, ye, EXT, "s", "external")):
        xs, ys = _reliability(s, y)
        axc.plot(xs, ys, mk + "-", color=c, lw=1.4, ms=4, mew=0, label=lab)
    axc.text(-0.02, 1.06, "(b)", transform=axc.transAxes, fontsize=9.5,
             fontweight="bold", va="bottom", color=INK)
    axc.set_title("calibration does not", loc="left", x=0.08, pad=6,
                  fontsize=8, color=INK)
    axc.annotate("over-confident:\nscore > observed", (0.55, 0.20),
                 fontsize=6.8, color=EXT, ha="left")
    axc.set_xlabel("mean predicted probability")
    axc.set_ylabel("observed frequency")
    axc.set_xlim(-0.02, 1.02); axc.set_ylim(-0.02, 1.02)
    axc.legend(frameon=False, loc="upper left", handlelength=1.6)
    _spare(axc)

    fig.tight_layout()
    # Label the diagonal, parallel to it, floated into the empty upper triangle so
    # it never sits on the internal curve (which hugs the diagonal). A data-slope-1
    # line's SCREEN angle depends on the final axes aspect, so read it off the
    # laid-out transform rather than hard-coding degrees (34 vs 41 is the gap that
    # made an earlier version's text drift onto the curve).
    fig.canvas.draw()
    p0 = axc.transData.transform((0.0, 0.0))
    p1 = axc.transData.transform((1.0, 1.0))
    diag_deg = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    axc.annotate("perfect calibration", (0.27, 0.40), rotation=diag_deg,
                 rotation_mode="anchor", fontsize=6.8, color="#9a9a9a",
                 ha="left", va="bottom")
    fig.savefig(os.path.join(OUT, "calibration.png"))
    plt.close(fig)
    print(f"  wrote assets/calibration.png  (AUROC internal {ai:.3f}, external {ae:.3f})")


def fig_roadmap():
    """§1: the four audits at a glance. Doubles as (i) a structural map for a
    new-Masters reader, (ii) the honest 'what each audit does NOT establish'
    column that is this tutorial's spine, and (iii) the disclosure that only two
    of the four run from one command — the rest you compose from primitives.
    A booktabs-style table (no vertical rules); wrapped by hand to fit columns."""
    # columns: (left x in axes fraction, header)
    cx = [0.010, 0.200, 0.520, 0.855]
    headers = ["audit", "what it detects", "what it does NOT establish",
               "one command?"]
    rows = [
        ("Shortcut", "§2", "an attribute is encoded\nin the features",
         "that the model uses it", "yes"),
        ("Leakage", "§3", "near-duplicate & group\noverlap across the split",
         "a clean split when you\ngave no group ids", "yes"),
        ("Calibration", "§4", "whether scores are\nreal probabilities",
         "discrimination; verdicts\nunder ~30 clusters", "compose"),
        ("Prevalence", "§5", "PPV at the deployment\nbase rate",
         "what AUROC shows —\nit is prevalence-blind", "compose"),
    ]
    ry = [0.685, 0.505, 0.325, 0.145]           # row centres
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def rule(y, lw, color="#4a4a4a"):
        ax.plot([0.005, 0.995], [y, y], lw=lw, color=color, zorder=1)

    rule(0.965, 1.0)                            # top
    for i, h in enumerate(headers):
        ax.text(cx[i], 0.885, h, fontsize=7.8, fontweight="bold", color=INK,
                va="center", ha="left")
    rule(0.815, 0.8)                            # under header
    for k, (name, sec, det, notdet, one) in enumerate(rows):
        y = ry[k]
        if k:                                   # faint inter-row separators
            rule((ry[k - 1] + y) / 2, 0.4, "#d8dade")
        ax.text(cx[0], y, name, fontsize=7.8, fontweight="bold", color=INK,
                va="center", ha="left")
        ax.text(cx[0] + 0.115, y, sec, fontsize=7.0, color="#8a8f98",
                va="center", ha="left")
        ax.text(cx[1], y, det, fontsize=7.3, color=INK, va="center", ha="left")
        ax.text(cx[2], y, notdet, fontsize=7.3, color=MUTED, va="center", ha="left")
        c = DATA if one == "yes" else "#8a8f98"
        ax.text(cx[3], y, one, fontsize=7.6, color=c, va="center", ha="left",
                fontweight="bold" if one == "yes" else "normal")
    rule(0.055, 1.0)                            # bottom
    ax.text(0.010, -0.02,
            "yes = runs from one command (medaudit audit)      compose = "
            "assemble from the toolkit's tested metric primitives (§4, §5)",
            fontsize=6.6, color="#8a8f98", va="top", ha="left")
    fig.savefig(os.path.join(OUT, "roadmap.png"))
    plt.close(fig)
    print("  wrote assets/roadmap.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_prevalence()
    fig_probe()
    fig_calibration()
    fig_roadmap()
    print("figures in", OUT)
