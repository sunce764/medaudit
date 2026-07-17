"""Shortcut probe — the flagship audit.

Question it answers: *does the model's feature representation encode a
non-diagnostic acquisition/metadata attribute* (imaging mode, scanner, site,
stain, …)? If it does, the model can exploit that attribute as a shortcut, and
performance will not transfer when the attribute's correlation with the label
changes (a new site, a rebalanced cohort).

Method: a **linear probe** — a ridge classifier trained on frozen features to
predict the attribute — evaluated **group-aware, out-of-fold** (no
patient/case spans train and eval). High held-out AUROC means the attribute is
linearly readable from the features.

Two readings, and the second is the one that matters:

  * **overall** probe — attribute vs. features across the whole set. High AUROC
    alone is ambiguous: if the attribute is collinear with the class (e.g. a
    mode only used on malignant cases), the probe can score high just by
    reading the class.
  * **within-class** probe — the same probe run *inside each fixed class*.
    Holding the diagnosis constant removes the class-collinearity explanation:
    if the attribute is still decodable here, the features genuinely encode the
    acquisition attribute beyond what the label explains. This is the strong
    evidence of a shortcut.

No sklearn. The probe is closed-form ridge regression on the class indicator
(monotonic in the logistic decision value for AUROC purposes), deterministic,
and standardisation statistics are fit on the training fold only — the same
leakage discipline the rest of the toolkit enforces.
"""
from __future__ import annotations

import numpy as np

from ..metrics import auroc, cluster_bootstrap

__all__ = ["linear_probe_auroc", "probe_report"]


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _group_kfold(groups, n_splits, seed):
    """Assign each row a fold id so that a whole group lands in one fold.

    Returns ``(N,)`` int fold ids in ``[0, n_splits)``.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    fold_of = {}
    for fi, chunk in enumerate(np.array_split(perm, n_splits)):
        for g in chunk:
            fold_of[g] = fi
    return np.array([fold_of[g] for g in groups], dtype=int)


def _ridge_scores(x_tr, y_tr, x_va, l2):
    """Ridge probe: fit on train, score val. Standardisation fit on train only.

    y_tr is the 0/1 indicator of the positive attribute value; the returned
    scores are the linear decision values (only their ordering matters for AUROC,
    so the intercept is dropped and y is centred).
    """
    mu = x_tr.mean(axis=0)
    sd = x_tr.std(axis=0) + 1e-8
    x_tr = (x_tr - mu) / sd
    x_va = (x_va - mu) / sd
    d = x_tr.shape[1]
    # errstate: some numpy builds emit spurious divide/overflow warnings from the
    # matmul SIMD kernel on finite data; inputs here are standardised and finite.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        a = x_tr.T @ x_tr + l2 * np.eye(d)
        w = np.linalg.solve(a, x_tr.T @ (y_tr - y_tr.mean()))
        return x_va @ w


def _binary_probe(features, target, groups, l2, n_splits, seed):
    """Out-of-fold linear-probe AUROC for a *binary* target, with a
    group-cluster bootstrap CI. Returns ``dict`` or ``None`` if not evaluable."""
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=int)
    groups = np.asarray(groups)

    if len(np.unique(target)) < 2:
        return None  # only one attribute value present -> nothing to decode
    if len(np.unique(groups)) < n_splits:
        n_splits = max(2, len(np.unique(groups)))

    fold = _group_kfold(groups, n_splits, seed)
    oof = np.full(len(target), np.nan)
    for f in range(fold.max() + 1):
        tr = fold != f
        va = fold == f
        if not va.any() or len(np.unique(target[tr])) < 2:
            continue  # can't fit a 2-class probe on this train fold
        oof[va] = _ridge_scores(features[tr], target[tr].astype(float),
                                features[va], l2)

    valid = ~np.isnan(oof)
    if valid.sum() < 2 or len(np.unique(target[valid])) < 2:
        return None
    v = np.where(valid)[0]
    sc, tg, gr = oof[v], target[v], groups[v]

    point, lo, hi, vf = cluster_bootstrap(
        gr, lambda idx: auroc(sc[idx], tg[idx]),
        n_boot=2000, seed=seed, verbose=False, return_valid_frac=True)
    return {"auroc": float(point), "ci": (float(lo), float(hi)),
            "n": int(valid.sum()), "n_groups": int(len(np.unique(gr))),
            "valid_frac": float(vf), "ci_kind": "matched"}


def linear_probe_auroc(features, target, groups, l2=1.0, n_splits=5, seed=42):
    """Group-aware, out-of-fold linear-probe AUROC that an attribute is decodable
    from ``features``.

    ``target`` may be binary or multi-class (int codes). Multi-class is scored
    one-vs-rest and macro-averaged; the reported CI is the widest one-vs-rest CI
    (a conservative summary). Returns a dict, or ``None`` if not evaluable
    (single attribute value, or too few groups/classes to fit a probe).
    """
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=int)
    vals = np.unique(target)

    if len(vals) == 2:
        return _binary_probe(features, (target == vals[1]).astype(int),
                             groups, l2, n_splits, seed)

    # multi-class: one-vs-rest macro average
    per = []
    for v in vals:
        r = _binary_probe(features, (target == v).astype(int),
                          groups, l2, n_splits, seed)
        if r is not None:
            per.append(r)
    if not per:
        return None
    macro = float(np.mean([r["auroc"] for r in per]))
    widest = max(per, key=lambda r: r["ci"][1] - r["ci"][0])
    return {"auroc": macro, "ci": widest["ci"], "n": per[0]["n"],
            "n_groups": per[0]["n_groups"], "one_vs_rest": per,
            "ci_kind": "widest_ovr",
            "valid_frac": float(min(r.get("valid_frac", 1.0) for r in per)),
            "note": "multi-class: point is the macro-averaged AUROC; the CI is the "
                    "widest per-class one-vs-rest CI, NOT the CI of the macro AUROC"}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
# A probe only counts as 'positive' if the CI lower bound clears chance by this
# margin (printed in the report so the threshold is never hidden).
MARGIN = 0.60


def _pow(detail, overall):
    """Append an underpowered-CI caveat when many bootstrap resamples degenerated."""
    if overall.get("valid_frac", 1.0) < 0.90:
        vf = overall["valid_frac"]
        return (detail + f" [CAUTION: only {vf:.0%} of bootstrap resamples were "
                "valid — the CI is underpowered; treat this verdict as tentative]")
    return detail


def _verdict(overall, within):
    """Turn probe numbers into a measured, non-overclaiming verdict.

    ``within`` maps class name -> probe dict (has a ``ci``) or a ``{"skipped":…}``
    marker; only entries that actually ran count. Distinguishes: decodable and
    confirmed within-class (SHORTCUT ENCODED), decodable but class-collinear
    (AMBIGUOUS), decodable with no usable within-class check (DECODABLE), a probe
    genuinely near chance (NOT DECODABLE), and a high point estimate whose CI is
    just too wide (NOT ESTABLISHED — underpowered, not truly absent).
    """
    if overall is None:
        return ("INCONCLUSIVE", "attribute has a single value or too few "
                "groups to fit a probe")
    lo = overall["ci"][0]
    point = overall["auroc"]

    # within-class probes that ACTUALLY ran (produced a CI), vs skipped markers
    ran = {c: r for c, r in within.items() if isinstance(r, dict) and "ci" in r}
    hits = [c for c, r in ran.items() if r["ci"][0] > MARGIN]

    if lo > MARGIN:                                    # decodable overall
        if hits:
            return ("SHORTCUT ENCODED", _pow(
                "the attribute is linearly decodable from the features and remains "
                f"decodable within fixed classes ({', '.join(hits)}) — encoded "
                "beyond class-collinearity. It is therefore available to the model "
                "as a potential shortcut; IF the decision head relies on it, expect "
                "degradation when its link to the label shifts", overall))
        if ran:                                        # ran, none cleared the margin
            return ("AMBIGUOUS", _pow(
                "decodable overall but not within any single fixed class — the "
                "overall signal may be driven by class-collinearity rather than a "
                "genuinely encoded attribute; gather more per-class samples", overall))
        return ("DECODABLE", _pow(                     # no within-class probe ran
            "the attribute is linearly decodable from the features, but the "
            "within-class check could not run (need >1 class with both attribute "
            "values present), so class-collinearity cannot be ruled out", overall))

    # not decodable at the margin — separate 'near chance' from 'underpowered'
    if point <= MARGIN:
        return ("NOT DECODABLE", "no evidence the attribute is linearly encoded in "
                f"the features (point AUROC {point:.2f}, near chance); this does not "
                "exclude a non-linear shortcut")
    return ("NOT ESTABLISHED", f"point AUROC {point:.2f} is above chance but the 95% "
            f"CI lower bound ({lo:.2f}) is below the {MARGIN:.2f} margin — likely "
            "underpowered rather than truly absent; gather more groups")


def probe_report(features, attr_codes, labels, groups, *, attr_name="attribute",
                 attr_values=None, class_names=None, l2=1.0, n_splits=5, seed=42,
                 min_per_class=40):
    """Full shortcut-probe audit for one metadata attribute.

    Args:
      features:   ``(N, D)`` frozen feature matrix (model penultimate layer, or
                  any embedding).
      attr_codes: ``(N,)`` int codes of the metadata attribute under test
                  (imaging mode, scanner, site, …).
      labels:     ``(N,)`` int class labels (the diagnostic target).
      groups:     ``(N,)`` group id (patient/case) for leakage-safe folds and CI.
      attr_name:  display name of the attribute.
      attr_values / class_names: optional decoded names for reporting.
      min_per_class: skip a within-class probe if the class has fewer rows.

    Returns a dict:
      ``{"attribute", "overall", "within_class", "verdict", "detail"}``
    """
    features = np.asarray(features, dtype=float)
    attr_codes = np.asarray(attr_codes, dtype=int)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)

    overall = linear_probe_auroc(features, attr_codes, groups, l2, n_splits, seed)

    # within-class probe only makes sense with >=2 classes: it holds the diagnosis
    # constant to strip class-collinearity. With a single class there is nothing to
    # hold constant, so we do not run it (and _verdict will not claim we did).
    within = {}
    classes_present = np.unique(labels)
    for c in (classes_present if len(classes_present) >= 2 else []):
        name = (class_names[c] if class_names is not None
                and c < len(class_names) else f"class_{c}")
        m = labels == c
        if m.sum() < min_per_class:
            within[name] = {"skipped": f"only {int(m.sum())} rows (<{min_per_class})"}
            continue
        if len(np.unique(attr_codes[m])) < 2:
            within[name] = {"skipped": "attribute constant within this class"}
            continue
        r = linear_probe_auroc(features[m], attr_codes[m], groups[m],
                               l2, n_splits, seed)
        within[name] = r if r is not None else {"skipped": "not evaluable"}

    verdict, detail = _verdict(overall, within)
    return {"attribute": attr_name, "overall": overall,
            "within_class": within, "verdict": verdict, "detail": detail}


def format_report(rep):
    """Render a ``probe_report`` result as a short human-readable block."""
    def fmt(r):
        if r is None:
            return "n/a"
        if "skipped" in r:
            return f"skipped ({r['skipped']})"
        lo, hi = r["ci"]
        s = f"AUROC {r['auroc']:.3f}  (95% CI {lo:.3f}–{hi:.3f}, n={r['n']}"
        vf = r.get("valid_frac")
        if vf is not None and vf < 0.90:
            s += f", only {vf:.0%} valid resamples — underpowered"
        s += ")"
        if r.get("ci_kind") == "widest_ovr":
            s += "  [multi-class: point=macro-AUROC; CI=widest per-class CI, not macro CI]"
        return s

    lines = [f"shortcut probe · attribute = {rep['attribute']!r}  "
             f"(decodable if CI lower bound > {MARGIN:.2f})",
             f"  overall        {fmt(rep['overall'])}"]
    for cname, r in rep["within_class"].items():
        lines.append(f"  within {cname:12s} {fmt(r)}")
    lines.append(f"  -> {rep['verdict']}: {rep['detail']}")
    return "\n".join(lines)
