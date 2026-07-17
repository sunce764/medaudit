"""Leakage audit — is train/test independence actually holding?

Two failure modes, both of which silently inflate reported performance:

  * **group leakage** — the same patient/subject/case has rows in more than one
    split. Cheap to check exactly from ids; the split tools already prevent it,
    this audit re-checks a split you were *handed*.
  * **near-duplicate leakage** — visually near-identical images (same lesion a
    few frames apart, re-exports, crops) split across train and test. There is
    no id to catch these; you have to compare the images themselves.

The instrument for near-duplicates is **cosine similarity in an embedding
space**, not a perceptual hash. In a homogeneous medical domain (every frame is
pink mucosa / grey ultrasound) perceptual hashes collapse — everything looks
alike — and flag hundreds of false pairs. A learned embedding keeps genuine
near-duplicates separable from merely same-domain images. (This is a lesson
paid for: a dHash scan once reported 535 cross-set "duplicates" that were, on
inspection, simply different images of the same tissue.)

Give this audit whatever embeddings you have (ImageNet backbone, the model's own
features); it reports the worst cross-split pairs and lets you eyeball them —
it does not silently delete anything.
"""
from __future__ import annotations

import numpy as np

__all__ = ["near_duplicate_pairs", "group_leakage", "leakage_report"]


def _unit(features):
    f = np.asarray(features, dtype=float)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    return f / np.clip(n, 1e-12, None)


def near_duplicate_pairs(features, split_labels, threshold=0.90, top_k=50):
    """Cross-split near-duplicate pairs by embedding cosine similarity.

    Args:
      features:     ``(N, D)`` embeddings (ImageNet backbone, model features, …).
      split_labels: ``(N,)`` split id per row (e.g. "train"/"val"/"test").
      threshold:    cosine ≥ this flags a pair (domain-dependent; calibrate by
                    eyeballing the top pairs, not by trusting a default).
      top_k:        keep at most this many worst pairs; ``None`` returns all.

    Returns list of ``(i, j, cosine)`` with ``split[i] != split[j]``, cosine
    descending. Only cross-split pairs — within-split duplicates are not leakage.
    """
    u = _unit(features)
    labels = np.asarray(split_labels)
    uniq = list(dict.fromkeys(labels.tolist()))  # preserve first-seen order
    pairs = []
    for a in range(len(uniq)):
        for b in range(a + 1, len(uniq)):
            ia = np.where(labels == uniq[a])[0]
            ib = np.where(labels == uniq[b])[0]
            if len(ia) == 0 or len(ib) == 0:
                continue
            # errstate: some numpy builds emit spurious divide/overflow warnings
            # from the matmul SIMD kernel on finite data; inputs are unit-norm here.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sim = u[ia] @ u[ib].T             # (|ia|, |ib|) cosine
            hits = np.argwhere(sim >= threshold)
            for r, c in hits:
                pairs.append((int(ia[r]), int(ib[c]), float(sim[r, c])))
    pairs.sort(key=lambda p: -p[2])
    return pairs if top_k is None else pairs[:top_k]


def group_leakage(groups, split_labels):
    """Groups (patient/case ids) appearing in more than one split.

    Returns ``{group: [splits...]}`` for every leaked group (empty dict = clean).
    """
    groups = np.asarray(groups)
    labels = np.asarray(split_labels)
    out = {}
    for g in np.unique(groups):
        s = sorted(set(labels[groups == g].tolist()))
        if len(s) > 1:
            out[g if not isinstance(g, np.generic) else g.item()] = s
    return out


def leakage_report(features, split_labels, groups=None, *, threshold=0.90,
                   top_k=50):
    """Full leakage audit: group leakage (if ids given) + cross-split
    near-duplicates.

    Returns a dict:
      ``{"group_leak", "n_group_leak", "near_dup", "n_near_dup",
         "threshold", "verdict", "detail"}``
    """
    split_labels = np.asarray(split_labels)
    n_rows = len(split_labels)

    # group leakage is only meaningfully assessed if real group ids were supplied
    # (present, and not all-unique — all-unique means there is no grouping to check).
    group_assessed = (groups is not None
                      and len(np.unique(np.asarray(groups))) < n_rows)
    gl = group_leakage(groups, split_labels) if group_assessed else {}

    # compute the FULL cross-split near-duplicate list, then show only the worst
    # top_k — but report the TRUE total so the count never silently saturates.
    all_pairs = near_duplicate_pairs(features, split_labels, threshold, top_k=None)
    n_total = len(all_pairs)
    shown = all_pairs[:top_k]

    if gl:
        verdict = "GROUP LEAKAGE"
        detail = (f"{len(gl)} group(s) span multiple splits (e.g. "
                  f"{list(gl)[:3]}) — fix the split before trusting any metric")
    elif n_total:
        cap = f"showing worst {len(shown)} of {n_total}; " if n_total > len(shown) else ""
        verdict = "NEAR-DUPLICATES"
        detail = (f"{n_total} cross-split pair(s) at cosine ≥ {threshold} "
                  f"({cap}worst {all_pairs[0][2]:.3f}) — eyeball them; if genuinely "
                  "the same image, they inflate the test score")
    else:
        verdict = "CLEAN"
        ga = ("no group leakage" if group_assessed
              else "group leakage NOT ASSESSED (no group ids supplied to this check)")
        detail = (f"{ga}; no cross-split pair at cosine ≥ {threshold}. Note this "
                  "only rules out near-duplicates the embedding can see")

    return {"group_leak": gl, "n_group_leak": len(gl), "group_assessed": group_assessed,
            "near_dup": shown, "n_near_dup": n_total, "n_near_dup_shown": len(shown),
            "threshold": threshold, "verdict": verdict, "detail": detail}


def format_report(rep):
    """Render a ``leakage_report`` result as a short human-readable block."""
    lines = ["leakage audit"]
    if rep.get("group_assessed", True):
        lines.append(f"  group leakage   {rep['n_group_leak']} group(s)")
    else:
        # group_note lets a caller state the REAL reason. The default is only
        # true when no ids reached this function; an orchestrator that withheld
        # them must say so itself rather than let this stand as the explanation.
        lines.append("  group leakage   NOT ASSESSED — " +
                     rep.get("group_note", "no group ids supplied to this check"))
    for g, s in list(rep["group_leak"].items())[:5]:
        lines.append(f"    group {g!r:>12} in splits {s}")
    total = rep["n_near_dup"]
    shown = rep.get("n_near_dup_shown", len(rep["near_dup"]))
    cap = f" (showing worst {shown})" if total > shown else ""
    lines.append(f"  near-duplicates {total} cross-split pair(s){cap} "
                 f"(cosine ≥ {rep['threshold']})")
    for i, j, c in rep["near_dup"][:5]:
        lines.append(f"    rows {i:>6} ~ {j:<6}  cosine {c:.3f}")
    lines.append(f"  -> {rep['verdict']}: {rep['detail']}")
    return "\n".join(lines)
