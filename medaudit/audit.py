"""The audit orchestrator — turn a manifest + features + config into one report.

This is the ``medaudit audit`` engine. It is deliberately split in two:

  * ``run_audit_arrays(manifest, features, config)`` — pure, array-in/dict-out,
    no file IO and no torch. This is what the unit tests drive with synthetic
    features, so the orchestration logic is verified without a GPU or a model.
  * ``run_audit(config_path)`` — the file-driven wrapper: load the manifest CSV,
    load a precomputed ``features.npy`` (rows aligned to the manifest), read the
    JSON config, and call the pure core.

Features are supplied precomputed on purpose. Extracting them from a torch model
is an optional, heavyweight step (``medaudit.models``, planned); keeping it out
of the orchestrator means the whole audit flow stays importable and testable
with numpy alone.

Config (JSON) schema — every section optional, sane defaults:

    {
      "manifest": "data.csv",        // required for run_audit()
      "image_root": "",
      "features":  "features.npy",   // (N, D), row-aligned to the manifest
      "split":   {"val_ratio": 0.15, "test_ratio": 0.15, "seed": 42},
      "probe":   {"attributes": ["mode"], "l2": 1.0, "n_splits": 5,
                  "min_per_class": 40},
      "leakage": {"threshold": 0.90, "top_k": 50}
    }

If ``probe.attributes`` is omitted, every ``attr_*`` column in the manifest is
probed.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .manifest import load_manifest
from .splits import group_stratified_split
from .audits import probe, leakage

__all__ = ["run_audit_arrays", "run_audit", "render_report"]


def _strip(attr):
    return attr[len("attr_"):] if attr.startswith("attr_") else attr


def run_audit_arrays(manifest, features, config=None):
    """Run the audit on in-memory arrays. Returns a plain-dict report.

    Args:
      manifest: a ``medaudit.manifest.Manifest``.
      features: ``(N, D)`` feature matrix, row-aligned to ``manifest``.
      config:   dict (see module docstring); ``None`` uses all defaults.
    """
    config = config or {}
    features = np.asarray(features, dtype=float)
    labels = manifest.labels()
    groups = manifest.groups()
    if len(features) != len(manifest):
        raise ValueError(f"features has {len(features)} rows but manifest has "
                         f"{len(manifest)} — they must be row-aligned")
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (N, D); got shape {features.shape}")

    # --- split of record ----------------------------------------------------- #
    # Prefer a user-provided 'split' column: then the leakage audit's group-leakage
    # check is a real one. Otherwise we generate a group-clean split ourselves — in
    # which case group leakage is vacuous by construction, so we pass leak_groups=None
    # and the audit honestly reports group leakage as NOT ASSESSED rather than
    # claiming a check it never performed.
    sp = config.get("split", {})
    seed = sp.get("seed", 42)
    if "split" in manifest.rows[0]:
        split_labels = np.array([r["split"] for r in manifest.rows], dtype=object)
        leak_groups = groups
        split_source = "provided"
        split_sizes = {s: int((split_labels == s).sum())
                       for s in dict.fromkeys(split_labels.tolist())}
    else:
        splits = group_stratified_split(
            groups, labels, val_ratio=sp.get("val_ratio", 0.15),
            test_ratio=sp.get("test_ratio", 0.15), seed=seed)
        split_labels = np.empty(len(labels), dtype=object)
        for name, idx in splits.items():
            split_labels[idx] = name
        leak_groups = None
        split_source = "generated"
        split_sizes = {k: int(len(v)) for k, v in splits.items()}

    # --- shortcut probe, one per requested attribute ------------------------- #
    pr = config.get("probe", {})
    attrs = pr.get("attributes") or manifest.attr_cols
    probe_reports = {}
    for a in attrs:
        try:
            codes, vals = manifest.attribute(a)
        except KeyError as e:
            probe_reports[_strip(a)] = {"error": str(e)}
            continue
        probe_reports[_strip(a)] = probe.probe_report(
            features, codes, labels, groups,
            attr_name=_strip(a), attr_values=vals, class_names=manifest.classes,
            l2=pr.get("l2", 1.0), n_splits=pr.get("n_splits", 5),
            seed=seed, min_per_class=pr.get("min_per_class", 40))

    # --- leakage audit ------------------------------------------------------- #
    lk = config.get("leakage", {})
    leak_report = leakage.leakage_report(
        features, split_labels, leak_groups,
        threshold=lk.get("threshold", 0.90), top_k=lk.get("top_k", 50))

    # Say WHY the group check was skipped. Passing leak_groups=None above makes
    # leakage_report explain it as "no group ids supplied", which is false when
    # the manifest carries them — we declined to check a split we generated
    # ourselves. A wrong reason is worse than no reason.
    if split_source == "generated" and not leak_report["group_assessed"]:
        leak_report["group_note"] = (
            "medaudit generated this split itself and made it group-clean by "
            "construction, so checking it would prove nothing about your pipeline; "
            "supply a 'split' column to audit your own assignment")
        if leak_report["verdict"] == "CLEAN":
            leak_report["detail"] = (
                "group leakage NOT ASSESSED (" + leak_report["group_note"] + "). No "
                f"cross-split pair at cosine ≥ {leak_report['threshold']}. Note this "
                "only rules out near-duplicates the embedding can see")

    return {
        "n_rows": len(labels),
        "n_features": int(features.shape[1]),
        "classes": list(manifest.classes),
        "split_sizes": split_sizes,
        "split_source": split_source,
        "probe": probe_reports,
        "leakage": leak_report,
    }


def run_audit(config_path):
    """File-driven audit: read config JSON, load manifest + features, run."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    base = os.path.dirname(os.path.abspath(config_path))

    def rel(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    if "manifest" not in config or "features" not in config:
        raise ValueError("config must give both 'manifest' and 'features' paths")
    manifest = load_manifest(rel(config["manifest"]), config.get("image_root", ""))
    features = np.load(rel(config["features"]))
    return run_audit_arrays(manifest, features, config)


def render_report(report):
    """Render an audit report dict as a plain-text block."""
    lines = ["=" * 68,
             "medaudit reliability audit",
             "=" * 68,
             f"rows={report['n_rows']}  features={report['n_features']}  "
             f"classes={report['classes']}",
             "split: " + "  ".join(f"{k}={v}" for k, v in report["split_sizes"].items())
             + f"   [{report.get('split_source', 'generated')}]",
             ""]
    if report.get("split_source") == "generated":
        lines += ["note: no 'split' column in the manifest, so medaudit generated a",
                  "      group-clean split itself — group leakage is therefore not",
                  "      assessed below. Add a 'split' column to audit your own",
                  "      train/val/test assignment.", ""]

    lines.append("SHORTCUT PROBE")
    lines.append("-" * 68)
    if not report["probe"]:
        lines.append("  (no attributes to probe — add attr_* columns to the manifest)")
    for attr, rep in report["probe"].items():
        if "error" in rep:
            lines.append(f"  {attr}: {rep['error']}")
        else:
            lines.append(probe.format_report(rep))
        lines.append("")

    lines.append("LEAKAGE")
    lines.append("-" * 68)
    lines.append(leakage.format_report(report["leakage"]))
    lines.append("=" * 68)
    return "\n".join(lines)
