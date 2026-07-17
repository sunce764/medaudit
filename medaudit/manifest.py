"""The audit manifest: a CSV describing the dataset to audit.

Required columns: ``path``, ``label``.
Optional: ``group`` (patient/subject/case id — for leakage-safe splits and
cluster bootstrap; defaults to the row index = no grouping), and any number of
``attr_*`` columns holding acquisition/metadata attributes (imaging mode,
scanner, site, stain, …) that the shortcut probe can test for.
"""
from __future__ import annotations

import csv
import os

import numpy as np

REQUIRED = ("path", "label")


class Manifest:
    """A validated view over the audit rows, exposing arrays the audits consume."""

    def __init__(self, rows, image_root=""):
        if not rows:
            raise ValueError("Manifest requires at least one row")
        self.rows = rows
        self.image_root = image_root
        self.classes = sorted({r["label"] for r in rows})
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.attr_cols = [k for k in rows[0].keys() if k.startswith("attr_")]
        self.has_group_col = "group" in rows[0]

    def __len__(self):
        return len(self.rows)

    def labels(self):
        """(N,) int class index."""
        return np.array([self.cls2idx[r["label"]] for r in self.rows])

    def groups(self):
        """(N,) group id.

        If there is no ``group`` column at all, every row is its own group
        (``_row{i}``) — the documented, no-grouping fallback. If the column
        *exists* but a cell is blank, that is an error, not a silent singleton:
        a leakage/uncertainty audit must not quietly treat an unlabelled row as
        an independent case (it would understate leakage and narrow the CI).
        """
        if not self.has_group_col:
            return np.array([f"_row{i}" for i in range(len(self.rows))])
        blanks = [i for i, r in enumerate(self.rows) if not (r.get("group") or "").strip()]
        if blanks:
            raise ValueError(
                f"'group' column present but blank in {len(blanks)} row(s) "
                f"(e.g. rows {blanks[:5]}); fill every group id or drop the column "
                f"entirely to opt out of grouping — blanks are not silently ungrouped")
        return np.array([r["group"] for r in self.rows])

    def attribute(self, name):
        """Encode an ``attr_*`` column to ints. Returns (codes, value_names)."""
        col = name if name.startswith("attr_") else "attr_" + name
        if col not in self.rows[0]:
            raise KeyError(f"no attribute column {col!r}; available: {self.attr_cols}")
        vals = sorted({r[col] for r in self.rows})
        v2i = {v: i for i, v in enumerate(vals)}
        return np.array([v2i[r[col]] for r in self.rows]), vals

    def paths(self):
        """Absolute-ish image paths (``image_root`` joined)."""
        return [os.path.join(self.image_root, r["path"]) for r in self.rows]

    def summary(self):
        lab = self.labels()
        lines = [f"{len(self)} rows · {len(self.classes)} classes · "
                 f"{len(set(self.groups()))} groups · attrs={self.attr_cols or 'none'}"]
        for c, i in self.cls2idx.items():
            lines.append(f"    {c:20s} {(lab == i).sum():6d}")
        return "\n".join(lines)


def load_manifest(csv_path, image_root=""):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"empty manifest: {csv_path}")
    for c in REQUIRED:
        if c not in rows[0]:
            raise ValueError(f"manifest {csv_path} missing required column {c!r}; "
                             f"got {list(rows[0].keys())}")
    return Manifest(rows, image_root)
