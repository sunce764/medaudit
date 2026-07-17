"""medaudit.audits — the individual reliability checks.

Each audit is a self-contained function that takes arrays (features, labels,
groups, a metadata attribute) and returns a plain-dict result plus a short,
non-overclaiming interpretation string. Audits never load heavy libraries at
import time and are unit-tested against synthetic data with a known ground truth.

Available:
  probe — acquisition/metadata shortcut probe (is an attribute linearly
          decodable from the model's features, overall and within each class?).
"""
from . import probe  # noqa: F401

__all__ = ["probe"]
