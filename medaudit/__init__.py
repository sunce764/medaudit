"""medaudit — a reliability audit toolkit for medical-image classifiers.

Runs the checks a clinical-ML reviewer would demand, from one config:
acquisition/metadata shortcut probing, near-duplicate & group leakage,
calibration, prevalence-shift decomposition, and external validation —
producing a single self-contained audit report.

Metrics are hand-written (no sklearn dependency) and unit-tested against
independent brute-force references (see tests/test_metrics.py).
"""
from . import metrics  # noqa: F401
from . import splits  # noqa: F401
from . import manifest  # noqa: F401
from . import audits  # noqa: F401
from . import audit  # noqa: F401

__version__ = "0.2.0"
__all__ = ["metrics", "splits", "manifest", "audits", "audit", "__version__"]
