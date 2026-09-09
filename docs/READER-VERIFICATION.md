# Reader revision: verification and provenance

Local verification snapshot: 2026-09-09, before GitHub publication. This record
describes the reader-facing revision; it is not a new challenge submission or
validation of medical performance. See Git history for the published revision.

## Baseline and preservation

The public repository and original local repository both resolved to
`71c6889f7b2aa34b247af9cffddcd4f2f12e4854` (read-only `git ls-remote` check).
An independent local clone was created with `--no-hardlinks`, on branch
`codex/mec-reader-entry`. The original checkout was not edited; its uncommitted
`DESIGN.md` was not imported or overwritten. The baseline tutorial remains
[available at its original commit](https://github.com/sunce764/medaudit/blob/71c6889f7b2aa34b247af9cffddcd4f2f12e4854/tutorial/from-pixels-to-patients.md).

The local historical PDF was retained unchanged (SHA-256
`5bb312d82fd3253f99eda33996086993cdcd9607a5c979c9795423cf7e93b555`).
A privately retained finalist email supports finalist status. It does not prove
which exact PDF OpenReview holds; that version match remains unverified. Private
submission files, email screenshots and video production materials are not part
of this reader revision.

## What changed

- README now leads with the tutorial, audience, learning outcomes and runnable path.
- `tutorial/RUNNING.md` explains outputs, limitations, input conventions and exercises.
- `tutorial/worked_examples.py` supplies explicit synthetic inputs for existing
  calibration metrics and fixed-sensitivity/specificity PPV arithmetic.
- Tutorial navigation, AI disclosure and reference [15] metadata were corrected.
  Demo docstrings no longer promise to reproduce every historical result or
  equate encoded metadata with classifier reliance.
- Core package behavior, dependency ranges, original research numbers, figures
  and license were not changed.

## Installation and execution

Observed platform: macOS, Python 3.14.5, NumPy 2.5.3; test runner pytest 9.1.1.
These are tested versions, not a claim of testing every supported Python/NumPy
combination. Windows activation instructions were not executed on Windows.

| Check | Observed result |
|---|---|
| Fresh venv, `python -m pip install -e '.[dev]'` | Exit 0; package and test dependencies installed |
| Second fresh venv, `python -m pip install .` | Exit 0; regular wheel installation, runtime dependencies only MedAudit and NumPy |
| `python -m pytest tests -q` | 29 passed, exit 0 |
| `python tutorial/make_demo.py` | A–D complete, exit 0; A/B numbers match the tutorial's six AUROC/CI rows |
| CLI C with explicit output path | Report saved, exit 0; generated split reported as NOT ASSESSED for group leakage |
| CLI D with explicit output path | Report saved, exit 0; planted `patient0` overlap detected as GROUP LEAKAGE |
| `python tutorial/worked_examples.py` | Exit 0; AUROC 0.938, ECE 0.300, Brier 0.250; PPV 0.947/0.667/0.154 |
| `python -m pip check` | No broken requirements, exit 0 |

The initial test invocation occurred before installation finished and reported
`No module named pytest`; it was rerun after installation completed. The successful
29-test result is the completed run, not that first invocation.

The generated C/D inputs and reports stay in ignored `tutorial/demo/`. Detailed
local logs, installed dependency snapshots, bibliographic metadata and review
checks stay in ignored `_internal/reader-review/`; they are not public assets.

## License, citations and disclosure

`LICENSE` grants MIT terms to the software and accompanying documentation;
`pyproject.toml` agrees and attributes Chao Sheng. The four existing tutorial
charts have generation code using synthetic data/arithmetic. No external dataset
license is inferred from the code license, and no patient data is needed here.

Bibliographic checks used publisher-deposited Crossref metadata for references
[1–3], [5–6], [9–10], [13]; PMLR for [4], [12]; CVF for [11]; arXiv for [7–8],
[15]; and Cambridge's book metadata for [14]. Some DOI landing pages rejected
web requests; Crossref supplied their metadata. Initial Crossref rate limits
were resolved by a later sequential retry. This is an identity/metadata check,
not a full-text verification of every claim supported by the citations.

Reference [15] was corrected to the title and author spelling of
[arXiv v1](https://arxiv.org/abs/2604.11171v1), including “van Eijck van Heslinga”.
The paper says its author list is being finalized; v1 is therefore cited explicitly.
No new research conclusion is inferred. The older research statistics and the
paper's numerical result cited in §5 were not recomputed in this reader revision.

The tutorial's AI disclosure describes the original Claude Code assistance,
the author’s responsibility, and Codex's work on this reader revision. The
earlier wording remains available through the fixed-commit tutorial link above.

## Official requirements and remaining limits

The [official MEC page](https://miccai-sb.github.io/challenge.html), checked on
2026-09-09, accepts GitHub source/documentation and requires English submissions,
public/free availability, and disclosure of AI tools and their use. Its list of
strong submission qualities is explicitly **at least one**, not a requirement
to satisfy every item. A synthetic teaching example is not claimed to satisfy
the distinct “runnable code on real medical data” item. Meeting these format
requirements does not establish competitiveness or an award.

Remaining outside this verification scope: matching the final platform PDF,
video upload and challenge submission. The original tutorial's
historical medical findings remain subject to their stated single-seed and
reproducibility limits. No GPU work or research rerun was performed.

## Final reader checks

- 22 relative links/anchors across the README, tutorial, running guide and this
  record resolved. A deliberately broken link was rejected by the same checker.
- Six anonymous HTTP reads (baseline tutorial, license and four charts) returned
  200, and their hashes matched the baseline/local preserved assets.
- The regular wheel was imported from its `site-packages` outside the repository.
  In that core-only environment A–D output matched the editable-install output
  byte for byte; the calibration/prevalence script also completed.
- The six tutorial probe AUROC/CI rows matched fresh output. Independent small
  arithmetic checks confirmed worked-example AUROC (15/16), Brier (2/8) and ECE
  (0.300). `git diff --check` passed.
- Final original-checkout status remained only `M DESIGN.md`; its SHA-256 remained
  `2d8713794b49e69dc1ce0e6f4db9865ef32fb9ae7a9b0c2cd4b13df9d1ac3366`.
  The PDF hash above remained unchanged. The outer isolated repository was clean.

At the end of the local review, the six reader-revision files were uncommitted
and no remote write, push, upload or submission had occurred. GitHub publication
was subsequently authorized for these six files only; this pre-publication
snapshot does not describe the later remote state. Private evidence remains local.
