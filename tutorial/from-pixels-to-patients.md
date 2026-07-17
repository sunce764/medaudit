# From Pixels to Patients: a hands-on reliability audit of a medical-image classifier

*The checks a clinical reviewer runs before trusting your AUROC.*

**MICCAI Educational Challenge 2026 submission.** Author: Chao Sheng
(Biomedical Engineering, Guangdong Medical University). Accompanying code:
the `medaudit` toolkit (open-source, MIT).

> **Audience & goal.** For Masters/PhD readers who can train a classifier but
> have not yet had one fail *silently*. By the end you will be able to run the
> **shortcut** and **leakage** audits on your own model from one command, and
> compute **calibration** and **prevalence** yourself from the toolkit's tested
> metric primitives (§4, §5) — and, more to the point, read all four the way a
> reviewer would.
>
> **On the data.** Code blocks run on a synthetic cohort — deliberately. The
> dataset behind the war stories cannot be redistributed, and a tutorial you can
> only run with restricted access is a tutorial nobody runs. Synthetic data also
> buys what real data cannot: **we know the ground truth**, so §2 can show the
> audit getting it right *and* getting it convincingly wrong. The war stories are
> **single-seed** figures from a real classifier trained on a 2025 public
> cystoscopy dataset — what the audit does and how to read it, not settled claims.
> Saying so is the first instance of the discipline this piece is about; §6 is
> what happened when we forgot it.

---

## 0. Would you deploy this?

You train a classifier on cystoscopy images to flag malignant tissue against
everything else. On an internal held-out split — malignant-vs-rest, white-light
images — it reaches **AUROC 0.796**. Not spectacular; but then you get *another
hospital's* data, and it holds: **0.808**. Discrimination transfers across
centres — the result everyone hopes for and most papers do not get.

Your supervisor asks the only question that matters: *would you put this in a
clinic?*

The honest answer is: **you still don't know.** In the real audit those numbers
came from, the model had all but memorised **which light source the endoscopist
had switched on** — and a clinician reaches for blue light precisely *when they
already suspect a lesion*. Transferring discrimination did not mean the model had
learned the biology.

An AUROC tells you the model separates the classes *on data drawn like your test
split*. It does not tell you *what* it separates them by, whether your test split
was independent of training, whether a "0.9" prediction is really 90% likely, or
what the number means when the disease is rare.

Those four unknowns are four audits. This tutorial walks through each one, with
runnable code, and ends with the audit's most uncomfortable and most valuable
output: **the findings that did not survive scrutiny.**

A high AUROC is where the work starts, not where it ends.

---

## 1. Set-up: the manifest and one command

`medaudit` audits a *frozen* model — you do not retrain anything. You give it a
**manifest**: a CSV with one row per image.

```
path,label,group,attr_mode
img_0001.png,malignant,patient_07,blue_light
img_0002.png,benign,patient_07,white_light
...
```

- `label` — the diagnostic target.
- `group` — patient/case id. Everything downstream is *group-aware*: no patient
  spans train and test, and confidence intervals resample whole patients, not
  rows (images from one patient are not independent).
- `attr_*` — any acquisition/metadata attribute you can record: imaging mode,
  scanner, site, stain. **The audit's raw material** — what a model might latch
  onto instead of the biology.
- `split` (optional) — your own train/val/test assignment. Supply it and the
  leakage audit checks *your* split; omit it and `medaudit` builds a group-clean
  one itself, in which case there is no leakage to find and it says so rather
  than claiming a clean bill.

You also give it **features**: your model's penultimate-layer activations (or any
embedding) as an `(N, D)` array, row-aligned to the manifest. Auditing frozen
features means the toolkit runs on a laptop — no GPU, no retraining.

```python
from medaudit.audit import run_audit          # or: medaudit audit --config audit.json
report = run_audit("audit.json")
```

The rest of this tutorial is what that audit covers — the first two from that one
command, the last two composed from the same toolkit's primitives. Output blocks
come from the synthetic demo; war-story numbers come from the real audit and are
labelled as such.

---

## 2. Audit I — the shortcut probe ★

> ★ marks the sections to read if you read nothing else.

**The war story.** Before any model saw a pixel, the dataset already had a
problem. Suspicious lesions had, in routine practice, been imaged more often
under **blue light** — a contrast mode a clinician reaches for *when they already
suspect pathology*; ordinary tissue was mostly **white light**. Imaging mode and
lesion status were entangled at an **odds ratio of 33.5×** (Cramér's V = 0.462,
χ² ≈ 1725). The shortcut was sitting in the data, waiting to be learned.

A model can get a high AUROC by learning "blue light → malignant." That is a
**shortcut** [1]: predictive in your data, but not the disease, and it breaks the
moment the correlation does — a new hospital that blue-lights everything, or a
screening cohort where mode follows protocol, not suspicion. Not an exotic failure: models that appeared to detect COVID-19 from chest
radiographs were substantially reading laterality markers and source artefacts,
not lung pathology [2]; and aggregate metrics routinely hide subgroups where a
model fails outright [3].

**How to detect it: the linear probe.** Freeze the features. Train a simple
linear classifier to predict the *attribute* (imaging mode) from those features,
evaluated group-aware out-of-fold. If mode is linearly decodable, the model's
representation *encodes* it — it is available to be used as a shortcut.

```python
from medaudit.audits import probe

rep = probe.probe_report(features, attr_codes=mode, labels=diagnosis,
                         groups=patient_id, attr_name="mode")
print(probe.format_report(rep))
```

On the real data the probe read imaging mode out of the frozen features at
**AUROC 0.994** (95% CI 0.986–0.999).

**Now be unimpressed.** White and blue light differ by a global colour transform:
the two populations are *visibly* different before any question of pathology.
Essentially *any* embedding decodes that at ~0.99 — including a model scrupulously
ignoring mode when it decides. A high probe here was never a discovery; it was a
near-certainty.

That defines what a probe is *for*. It answers **"is this attribute available to
the model?"** — cheaply, on frozen features, in seconds. It does not answer **"is
the model using it?"** For a visually dominant attribute the first answer is
almost always yes, and all the interesting work lives in the second. So
`medaudit`'s verdict is hedged on purpose:

> *…the attribute is available to the model as a potential shortcut; **IF** the
> decision head relies on it, expect degradation when its link to the label shifts.
> Note this shows the attribute is* encoded, *not that it is* used.

The probe earns its keep on attributes you would **not** have guessed were
visible — scanner, site, acquisition date, stain batch. There, "available at
all?" is news.

**And detecting a shortcut is far easier than removing one.** We tried:
adversarial de-biasing (a gradient-reversal head, GRL) moved the probe only from
**0.994 to 0.969** — indistinguishable from noise at this CI width, and nowhere
near the 0.5 you would get if mode were gone. That is one mitigation, properly
configured, failing to erase one attribute; it is not a verdict on de-biasing in
general, and we do not have the evidence for one. (If you do reach for a
mitigation: Group-DRO [7] needs group labels throughout training plus careful
regularisation; last-layer retraining [8] needs them only for a small reweighting
set — cheaper to try first.)

**The trap in the probe — and how to escape it.** A high overall probe score is
*ambiguous*. If mode is correlated with the diagnosis (it was, at OR 33.5×), a probe
can score high just by reading the *class* and exploiting the correlation — that
is not the same as the features genuinely encoding mode. So `medaudit` runs a
second probe **within each fixed class**: among malignant-only images, can you
still decode mode? Holding the diagnosis constant removes the class-collinearity
explanation.

**Be precise about what that buys**, because it is less than it looks. The
within-class probe removes the ***label*-collinearity** explanation, not a
***severity*-collinearity** one. The endoscopist chose blue light by *looking*, so
within the malignant class the blue-lit lesions are plausibly the ones that looked
worse. A positive within-class probe is then consistent with two stories: the
features encode the light source, or they encode *how advanced the lesion is*,
which predicts it. Conditioning on a coarse label holds the paperwork constant,
not the biology. Separating those needs a severity covariate to condition on too,
or an attribute assigned independently of appearance. Say which you have.

Still, the distinction is the whole game. `make_demo.py` builds two cohorts
differing in **exactly one variable** — whether the features encode mode at all
(`beta`). The 85/15 correlation, class signal, noise, patient structure and sample
size are identical; a tutorial telling you to control your variables has to
control its own.

**Case A — the features encode only the class.** Mode is never written into them:

```
shortcut probe · attribute = 'mode'  (positive if CI lower bound > 0.60; point = median over fold partitions)
  overall        AUROC 0.841  (95% CI 0.796–0.884; 300 groups, 900 rows)
  within benign       AUROC 0.551  (95% CI 0.488–0.620; 149 groups, 447 rows)
  within malignant    AUROC 0.571  (95% CI 0.510–0.629; 151 groups, 453 rows)
  -> AMBIGUOUS: decodable overall but not within any single fixed class — the
     overall signal may be driven by class-collinearity rather than a genuinely
     encoded attribute; gather more per-class samples
```

**Case B — identical cohort; the features also encode mode.** One variable moved:

```
shortcut probe · attribute = 'mode'  (positive if CI lower bound > 0.60; point = median over fold partitions)
  overall        AUROC 0.905  (95% CI 0.873–0.933; 300 groups, 900 rows)
  within benign       AUROC 0.786  (95% CI 0.734–0.835; 149 groups, 447 rows)
  within malignant    AUROC 0.768  (95% CI 0.713–0.821; 151 groups, 453 rows)
  -> SHORTCUT ENCODED: … remains decodable within every fixed class tested
     (benign, malignant) — encoded beyond class-collinearity …
```

Look at the headlines: **0.84 and 0.91**. Both high. Both, on their own, look
like a model that has learned the light source. One of them is a model that never
saw mode at all. **The headline cannot tell them apart — the within-class row can.**
In case A the within-class rows fall to 0.55–0.57 — near chance, and far below
the 0.60 decision line; in case B they stay up near 0.77.

Two details there do quiet work. **"300 groups, 900 rows"**: the group count is
the sample size that matters — three images of one patient are not three
observations. And **"median over fold partitions"**: the pass runs five times over
different splits, because a verdict that changes with the split you drew is not a
verdict. When the runs disagree, the report says so.

**A caution that matters.** A linear probe near chance does *not* prove the model
is clean — it only rules out a *linearly* decodable shortcut. Non-linear
shortcuts exist. An audit narrows the space of failure modes; it never certifies
their absence.

---

## 3. Audit II — leakage

**The war story.** Endoscopy is video. Nearby frames of the same lesion are
near-identical. If two near-duplicate frames land one in train and one in test,
the model can *memorise* rather than generalise, and your test score is inflated
by an amount you cannot see in the AUROC.

Two kinds of leakage, both silent:

1. **Group leakage** — the same patient in two splits. Cheap to check exactly
   from ids; `medaudit` checks it exactly and returns a GROUP LEAKAGE verdict
   telling you to fix the split before trusting any metric. It reports; it does
   not gate — and it can only run this check on a split you actually give it
   (the optional `split` column in §1). Supply it, or the report says NOT
   ASSESSED — which it does out loud, not silently.
2. **Near-duplicate leakage** — visually near-identical images with *no shared
   id*. There is no key to join on; you must compare the images themselves.

**The instrument matters.** The intuitive tool is a perceptual hash (dHash).
It is the wrong one here. In a homogeneous domain — every frame is pink mucosa —
a perceptual hash sees everything as similar and **over-flags**. On the real data
a dHash scan reported **hundreds of cross-split "duplicates"** that were, on
visual inspection, simply *different images of the same kind of tissue.* We
verified by eye that a hash-distance-zero pair was two genuinely different frames.

The right instrument is **cosine similarity in a learned embedding space**: it
keeps true near-duplicates separable from merely same-domain images.

**But not just any embedding — here §2 bites back.** The obvious move is to reuse
the model's own features. Don't: §2 showed they decode mode at 0.994, so a
**giant mode axis** runs through that space. Cosine similarity in it is dominated
by "shot under the same light?", and your detector will rank two unrelated
blue-light frames above a genuine duplicate pair — rebuilding the dHash failure
with better tooling. Use an embedding independent of the attribute you are
worried about (a generic ImageNet backbone is usually fine), or residualise that
direction out first.

```python
from medaudit.audits import leakage

rep = leakage.leakage_report(features, split_labels, groups=patient_id,
                             threshold=0.90)
print(leakage.format_report(rep))
```

`medaudit` reports the worst cross-split pairs and their similarity — it **does
not silently delete anything.** You look at the top pairs and decide; the
threshold is something you calibrate by eye for your domain, not a magic default.
The lesson generalises past endoscopy: *choose the duplicate-detector that
matches your domain's homogeneity, and always eyeball what it flags.*

---

## 4. Audit III — calibration ★

**This mattered most in the real audit, and it is the section you are most likely
to skip.**

Go back to §0. Internal white-light malignant AUROC **0.796**; external **0.808**.
Discrimination transferred — genuinely good news. And yet under the *same* rule
(P > 0.5), sensitivity fell from **0.570** to **0.378**. Same ranking ability,
same threshold, a third of the sensitivity gone.

Nothing about the model's ability to *order* patients broke. What broke was the
mapping from scores to probabilities, so a threshold calibrated in one hospital
sat in the wrong place in the other. **The ROC curves nearly superimpose while the
reliability curves come apart.** Plot only the ROC and this failure is invisible —
it lives entirely in the axis AUROC discards [9]. A model can rank perfectly and
still be systematically over-confident, as modern networks generally are [4]:
at the bedside, that is the difference between a probability a clinician can act
on and a number that merely sorts.

`medaudit` uses hand-written, unit-tested calibration metrics (no `sklearn`, so
every number is auditable against its definition):

```python
from medaudit import metrics

print("ECE :", metrics.ece(probs, labels, strategy="equal_mass"))
print("Brier:", metrics.brier(probs, labels))
for conf, acc, n in metrics.reliability_curve(probs, labels):
    ...   # plot acc vs conf; the diagonal is perfect calibration
```

Three habits worth teaching:

- **Report Brier alongside ECE.** ECE is a biased estimator of calibration error
  and its value depends on how you bin [11, 12]; equal-mass (quantile) bins are
  more robust than equal-width, but ECE alone can mislead. Brier [5] is a proper
  scoring rule — pair them.
- **Calibrate on data the model was not selected on.** Picking a temperature on
  the same split you report calibration for is the calibration version of
  training on the test set.
- **Count your clusters, not your images.** This is where the real audit ran out
  of road. Its blue-light subset was 51 images — but only ~20 patients,
  about 3 images per bin at 15 bins — or 1.3 patients. We reported a per-mode ECE ranking off that, then
  withdrew it: at that size the number is noise wearing a decimal point.
  **Below roughly 30–40 clusters, stop** — report Brier with a cluster bootstrap,
  say the subgroup is underpowered, resist the ranking. `medaudit` prints the
  group count beside every interval and warns when it is small, because "n=51"
  reads as reassuring and "20 patients" does not. And note what the toolkit
  cannot fix: its percentile cluster bootstrap is itself **anti-conservative** at
  such counts — narrower than the truth. An honest CI is not a correct one.

---

## 5. Audit IV — prevalence, and the metric that matters clinically

Here is the audit that most often explains "it worked in the lab, it failed in
the clinic."

**AUROC is prevalence-invariant.** By construction it does not change when the
disease gets rarer. That is a feature for comparing models — and a trap for
predicting clinical behaviour, because the clinic *feels* prevalence directly.

At a screening prevalence of ~1%, the quantity a clinician lives with is
**positive predictive value (PPV)** [10]: of the cases the model flags, how many
are truly positive? You do not need a citation for why it collapses — just the
arithmetic:

```
              sens · prev
PPV = ─────────────────────────────────
      sens · prev + (1 − spec)(1 − prev)
```

Hold sensitivity at 0.90 and specificity at a strong 0.95. At 50% prevalence,
PPV = **0.95**. At 10% it is already **0.67**. At **1% prevalence the same model
gives PPV = 0.15** — roughly **85% of its flags are false alarms**, and about six
and a half flags for every true positive found. Not one number describing the
model changed. Prevalence did all of it.

This is not a thought experiment — screening challenges now score exactly this.
The **RARE25** challenge at EndoVis/MICCAI 2025, on early Barrett's oesophagus
neoplasia detection, ranked entries by *PPV at 90% recall under a simulated ~1%
prevalence* (neoplasia resampled 1:100, repeated 1000×, scored as the median).
Teams reported AUROCs above 0.9; the winning entry's median bootstrapped
PPV@90%recall was **0.035** [15]. That is not a bad model — it is what a good
model looks like when you finally ask it the question the clinic asks. If you
report only the AUROC, you have not told the reader the number they need.

Two things to do:

- **Report the operating point, not just the curve.** Fix a clinically relevant
  recall; report PPV *at the prevalence you'll deploy at*, not your test split's.
  If the deployment base rate differs from training, the classifier's outputs can
  be corrected for the new prior rather than merely re-thresholded [6].
- **Give it an honest confidence interval.** Use the **cluster bootstrap** —
  resample whole patients, not rows [13; 14, ch. 3]. A per-image bootstrap understates
  uncertainty because one patient's frames move together; it is the same
  independence assumption that group-aware splitting protects in §1, showing up
  again in your error bars.

```python
from medaudit.metrics import cluster_bootstrap, auroc
point, lo, hi = cluster_bootstrap(patient_id, lambda idx: auroc(scores[idx], y[idx]))
print(f"AUROC {point:.3f}  (95% CI {lo:.3f}–{hi:.3f})")
```

When a resampled subgroup holds only one class the statistic is undefined;
`medaudit` drops those resamples and *warns* that the subgroup is underpowered —
a silently-narrow CI being worse than an honestly wide one.

---

## 6. The honest part: findings that died ★

The most valuable output of the real audit was not a green check. It was a series
of headlines we **retracted ourselves.** Four, each with a transferable lesson.

**1. We compared apples to oranges.** Our first alarming result: malignant
sensitivity fell from **0.586 internally to 0.378 externally**. Except the
internal number was computed by `argmax` over five classes and the external one
by thresholding *P > 0.5* — two different decision rules. Measured like for like
(0.570 → 0.378), the unit error accounted for about **8%** of the gap —
but until we fixed it we could not know that, and it is the first thing an honest
reader would have asked. *Lesson: before you explain a gap, check that both sides were
measured the same way.*

**2. "The external collapse proves the shortcut fails on new data."** It doesn't
— because there was no collapse in *discrimination*. Re-measured under one rule
on a matched domain, internal white-light malignant AUROC was **0.796** and
external **0.808**. The ranking ability transferred almost perfectly. What
actually degraded was the **operating point**: with the same P > 0.5 rule,
sensitivity was 0.570 internally and 0.378 externally — a calibration mismatch,
not a discrimination failure. *Lesson: "it got worse externally" is not a
diagnosis. Separate discrimination from calibration before assigning blame — and
be suspicious when the culprit happens to be your favourite hypothesis.*

**3. "The two biases cancel."** We had a satisfying story where the shortcut bias
and a prevalence shift offset each other. It died twice over. First, a
**metric-category error**: we were measuring the effect with **AUROC**, which is
*prevalence-invariant by construction* — a metric that literally cannot see a
prevalence effect cannot be used to argue about one. Second, re-measured with a
prevalence-*sensitive* metric, the effect was **≈ 0**. There was never a signal to
explain. (The prevalence shift was real and large — malignant 9% internally vs
65% externally — which is why it, rather than the shortcut, is the *leading*
explanation for the calibration mismatch in #2. Leading, not established: that
decomposition is itself single-seed and we have not confirmed it across seeds.
Retracting one claim does not entitle you to assert its replacement.)

**4. Hundreds of duplicates that never existed** — the dHash false alarm from §3.

**Why put this in a tutorial?** Because *self-skepticism is the method, not a
disclaimer at the end.* Every one of those four was killed by a discipline that is
cheap to apply and boring to describe: **compare like with like; pick a metric
that can see your claim; get an honest CI; match the instrument to the domain.**
That list is the payload of this whole piece — the same discipline that catches
someone else's over-claim in review. An audit you cannot fail is not an audit.
Reporting the retraction is the most honest thing this project did, and it is the
transferable skill.

---

## 7. Run it yourself

Everything below is the sequence re-run from a clean clone before submitting.

```bash
git clone https://github.com/sunce764/medaudit && cd medaudit
pip install -e .                                    # numpy only — no torch, no sklearn
python tutorial/make_demo.py                        # builds the cohort, prints the reports
medaudit audit --config tutorial/demo/audit.json    # the same audit via the CLI
medaudit audit --config tutorial/demo/leaked.json   # a planted leak, so you see it fire
```

**Be clear what the report is: two audits, not four.** `medaudit audit` runs the
shortcut probe — with cluster-bootstrap intervals — and the leakage audit, which
reports counts and the worst offending pairs rather than an interval, there being
nothing to put one around. Both end in a verdict that states what it does *not*
establish. Calibration (§4) and prevalence (§5) are *not* wired in; you compose
them from `medaudit.metrics`, which is why those sections show library calls
rather than report output. Four audits is the *checklist*; two are automated so
far. Claiming otherwise would be the exact overclaim this piece is about.

To audit your own model, extract penultimate-layer activations to an `(N, D)`
`features.npy` aligned with your manifest rows. Feature extraction is out of scope
by design — the one step needing your framework and GPU, and leaving it out is
what lets the audit run anywhere.

**An audit narrows failure modes; it does not certify.** Every check here has a
blind spot, named where it appears. The job is to move you from "AUROC 0.80,
discrimination transfers, ship it" to an itemised account of *what you checked,
what you found, and what you still don't know* — the account a clinical reviewer,
and a patient, deserve.

From pixels to patients, the last mile is not a higher number. It is the audit
that tells you whether the number means what you hoped.

---

## Code & reproducibility

All code: the `medaudit` repository (MIT). The *scoring* primitives — AUROC,
Brier, and ECE under both binning strategies, including the quantile-binned ECE
this tutorial recommends — are each checked against an independent brute-force
reference written from the definition (`tests/test_metrics.py`). The other two,
`reliability_curve` and `cluster_bootstrap`, are checked only for structural and
reproducibility properties, not against a definitional reference: worth knowing
before you lean on them. The suite is 29 tests across metrics, splits, manifest,
probe, leakage and the orchestrator.

**Reproducing the numbers.** §2's probe blocks and the full audit report
regenerate exactly:

```bash
pip install -e . && python tutorial/make_demo.py
```

The war-story figures in §0/§2/§3/§6 are aggregate results from the internal audit
and do **not** regenerate from this repository — that data cannot be redistributed.
§5's PPV values follow from the formula given there.

No patient data is redistributed and none is needed to follow along: the runnable
examples are synthetic by design, built to mirror the real audit's structure in
miniature. The war stories quote aggregate results from an audit of a classifier
trained on a 2025 public cystoscopy dataset — no images, no rows, no raw data.
(The repository's `.gitignore` refuses images, arrays and weights by default. If
you fork this to audit your own model, keep it that way: the code is what ships.)

---

## AI assistance

The rules ask which tools were used and how, so: an AI coding assistant (Claude,
via Claude Code) was used throughout this work. It assisted the underlying
cystoscopy audit — experiment code, analysis, and adversarial review of my own
conclusions. It drafted this tutorial's prose. It implemented the accompanying
`medaudit` toolkit, generalising my research code to a dataset-agnostic tool. It
checked the citations, and it ran an adversarial review of this draft that caught
errors on both sides — including a claim in §2 that my own records had already
marked unclaimable, and that I had let through.

What is mine: the choice of problem, the direction of the work, the judgments
about what could and could not be claimed, and — the part I would defend hardest
— the decision to retract the four findings in §6 rather than publish the
flattering versions of them. I take responsibility for everything asserted here.

---

## References

[1] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and
F. A. Wichmann. Shortcut learning in deep neural networks. *Nature Machine
Intelligence*, 2(11):665–673, 2020. doi:10.1038/s42256-020-00257-z

[2] A. J. DeGrave, J. D. Janizek, and S.-I. Lee. AI for radiographic COVID-19
detection selects shortcuts over signal. *Nature Machine Intelligence*,
3(7):610–619, 2021. doi:10.1038/s42256-021-00338-7

[3] L. Oakden-Rayner, J. Dunnmon, G. Carneiro, and C. Ré. Hidden stratification
causes clinically meaningful failures in machine learning for medical imaging. In
*Proceedings of the ACM Conference on Health, Inference, and Learning (CHIL)*,
pages 151–159, 2020. doi:10.1145/3368555.3384468

[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern
neural networks. In *Proceedings of the 34th International Conference on Machine
Learning (ICML)*, PMLR 70:1321–1330, 2017.

[5] G. W. Brier. Verification of forecasts expressed in terms of probability.
*Monthly Weather Review*, 78(1):1–3, 1950.
doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

[6] M. Saerens, P. Latinne, and C. Decaestecker. Adjusting the outputs of a
classifier to new a priori probabilities: A simple procedure. *Neural
Computation*, 14(1):21–41, 2002. doi:10.1162/089976602753284446

[7] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang. Distributionally robust
neural networks for group shifts: On the importance of regularization for
worst-case generalization. arXiv:1911.08731, 2019. Published at the 8th
International Conference on Learning Representations (ICLR), 2020.

[8] P. Kirichenko, P. Izmailov, and A. G. Wilson. Last layer re-training is
sufficient for robustness to spurious correlations. In *The Eleventh
International Conference on Learning Representations (ICLR)*, 2023.

[9] M. H. Zweig and G. Campbell. Receiver-operating characteristic (ROC) plots: A
fundamental evaluation tool in clinical medicine. *Clinical Chemistry*,
39(4):561–577, 1993. doi:10.1093/clinchem/39.4.561

[10] D. G. Altman and J. M. Bland. Statistics Notes: Diagnostic tests 2:
Predictive values. *BMJ*, 309(6947):102, 1994. doi:10.1136/bmj.309.6947.102

[11] J. Nixon, M. W. Dusenberry, L. Zhang, G. Jerfel, and D. Tran. Measuring
calibration in deep learning. In *CVPR Workshops*, pages 38–41, 2019.

[12] R. Roelofs, N. Cain, J. Shlens, and M. C. Mozer. Mitigating bias in
calibration error estimation. In *Proceedings of the 25th International
Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR
151:4036–4054, 2022.

[13] C. A. Field and A. H. Welsh. Bootstrapping clustered data. *Journal of the
Royal Statistical Society: Series B*, 69(3):369–390, 2007.
doi:10.1111/j.1467-9868.2007.00593.x

[14] A. C. Davison and D. V. Hinkley. *Bootstrap Methods and their Application*.
Cambridge University Press, 1997. ISBN 0-521-57471-4

[15] T. J. M. Jaspers, F. Caetano, C. H. B. Claessens, C. H. J. Kusters, R. A. H.
van Heslinga, F. Slooter, J. J. Bergman, P. H. N. De With, M. R. Jong, A. J. de
Groof, and F. van der Sommen. Development and evaluation of CADe systems in a
low-prevalence setting: the RARE25 challenge for early Barrett's neoplasia
detection. arXiv:2604.11171. doi:10.48550/arXiv.2604.11171

---
