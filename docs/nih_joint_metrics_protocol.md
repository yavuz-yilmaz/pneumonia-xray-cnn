# Secondary validation policy for joint metric improvement

## Actual registration scope

The rule below was written at 15:29 UTC on 2026-10-01 while primary OpenI inference
had not started. Implementation/testing finished after primary inference began
at 15:33 UTC. The intended full registration gate was not met. Consequently this
follow-up is **NIH validation only**, registered after primary scoring started,
and receives no sixth OpenI evaluation. Original OpenI outcomes are preserved.
The original intended external contract below is retained as a timing requirement,
not as a claim that this follow-up satisfied it. No OpenI scores were inspected to
set its rule or threshold. A new independent cohort would be needed to establish
external performance for any exported policy.

Register this policy and its implementation before any OpenI scoring. The user's
objective names accuracy, precision, recall and F1 together; a fixed 90% recall
constraint is one operating policy, not evidence that these four improve jointly.
This additional policy preserves the original four-model comparison and fixed
source-head comparator. No existing candidate, threshold or protocol is changed.

## Fixed validation procedure

- Use only the primary MIMIC frozen-feature model already selected by NIH
  validation average precision (ties lower C). Do not search other architectures,
  C values, source heads, preprocessing or feature states for this policy.
- Reuse its saved NIH validation probabilities and verify labels against the
  immutable NIH validation manifest. Preserve its original checkpoint unchanged.
- Score the original and current deployment checkpoints on exactly that NIH
  validation manifest, using their own fixed source thresholds and preprocessing.
  Verify their previously registered hashes. No NIH test or OpenI image, label,
  score or result is used for this procedure. The previously observed NIH test
  remains disclosed and cannot serve as a fresh test of this follow-up.
- Consider unique observed candidate validation probabilities as thresholds,
  using inclusive probability >= threshold classification. A threshold is
  feasible only if accuracy, precision, recall and F1 are each >= the greater
  corresponding validation metric of the two baselines, and at least one is
  strictly greater. Use tolerance 1e-12 only for floating-point comparisons.
- Among feasible thresholds maximize F1, then accuracy, then threshold. If none
  is feasible, report `no_feasible_validation_threshold` and export no candidate.
  Do not relax floors, change the feature/head selection, or create a fallback.
- Export the exact primary model tensors with only this separately recorded
  decision policy and validation threshold changed. Preserve all original
  artifacts. This is threshold calibration, not an additional head fit. A
  feasible validation threshold is not proof of improvement on unseen cases.

## Execution and independent comparison

Queue validation calibration after successful completion of the already
registered primary and source-head comparisons, to avoid GPU overlap. The recipe
and source/manifest hashes must be recorded before primary OpenI inference;
calibration itself does not open external outputs. Do not inspect OpenI scores
to decide this policy or its threshold. It is a prespecified secondary analysis.

If calibration is feasible, compare it on the same locked OpenI cases by reusing
the primary MIMIC model's saved probabilities, never repeating image inference.
Preserve all five original comparator outcomes and add the sixth operating
policy. Report paired case-bootstrap differences against every original
comparator with 1,000 draws and seed 20261001. The secondary evaluator and its
exact source/cohort hashes must also be registered before primary scoring.
If implementation/registration is not ready before external inference starts,
do not present this policy as a prospectively registered external comparison.

No OpenI-driven recalibration, candidate replacement or automatic deployment is
permitted. Report weak/failed results and the 32-positive-case uncertainty, report
label limitations, view provenance and conservative overlap-screen limitations.
Even a point improvement in all four metrics does not by itself establish the
requested substantial improvement or clinical reliability.
