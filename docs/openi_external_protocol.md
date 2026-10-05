# Prospective OpenI external comparison

Registered after metadata screening and acquisition-prefix checks, before any
OpenI model inference. This document fixes an evaluation recipe; it does not
establish that acquisition, overlap audits or candidate training have finished.
NIH v1 test scores have not been read when this recipe is registered.

## Target and cohort

- Use the previously screened `openi_candidate_v1/metadata_candidates.json`
  (SHA256 `c919e0b57ce61877923ca91a9d7d6752e26ad757cd9772aef59203252bbaa149`).
  Require `metadata_eligible: true`. Positive means manual report indexing of
  pneumonia; negative means no coded or mentioned pneumonia. Neither establishes
  clinically confirmed disease or clinical normality. Preserve every original
  exclusion reason and disclose the screened population.
- Require the official PNG archive's complete gzip CRC and image decoding audits.
  Require unique linkage to official reports and verified stored image hashes.
- Use only images mapped as Frontal by the secondary manually classified
  `indiana_projections.csv` (SHA256
  `7af9c8a6b4f8ae695c654ecd4208edd193a37892a5607b7cb9e609dc997e6b7a`).
  Join case UID and full image suffix, with exactly one match. Do not infer views
  from filename order. These are secondary manual view labels, not verified
  original DICOM tags. Unmapped/lateral images are not scored.
- Exclude an entire case if any of its images matches NIH or the legacy source
  inventory by bytes, decoded grayscale pixels, or the registered conservative
  near-duplicate screen. This includes references outside current training roles.
  Exclude both cases for any cross-case OpenI exact/near match, even if the
  duplicated image is lateral. Retain a case's unique decoded frontal images.
- Record all missing/unmapped images and exclusions. Require at least one linked,
  decoded, mapped, unique frontal image per retained case and both target classes.
  No exclusions may depend on model outputs. Initial metadata counts allow at
  most 32 positive and 2,860 negative frontal cases; final counts may be lower.
- The source publication describes one study per patient. Use case ID as the
  evaluation unit; independent access to original patient identities is absent.
  No age restriction is introduced when reliable age metadata is unavailable.

## Fixed model comparison

1. MIMIC-only frozen-feature candidate: the completed registered
   `models/nih_mimic_v1/frozen/result.json` must select its head by NIH validation
   AP, breaking ties with lowest C. Require the pinned MIMIC source and tensor
   hashes, frozen-feature training, NIH train/validation manifest hashes and
   `test_access: none`. Mixed `all` XRV weights are forbidden because they include
   OpenI. No replacement model is selected after observing OpenI outcomes.
2. NIH v1 ImageNet candidate: the original registered coordinator's validation
   selection, `models/nih_v1/selected/best_model.pt`. Its NIH test outcome is
   disclosed but cannot change this candidate's inclusion.
3. Current deployed comparator: `models/clean_v3/selected_matched98/best_model.pt`,
   SHA256 `da5c58d36865cc01d8df1dce5c358dac2a06b0b0b4981115bb7e35d806e2be9b`.
4. Original comparator: `models/best_model.pt`; pin its exact hash in the local
   registration before inference. Require its original .70 operating threshold.

All models must finish and have compatible binary pneumonia output semantics.
The pediatric comparators retain their original source-task semantics; this
comparison measures transfer to manually indexed OpenI report pneumonia.
Require exact model construction, preprocessing, normalization, full-precision
evaluation mode and each checkpoint's already selected decision threshold.
Average each model's probabilities across a case's unique frontal images and
apply its fixed threshold once to that case mean. Report aggregation explicitly:
these thresholds were originally selected at image level on source validation.
Do not search or recalibrate thresholds on OpenI.

## Lock, score once and report

Before inference, save an exclusive registration containing this document's
hash, final cohort/image hashes, exclusions, all four checkpoint hashes,
thresholds, preprocessing metadata, training/provenance records and evaluator
source hashes. Refuse changed inputs, missing prerequisite results, mixed-source
pretraining, or an existing evaluation marker. Record started/completed/failed
states; a failed run is preserved for explicit diagnosis rather than silently
restarting. Never modify the frozen NIH v1 evaluator to admit this comparison.

Score the same locked cases once with all four models. Report case-level
accuracy, precision, recall, F1, specificity, balanced accuracy, ROC-AUC, average
precision, confusion matrices and exact positive/negative case counts. Use 1,000
paired case bootstrap repetitions with seed 20261001 for 95% percentile
intervals and MIMIC candidate minus each comparator differences. Record valid
replicate counts; do not replace undefined AUC/precision with invented evidence.
Preserve all model outcomes, including failures and negative comparisons.

A gain in accuracy alone can reflect the predominantly negative cohort. Neither
passing pipeline checks nor improving one metric establishes the user's requested
large improvement across accuracy, precision, recall and F1. The small positive
case count, uncertain report targets, secondary view labels, duplicate-screen
limits and prior source-cohort exposure must accompany results. Do not deploy
automatically or describe this evaluation as proof of clinical reliability.

This external source is reserved for final comparison and is not used for
training, candidate/threshold selection, error-driven tuning or subsequent recipe
searches after results are observed. The data's CC BY-NC-ND 4.0 terms also remain
separate from the repository code license.
