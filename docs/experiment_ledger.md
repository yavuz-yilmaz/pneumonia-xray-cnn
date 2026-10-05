# Experiment ledger

This ledger retains unsuccessful experiments alongside the current deployment. Each candidate and threshold was selected using validation, but repeated inspection of the legacy benchmark makes subsequent development exploratory. Retaining the incumbent after a candidate loses on test is itself informed by test results; deployment performance is not an unbiased final holdout estimate.

| Experiment | Validation winner | Legacy accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| v1 | resnet18_unweighted | 87.34% | 83.30% | 99.74% | 90.78% |
| v2 | c0p01 | 83.97% | 79.84% | 99.49% | 88.58% |
| v3 primary | resnet18_clahe | 89.58% | 86.03% | 99.49% | 92.27% |
| v3 sensitivity | resnet18_clahe | 86.06% | 81.76% | 100.00% | 89.97% |
| v4 | resnet18_clahe | Incumbent retained | Incumbent retained | Incumbent retained | Incumbent retained |
| v5 | ensemble3 | 89.26% | 85.49% | 99.74% | 92.07% |
| v6 | rsna_then_pediatric | 89.10% | 85.15% | 100.00% | 91.98% |

Current deployment: `models/clean_v3/selected_matched98/best_model.pt`. The original checkpoint is preserved. v7 completed: 384px ResNet18 won validation but reached legacy accuracy 87.98%, precision 84.02%, recall 99.74%, F1 91.21%. It is not deployed. See [v7 report](clean_v7_results.md).

## What is and is not established

- Audited filename-derived patient groups and detected duplicates do not cross the pediatric split. True patient identities are not independently verified.
- The clean incumbent improves all requested point estimates over the original deployed model, but the gains on the pediatric benchmark are modest.
- A separately reserved 1,000-patient RSNA cohort initially provided separate opacity transfer evidence: accuracy/F1 improved, recall decreased. It is now observed and remains reserved. Updated NIH age metadata identifies 60 under-18 acquisitions in this cohort; earlier adult-only descriptions were inaccurate.
- Additional architecture, augmentation, ensemble and adult-pretraining experiments did not establish the requested large improvement across all metrics.
- The old recall was already 99.23%; its absolute headroom is only 0.77 percentage points. A large recall increase on that finite benchmark is mathematically unavailable.
- A new untouched cohort with reliable patient identities is needed for a strong final generalization claim. Pediatric coverage is optional; the user explicitly permits adult development. No result here establishes clinical reliability.

## Next evidence requirement

Current status (2026-10-01): NIH, the MIMIC probe, source-head calibration and
both registered OpenI comparisons have completed. All five external outcomes
passed saved-prediction consistency review. No successful replacement is
established. The [completed MIMIC/OpenI report](mimic_openi_results.md) records
the results and paired uncertainty. OpenI is now observed and remains reserved.

Seven legacy experiment families have not established a large across-metric improvement. Further benchmark-driven recipe searches would increase selection bias. Pediatric access is not a prerequisite. All 112,120 NIH images have now been acquired and decoded; the registered adult development cohort uses patient and duplicate separation and preserves previously exposed RSNA patient roles. NIH v1 completed both candidates, validation selection and its one-time test at 14:42 UTC on 2026-10-01. The MIMIC follow-up had been registered before that test started. These report-derived targets are not equivalent to clinically adjudicated pneumonia, and negative images include other diseases. See [NIH results](nih_v1_results.md).

A separate MIMIC-only frozen-feature probe was registered before NIH v1 test
access. Four fixed logistic heads were trained, and validation average precision
selected C=0.001. The learned head's OpenI accuracy/precision/recall/F1 are
36.86% / 1.46% / 84.38% / 2.87%, versus current deployment's
78.49% / 1.32% / 25.00% / 2.51%. Precision and F1 paired intervals include zero.
Its 1,821 false positives prevent interpreting the recall increase as the
requested across-metric improvement. No candidate was deployed.

All 7,470 official OpenI PNGs were acquired and decoded. The locked external
cohort has 2,892 cases, 32 positives, and 2,983 distinct frontal images.
Independent cohort and saved-output readbacks passed. No NIH/legacy exact or
near match was detected under the fixed conservative screen. Secondary manual
projection labels are not original DICOM-tag verification.

A further [joint-metric threshold policy](nih_joint_metrics_protocol.md) is
registered for NIH validation only. Full implementation registration finished
after primary OpenI inference started; it receives no sixth external comparison.
The original model tensors and five external outcomes are preserved. Its
validation-only calibration completed with no feasible threshold. Independent
baseline metric recomputation and exhaustive threshold enumeration confirmed
the failure. No checkpoint was exported, no floors were relaxed, and no new
external scoring was performed. See the completed report above.

The user confirmed a personal portfolio project and no approved PhysioNet/CITI
access. PadChest-pneumonia metadata was audited, but its images were not acquired
because nonacademic use requires publisher approval. See the
[current access inventory](dataset_access_inventory.md).

## Additional Epic candidate audit — 2026-10-01

The openly licensed Epic Chittagong v5 candidate has working file access after
earlier failed routes. Its two-page PDF passed publisher checksum verification.
The complete ZIP passed publisher checksum, all member CRC and all 2,626 image
decoding checks, while the page/PDF describe 3,355 images. The full screen found
2,545 images matching NIH/legacy (96.9%), 62 cross-partition match pairs, and 48
identical-byte pairs with opposing normal/pneumonia labels. Two prefix examples
were visually confirmed. Only 81 images have no recorded cross-source match;
their patient identity/labels are unverified. The candidate is rejected as new
clean data or an independent hospital test. This is source screening, not a new
model experiment or an independent hospital benchmark.
See the [candidate review](epic_candidate_review.md). Deployment is unchanged.
Separate full readback passed for all 2,626 images and 2,722 recorded match
pairs. The initial numerical-tolerance failure is retained, with its diagnosis
and correction; no scientific screen threshold was changed.

## NIH MIMIC final-block adaptation — completed 2026-10-02

The [fixed development recipe](nih_transfer_protocol.md) adapts DenseNet's final
block and learned NIH classifier while preserving earlier MIMIC-only features
and every BatchNorm running statistic. The initial head hash and exact equality
of the whole starting feature backbone were verified before launch. Training
uses the existing audited NIH patient/duplicate-disjoint train/validation split,
ten epochs and no NIH/OpenI test access. A separate validation threshold must
meet all four metric floors from the two fixed saved validation baselines;
failure is retained without relaxing floors.

The original `models/nih_mimic_tail_v1` run was stopped after epoch one
recorded NaN loss and unchanged validation predictions. A two-training-image
diagnosis reproduced AMP classifier/gradient overflow and verified finite
float32 logits/loss/gradients on identical inputs. The failed run and diagnosis
are preserved in local artifacts.

The corrected source uses float32 training and immediately rejects nonfinite
logits/loss/gradient norm. Twenty-two focused tests passed, including an actual
tiny end-to-end training run guarded against NIH test/OpenI access and a forced
nonfinite-logit failure. Ruff passed. A real NIH training-image GPU forward,
backward and gradient-clipping check also passed with finite values.
The same fixed initial model and recipe were restarted with float32 as a distinct
`models/nih_mimic_tail_v2` run, preserving the failed run. Saved-tensor review
confirmed finite tensors, updates restricted to the allowed tail/head, and
unchanged BatchNorm buffers. Runtime and intermediate readbacks remain in local
artifacts. The corrected ten-epoch run completed. Epoch 9 won with validation AP 0.0229255 versus initial 0.0211108.
At its recall-oriented operating point, accuracy/precision/recall/F1 are
24.87% / 1.30% / 90.32% / 2.56%, including 10,663 false positives. Independent
enumeration of all 14,197 distinct thresholds found zero meeting all four
baseline metric floors. Final tensor/selection/metric/hash reviews passed;
post-run original-image hashes also passed for all train/validation images.
No test images were opened. This is a failed replacement candidate, not the
requested across-metric improvement. See [final results](nih_transfer_results.md).
Deployment and observed external cohorts remain unchanged.

## Additional public source acquisition — 2026-10-02

BDCXR-3257's complete ZIP passed publisher checksum, CRC and all 3,257 decodes.
The fixed screen found no NIH/legacy/OpenI overlap and one internal pair, but
patient identifiers are absent. Bundled academic-only terms conflict with the
public CC BY record; authority/scope remains unresolved for a personal portfolio.
Do not train, score or distribute it pending clarification. No provider contact
was made. Private acquisition and document-review evidence is preserved.

SSMU pneumonia/norm (Zenodo 5732746, CC BY 4.0) completed exact-range recovery
of its 4.185 GB archive at 11:11 UTC, with matching whole-file publisher MD5.
The registered audit completed at 11:15 UTC: gzip CRC and all 1,837 PNG decodes
passed. Source directories contain 1,035 pneumonia and 802 norma images;
934 filenames indicate frontal views and 903 lateral views. The fixed screen
found zero NIH/legacy/OpenI matches and 11 internal pairs (five exact bytes,
six near), without class-directory conflicts. Source/reference hashes were
pinned and verified. No split, extraction, model training or scoring occurred.

A separate inventory readback verified artifact hashes and documented the
initial parser's omission of `norma` from its normal-directory aliases. Six
16-bit lateral images were clipped by the original L conversion; their screen
is limited, none of the recorded pairs involves them, and all 934 frontal
filename-hint images are original 8-bit. No patient/age/clinical label metadata
exists in the archive. 401 numeric stems occur in both class directories; neither
stems nor directory/stem pairs are verified patient identities. The new source
is an accessible candidate, not yet a patient-independent model cohort.
Evidence: `reports/data_access_20261002/ssmu_inventory_readback_v1.json` and
`data/processed/ssmu_images_v1/image_audit.json`.

The MultiCaRe thoracic subset ZIP was acquired and CRC verified; it contains
tables, no images. Its 713/4,539 pneumonia one/zero counts are intermediate
pre-negation NLP targets, not verified positive/normal truth. Original publisher
metadata was acquired; per-image licenses and image/case/time linkage still
need audit. See [candidate details](additional_dataset_candidates_20261002.md).
Further screening verified unique file/patient caption linkage and mixed
per-article licenses. A phrase screen flagged 29 positive rows for possible
negation/temporality review, not automatic label corrections. The dictionary's
patient IDs are article/case identifiers, not verified cross-article identities.

## SSMU frontal model development and test — completed 2026-10-02

The [fixed development protocol](ssmu_development_protocol.md) was registered
before source-specific model scoring. It joins numeric stems across both classes
and views, plus all eleven recorded duplicate pairs including lateral pairs.
After excluding laterals and five redundant frontal images, 929 frontal inputs
remain: 665 train / 133 validation / 131 reserved test. The seven-fold seed/rule
was fixed in advance. Audit found zero byte/pixel/numeric-proxy/component overlap
or recorded cross-split near pairs. This is proxy separation, not verified patient
independence; unknown identities/clinical labels remain unresolved.

Preparation completed and all selected images matched the source fingerprints.
Its executed source snapshot is retained in
`reports/data_access_20261002/ssmu_preparation_source_snapshot.py` (SHA256
`2cdd9678f67070dba8c55df60b93331ddbec64864e256d029c1b179e05825eb9`).
The checkout source was subsequently reformatted without changing the data rule.
Six split tests passed; after probe implementation, sixteen combined split/probe/
joint-threshold tests passed, including actual logistic fitting/export and a
test-manifest-free end-to-end fixture retaining infeasible floors. Ruff passed.
Original train/validation source-byte verification passed again before launch.

The first classifier experiment uses hash-verified MIMIC-only frozen DenseNet
features and four fixed logistic heads. Scaler fitting uses training only;
validation AP selects the head, followed by the joint four-metric threshold
requirement against original/current validation predictions at saved thresholds.
Run: `models/ssmu_probe_v1`.
The run completed with C=0.01 selected by validation AP 0.9587472. The joint
threshold 0.7265916466712952 was feasible; validation accuracy/precision/recall/F1
are 79.70% / 100% / 63.51% / 77.69%. No test access occurred during training.

A separately [registered one-time test](ssmu_test_protocol.md) froze tensors,
thresholds, original/current baselines, evaluator hashes and the 131-image test
manifest before inference. Nine evaluation/paired-comparison tests and Ruff
passed before launch. The evaluator completed at 11:43 UTC. New test metrics
are **86.26% / 95.00% / 79.17% / 86.36%**, versus current
48.85% / 69.23% / 12.50% / 21.18% and original
64.89% / 90.63% / 40.28% / 55.77%. All four point estimates improve, but paired
precision-difference intervals include zero. Accuracy, recall, F1, ROC-AUC and
AP difference intervals are positive versus both baselines. Baseline thresholds
remain their original operating points; new-model calibration is source-specific.

Independent saved-output review passed without new image inference: prediction/
label/group/member alignment, source/checkpoint hashes, frozen export tensor
equality, thresholds and independent confusion/primary-metric recomputation.
The selected checkpoint is `models/ssmu_probe_v1/selected_joint/best_model.pt`,
SHA256 `6e8202040581f8e8412fea206f66e1eb748471e0161fff3d4cc56396b237fbc6`.
No deployment change occurred. SSMU test is now observed and remains reserved;
it cannot be used as a fresh final test after further tuning. Patient identity
and clinical labels remain unverified, and the full goal is not established.
See the [completed source-specific results](ssmu_model_results.md).

## SSMU identity investigation and exploratory transfer check — 2026-10-02

All 929 selected files had empty PNG/EXIF metadata after source-byte checks.
Six training-only visual samples did not reveal apparent identifying text, but
this does not establish absence across all images. Anonymous grouping and
clinical-label provenance remain unverified; the [publisher draft](ssmu_identity_questions.md)
is unsent.

A retry of SSMU 2026 part 1 publisher metadata succeeded: 99 PNGs (50 frontal
filename hints, 49 lateral), CC BY 4.0, collected 2024–2025. The record supplies
only pneumonia images and no patient table in its file list. No new images
were acquired or scored; it is not a complete binary evaluation cohort.

The frozen selected SSMU model was then scored once on the previously observed
OpenI cohort under a [separate exploratory protocol](ssmu_openi_check_protocol.md).
Original/current saved probabilities were reused after hash/order/aggregation
and metric checks. All cohort bytes, source hashes and fixed MIMIC-only tensors
passed verification before inference; the new registration precedes scoring.
New accuracy/precision/recall/F1: **47.61% / 1.44% / 68.75% / 2.82%**.
The model produces 1,505 false positives among 2,860 negative report cases.
This does not establish generalization or an across-metric improvement.
Twenty-five focused tests and Ruff passed. Evidence:
`reports/metrics/ssmu_openi_check_v1/`; earlier studies remain unchanged.
No model/threshold tuning, deployment, provider contact or publication occurred.

## NIH / RSNA metadata target review — 2026-10-02

Pinned development manifests and official updated NIH/RSNA metadata were
audited without test-manifest access or image inference. Original patient-ID
linkage and train/validation separation passed. NIH training has 700 pneumonia,
36,894 No Finding and 28,068 other-finding negatives; validation has 155/7,897/
6,160 respectively. Removing other-disease examples would change the task and
population, not establish the requested general improvement.

Among 96 NIH validation pneumonia images linked to final RSNA labels, 27 have
Lung Opacity, 22 Normal and 47 No Lung Opacity / Not Normal. Another 608 NIH
validation negatives have RSNA opacity. These different targets do not yield
an annotation-error count or an automatic pneumonia relabeling rule. An
independent pandas readback confirmed source hashes, NIH counts, labels,
patient separation and cross-table totals. See the
[target review](nih_rsna_target_review.md). No training, split changes or tuning
were initiated from this audit. Verified patient grouping/new evaluation data
remain the external dependency for proving the complete goal.

## SSMU public provider contact verified — 2026-10-02

A focused public search led to Digital Diagnostics article 633978 (2025).
Its [author section](https://jdigitaldiagnostics.com/DD/article/view/633978)
lists Vladimir D. Udodov's public contact route, SSMU affiliation and
ORCID `0000-0002-1321-7861`.
The ORCID matches both pinned SSMU dataset publisher records (2021 and 2026).
The [unsent clarification draft](ssmu_identity_questions.md) now names this
specific recipient and the source. No message or image was transmitted;
mailbox availability, provider response and patient grouping remain unverified.
