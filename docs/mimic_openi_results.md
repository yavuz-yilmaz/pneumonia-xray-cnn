# MIMIC transfer and OpenI external results

## Outcome

The completed experiments **do not establish substantial improvement across
accuracy, precision, recall and F1**. Higher recall comes with many false
positives. The deployment remains `models/clean_v3/selected_matched98/best_model.pt`.
All unsuccessful outcomes are retained. These are research classifiers, and the
results do not establish clinical reliability.

## Development and selection

Public author weights from a **MIMIC-only** TorchXRayVision DenseNet were used;
MIMIC raw images were not downloaded. Mixed-source weights were excluded to
avoid known NIH/Indiana pretraining overlap. The frozen backbone and train-only
feature scaler supported four fixed logistic heads on NIH training data, with
C in {0.001, 0.01, 0.1, 1}. Validation average precision selected C=0.001.
Threshold selection maximized validation F1 subject to recall >=90%.

On 14,212 NIH validation images (155 report-pneumonia positives), this head had
accuracy 25.91%, precision 1.31%, recall 90.32%, F1 2.59%, ROC-AUC 0.6471 and
average precision 0.0211, at threshold 0.005762157496064901.
The separately registered original MIMIC pneumonia head was unchanged; NIH
validation calibrated only its threshold, 0.006846125237643719. It had validation
accuracy 20.08%, precision 1.23%, recall 90.97% and F1 2.42%.

The [probe protocol](nih_mimic_probe_protocol.md) and
[source-head protocol](nih_mimic_source_head_protocol.md) specify these separate
procedures. Neither used NIH test for fitting or threshold selection.

## Fixed external comparison

The [primary comparison](openi_external_protocol.md) and fifth source-head
comparator were registered before OpenI scoring. The locked cohort contains
2,892 cases, 32 positive and 2,860 negative report-code cases, represented by
2,983 unique frontal images. Each case has equal weight; probabilities from
multiple distinct frontal images are averaged. Thresholds were carried from
source development and were not recalibrated on OpenI.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Current deployment | 78.49% | 1.32% | 25.00% | 2.51% | 0.6031 | 0.0154 |
| Original deployment | 64.28% | 1.36% | 43.75% | 2.64% | 0.5724 | 0.0169 |
| MIMIC learned head | 36.86% | 1.46% | 84.38% | 2.87% | 0.6869 | 0.0378 |
| NIH EfficientNet | 28.53% | 1.43% | 93.75% | 2.82% | 0.7223 | 0.0622 |
| MIMIC original head | 14.11% | 1.27% | 100.00% | 2.51% | 0.7750 | 0.0458 |

The learned MIMIC head detects 27 of 32 positives but also produces 1,821 false
positives among 2,860 negative cases. Current deployment detects 8 positives
with 598 false positives. The original MIMIC head detects all 32 positives but
produces 2,484 false positives. Recall alone would misrepresent these outcomes.
Positive prevalence is only 1.11%; predicting every case negative would give
98.89% accuracy and zero recall. Accuracy alone is also inadequate.

### Paired uncertainty

The registered bootstrap used 1,000 case-level resamples and seed 20261001.
For the learned MIMIC head minus current deployment, the 95% intervals are:

| Metric difference | 95% interval |
|---|---:|
| Accuracy | -43.57 to -39.63 percentage points |
| Precision | -0.70 to +0.95 percentage points |
| Recall | +39.98 to +77.79 percentage points |
| F1 | -1.21 to +1.92 percentage points |
| ROC-AUC | -0.0492 to +0.2034 |
| Average precision | +0.0010 to +0.0739 |

Precision and F1 intervals include zero. For the original MIMIC head versus
current deployment, ROC-AUC and average precision intervals are positive
([0.0600, 0.2704] and [0.0050, 0.1013]), but this does not compensate for poor
operating-point accuracy, precision and F1. Only 32 positive cases support the
external estimates. These multiple comparisons are descriptive evidence.

## Leakage checks and limitations

NIH development partitions separate recorded patients and detected byte/pixel
duplicates and registered near matches. The OpenI screen found no NIH/legacy
exact or near match under its conservative rule. This is an audit result,
not proof of every possible patient or acquisition overlap.
OpenI labels are manually coded report findings, and a negative case is not
necessarily clinically normal. NIH labels are report-derived and negative images
include other diseases. Secondary manual projection labels are not verified
original DICOM tags. No OpenI age filter was applied without reliable metadata.
The old pediatric benchmark and these report cohorts have different populations,
labels and prevalence; their accuracies cannot be compared as a numerical gain.

OpenI and the NIH one-time test are now observed and remain reserved. They must
not be used to tune a new recipe and then presented as fresh final evaluation.
A later [joint-metric policy](nih_joint_metrics_protocol.md) uses NIH validation
only. Its full registration finished after primary OpenI inference began, so
it receives no additional OpenI comparison and needs a new independent cohort.

## Verification and retained evidence

### Completed joint-metric validation policy

The additional validation-only calibration completed with
`no_feasible_validation_threshold`. The fixed requirements, taking the larger
corresponding NIH validation metric of the original and current baselines, were
accuracy >=62.8413%, precision >=1.4718%, recall >=69.0323% and F1 >=2.8822%,
with at least one strict improvement. No unique candidate score met them all.
No checkpoint was exported, the requirements were not relaxed, and no additional
NIH test or OpenI scoring was performed. This only rules out the specified
operating policies for this fixed learned head; it does not establish that all
future models or data sources must fail.

An independent saved-validation review recomputed both baselines' metrics,
verified labels and registered hashes, and exhaustively counted predictions at
each unique threshold without using the production cumulative selector. It
confirmed zero feasible thresholds and absence of an exported checkpoint.

### Original model and external reviews

Saved-artifact reviews passed validation label alignment, all four frozen
backbones' exact equality to the approved source, exported head reproduction of
saved probabilities, independent metrics and fixed validation selection.
The external review independently checked all five models' case IDs/labels,
image-to-case means, confusion matrices, metrics and hashes. It did not rerun
image inference, fitting, threshold selection or bootstrap sampling.

Local ignored evidence:

- `models/nih_mimic_v1/frozen/result.json`
- `models/nih_mimic_v1/source_head/result.json`
- `reports/metrics/nih_mimic_run_v1/validation_artifact_review.json`
- `reports/metrics/openi_external_v1/result.json`
- `reports/metrics/openi_source_head_v1/result.json`
- `reports/metrics/openi_source_comparison_run_v1/artifact_review.json`
- `models/nih_mimic_v1/joint_policy/result.json`
- `reports/metrics/joint_metrics_run_v1/artifact_review.json`

See the [NIH results](nih_v1_results.md) and
[access inventory](dataset_access_inventory.md) for other completed evidence.
