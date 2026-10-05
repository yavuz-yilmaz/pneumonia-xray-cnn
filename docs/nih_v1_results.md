# NIH v1 results — 2026-10-01

## Outcome

The registered NIH experiment completed. It **did not establish the requested
improvement across accuracy, precision, recall and F1**. The validation-selected
EfficientNetV2-S detects more positive report labels than the two preserved
classifiers, but produces far more false positives and worse accuracy,
precision and F1. It is not deployed.

These are adult NIH **report-pneumonia** results, not a new measurement of the
legacy pediatric NORMAL/PNEUMONIA task. No reported pneumonia includes other
diseases and does not mean clinically normal. Scores from these two tasks cannot
be compared as a before/after improvement.

## Data and selection

All 112,120 official NIH images were acquired and decoded. The fixed split uses
original patient IDs connected by detected duplicate images, retains historical
RSNA patient roles, and excludes previously exposed patients from the fresh test.
Audited patient, byte-hash, pixel-hash and connected-group overlap across splits
is zero; the registered conservative near-duplicate screen found zero cross-split
matches. This is not exhaustive proof of every possible overlap or label error.

| Partition | Images | Report-pneumonia positives |
|---|---:|---:|
| Training | 65,662 | 700 |
| Validation | 14,212 | 155 |
| Test | 12,158 | 118 |

ResNet18 and EfficientNetV2-S were selected by validation average precision with
validation loss as tie-breaker. The selected EfficientNet validation average
precision was 0.031215, ROC-AUC 0.680657. Its final float32 validation threshold
was 0.00604498665779829, selected by F1 subject to recall >=90%. Final selection
recomputed validation in float32; its metrics differ slightly from AMP training
history. Both records are preserved. Test labels were not used for selection.

## One-time test comparison

All three models were scored on the same 12,158 images at thresholds fixed before
test access. Percentages below retain the source cohort's 0.9706% positive rate.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| NIH-selected EfficientNetV2-S | 21.13% | 1.15% | 94.07% | 2.26% | 0.6987 | 0.0233 |
| Current deployed classifier | 61.62% | 1.62% | 64.41% | 3.15% | 0.6538 | 0.0155 |
| Original classifier | 49.31% | 1.34% | 70.34% | 2.62% | 0.6333 | 0.0149 |

The NIH-selected model produced 111 true positives, 7 false negatives, 9,582
false positives and 2,458 true negatives. An always-negative rule would reach
99.03% accuracy but zero recall and F1; accuracy alone is particularly misleading
at this prevalence.

Paired patient/duplicate-cluster bootstrap intervals (1,000 repetitions) show
that selected-minus-current differences in accuracy, precision and F1 are
negative, while recall is positive. ROC-AUC and average-precision difference
intervals include zero. The evidence does not establish a successful replacement.

## Artifact review and follow-up

The coordinator completed at 14:42 UTC. A separate saved-artifact review passed
at 14:51 UTC: manifest and checkpoint hashes, label/group order, independently
recomputed confusion matrices and metrics, validation selection rule, exact
selected/winner model tensor equality and evaluator source hashes. The review
performed no new image inference or threshold fitting. Its first restricted
checkpoint read rejected local TorchVersion metadata; explicitly allowing that
known PyTorch class resolved the read while retaining restricted loading.

Evidence is retained locally in `reports/metrics/nih_v1/test_metrics.json`,
`models/nih_v1/selected/selection.json` and
`reports/metrics/nih_run_v1/final_artifact_review.json`.

The MIMIC-only frozen-feature probe and the OpenI comparison were registered
before NIH test observation. A separate original MIMIC pneumonia-head comparator
was registered before any OpenI scoring. These completed without the requested
across-metric improvement; see [MIMIC/OpenI results](mimic_openi_results.md).
Their original recipes and outcomes are preserved. The deployed checkpoint
remains `models/clean_v3/selected_matched98/best_model.pt`.

On 2026-10-02 a separate [final-block adaptation](nih_transfer_protocol.md)
experiment started using NIH training/validation only. This is exploratory
development after NIH/OpenI test observation, not a newly untouched evaluation.
The first attempt was stopped for nonfinite AMP training loss, with failed
artifacts preserved. The corrected float32 run completed all ten epochs; its
epoch-9 checkpoint raised validation AP slightly to 0.0229255 but met no joint
four-metric threshold. Final artifact and post-run train/validation source-hash
reviews passed. No new test or external inference occurred. See
[final adaptation results](nih_transfer_results.md); deployment is unchanged.
