# MIMIC final-block adaptation on NIH — exploratory development

## Reason and limits

The frozen MIMIC feature probe did not improve all four requested metrics. Its
feature extractor was never adapted to NIH. This single follow-up trains the
final DenseNet block at a small learning rate, preserving earlier features and
all BatchNorm population statistics. There are only 700 NIH training positives;
full unrestricted fine-tuning would risk overfitting these repeated examples.

NIH test and OpenI outcomes have already been observed. This run cannot establish
a fresh external success claim and will not read their manifests, images or
scores. Patient/duplicate-separated NIH train/validation membership stays fixed.
The task remains report-derived pneumonia, including other diseases among
negatives. This is not clinically adjudicated pneumonia. Development improvement
alone does not satisfy the user's requested final outcome.

## Numerical correction and preserved first attempt

The first run (`models/nih_mimic_tail_v1`) was stopped after its first epoch
reported NaN training loss and unchanged validation results. A two-training-image
diagnosis reproduced nonfinite AMP classifier logits and gradients. The same
images/model in float32 produced finite logits, loss (1.9133) and every gradient.
The failed run, original source/protocol snapshots and precision diagnosis are
preserved under `reports/metrics/nih_mimic_tail_v1_failure`.

The corrected run uses float32 and stops immediately on nonfinite logits, loss
or gradient norm. It restarts from the same pinned initial checkpoint in a new
immutable directory, `models/nih_mimic_tail_v2`; it does not resume the failed
attempt or change the scientific recipe in response to validation performance.

## Fixed corrected experiment

- Verified MIMIC-only DenseNet121 feature artifact; no mixed NIH/OpenI weights.
- Initialize the head from the fixed C=0.001 NIH frozen probe. Verify checkpoint
  SHA256 and exact equality of every backbone tensor to the MIMIC artifact.
- Train denseblock4 and classifier only. Every BatchNorm module stays in eval
  mode; all running statistics and earlier parameters must remain unchanged.
- Seed 20261002. Ten epochs, 12,000 class-balanced samples per epoch with
  replacement, unweighted cross entropy. AdamW feature LR 1e-5, head LR 1e-4,
  weight decay 1e-4; batch 16; float32 training/validation, gradient clip 5.
- Existing 224px resize cache, XRV normalization, existing mild acquisition
  augmentation. Natural validation prevalence; no oversampling in validation.
- Choose the checkpoint by validation average precision, earliest epoch on ties.
  Retain recall >=90% metrics as a comparison to prior recipes.
- After checkpoint selection, compute a separate threshold satisfying all four
  accuracy/precision/recall/F1 floors of both original/current classifiers on the
  same validation manifest. Reuse pinned saved baseline metrics. Maximize F1,
  then accuracy, then threshold. If infeasible, retain the failure explicitly;
  do not relax floors or select a different checkpoint after seeing that result.
- Preserve checkpoint, scores, history, input hashes, process status and failures.
  No automatic deployment or additional NIH/OpenI test scoring.

```powershell
python -m src.training.train_nih_transfer
```

The existing ImageNet-only training/evaluation guards remain intact. A new
reliable independent evaluation cohort remains necessary for a final claim.
