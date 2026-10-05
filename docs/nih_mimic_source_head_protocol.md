# Secondary MIMIC pneumonia-head transfer

Register this bounded follow-up before observing any OpenI model scores. The
ImageNet candidates' weak NIH validation precision/F1 motivate checking the
public MIMIC model's existing pneumonia classifier rather than discarding it.
This is an additional, fully disclosed source-transfer experiment; it does not
alter the registered four-model OpenI comparison or its candidate selection.

## Fixed source and calibration

- Use only the already reviewed MIMIC/CheXpert-labeler source checkpoint, SHA256
  `23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189`.
  Its stored `targets` and the author registry both identify row 8 as Pneumonia.
  The copied feature state exactly equals the approved MIMIC-only tensor artifact.
- The separately saved classifier-row artifact has SHA256
  `2381ed671a42a3c02418752b332660c37ca6c1f8ae686f715e63a9e67073b94e`.
  Load it with `weights_only=True`, require finite 1024-element weights and one
  scalar bias, and verify its pathology/source metadata. Features remain pinned
  to SHA256 `7c5df8cf35f00b01d2b90e77f4301027c9d1555395a0c15512aaf2947973e7c0`.
- Keep the original pneumonia logit. A two-logit head `[0, pneumonia_logit]`
  gives the original raw sigmoid probability. Do not apply XRV's optional
  operating-point normalization. Synthetic-feature parity was checked against
  the original classifier. This is not evidence of classification performance.
- Freeze all features and classifier parameters. No fitting on NIH training
  images/features, no alternative pathology row, no C grid, no architecture or
  preprocessing search. This classifier was trained by the source authors on
  MIMIC; local work exports it and calibrates its threshold, not a new fit.
- After the registered MIMIC probe completes successfully, reuse its ordered,
  unstandardized NIH validation features. Verify their labels against the fixed
  NIH validation manifest. Do not load train feature arrays or NIH test data.
- Inputs and pooling exactly match the registered MIMIC probe: resize 224,
  repeated grayscale RGB, normalization mean .5 / standard deviation 1/2048,
  frozen MIMIC DenseNet121 in evaluation mode. No scaler is used for this source
  head. Select its single threshold by validation F1 subject to recall >=90%.
- Export an immutable checkpoint, complete source metadata, feature/manifests
  hashes, validation predictions and all validation metrics. Keep NIH report
  label names explicit; negative does not mean clinically normal. Preserve weak
  or unsuccessful results. No automatic deployment.

## Secondary external evaluation contract

The primary four-model comparison remains unchanged and runs first. The source
head is an additional prospectively specified comparator on the exact same
locked OpenI cases, with its NIH validation threshold fixed. It must not replace
the primary learned MIMIC head based on OpenI outcomes. Any secondary evaluator
must pin this recipe, source/checkpoint/cohort hashes before inference and score
the source head once; reuse the primary models' saved aligned case predictions
instead of scoring those models again. Report all five models and paired case
bootstrap differences (1,000 repetitions, seed 20261001), including failed or
negative comparisons. No source-head threshold/candidate decisions may depend on
the primary OpenI scores, and no subsequent OpenI tuning is permitted.

The local export/calibration command does not open OpenI or NIH test images,
manifests or predictions. External evaluation implementation/registration is a
separate prerequisite; successful export is not proof of improvement. Publisher
source identity is not independent inspection of every pretraining patient, and
the OpenI cohort still has only 32 positive report-code cases.
