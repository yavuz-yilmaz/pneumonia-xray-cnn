# NIH pneumonia with MIMIC-pretrained frozen features

## Scope

Register this follow-up before reading any NIH v1 test results. The poor NIH v1
validation results motivate this bounded comparison; they are not final test
evidence. Preserve the running ImageNet experiment and its frozen implementation.
This command trains and selects on NIH train/validation only. It never opens NIH
test manifests/images/predictions or OpenI images/model scores. A separate final
evaluation contract and source/duplicate audit are still required before claiming
the user's requested improvement. Do not automatically deploy this checkpoint.

## Fixed recipe

- Reuse the finalized NIH patient/duplicate-disjoint manifests, with historical
  patient roles preserved. The label mapping remains `NO_REPORTED_PNEUMONIA: 0`,
  `REPORT_PNEUMONIA: 1`; other diseases are included among negatives.
- Initialize only from the author-designated MIMIC-CXR/CheXpert-labeler DenseNet121
  features. Do not use mixed `all` weights, which include NIH and OpenI. Original
  public weight SHA256: `23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189`.
  Verified tensor-only feature artifact SHA256:
  `7c5df8cf35f00b01d2b90e77f4301027c9d1555395a0c15512aaf2947973e7c0`.
  Publisher-designated provenance is not independent access to all training IDs.
- Discard the original classifier. Freeze all backbone parameters and BatchNorm
  buffers in evaluation mode. Extract each development image once, without
  balancing, stochastic augmentation or repeated positive oversampling.
- Resize to 224 pixels without CLAHE. Normalize repeated grayscale RGB with
  mean `(0.5, 0.5, 0.5)` and standard deviation `(1/2048, 1/2048, 1/2048)`.
- Fit feature standardization on training features only. Fit unweighted L2
  logistic heads with fixed `C = 0.001, 0.01, 0.1, 1.0`, `lbfgs`, maximum 2,000
  iterations. Refuse nonconverged fits. Seed 42. Fold standardization into the
  exported two-logit classifier, retaining exact checkpoint input semantics.
- Select C by highest validation average precision, breaking ties with lowest C.
  Each candidate's threshold maximizes validation F1 subject to recall >=90%.
  Retain all candidates and report accuracy, precision, recall, F1, specificity,
  ROC-AUC and AP. No threshold search on test or OpenI is allowed.
- Run the GPU extraction after the existing NIH coordinator exits successfully.
  The local launch registration pins the command, source hashes, train/validation
  manifests and feature artifact before waiting. Do not change this registered
  recipe or its implementation while queued/running.

```powershell
python -m src.training.train_xray_probe --task nih_report_pneumonia --pretraining mimic_ch --feature-checkpoint models/pretrained/xrv_mimic_ch/features_safe.pt --manifests data/processed/nih_pneumonia_v1 --output models/nih_mimic_v1/frozen --cache-images --device cuda
```

The initial NIH evaluator intentionally rejects mixed-pretrained XRV models. Do
not weaken that initial evaluator's guard to admit this follow-up. The new model
requires an explicit evaluation contract that verifies its MIMIC-only provenance
and discloses prior cohort exposure and source-label limitations. Training success
or passing implementation checks does not establish improved generalization.
