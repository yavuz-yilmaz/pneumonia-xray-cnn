# Frozen SSMU model: exploratory OpenI check

Written before inference with the frozen SSMU checkpoint on OpenI. OpenI was
already observed for five earlier models before SSMU development. This is a
separate exploratory transportability check, not a new untouched final test,
and does not amend the original registered four/five-model OpenI studies.

- Pin `models/ssmu_probe_v1/selected_joint/best_model.pt` to SHA256
  `6e8202040581f8e8412fea206f66e1eb748471e0161fff3d4cc56396b237fbc6`.
  Keep its SSMU-validation threshold **0.7265916466712952**. Verify the frozen
  MIMIC-only features and binary label mapping. No OpenI fitting or tuning.
- Reuse the existing 2,892-case / 2,983-frontal-image cohort, including all
  original exclusions. Verify every retained image hash and cohort artifacts.
  Average probabilities per case, then apply each fixed threshold.
- Reuse original/current saved image and case scores after checking their
  hashes against the completed secondary-study registration, labels, case
  alignment, aggregation and metrics. Do not rerun baseline image inference.
- Register the model, inputs, sources, protocol, threshold and baseline arrays
  before one new model inference. Refuse an existing output directory; preserve
  failure state. Keep the earlier protocols and studies unchanged.
- Report accuracy, precision, recall, F1, specificity, balanced accuracy,
  ROC-AUC, AP and confusion matrices. Use 1,000 paired case bootstrap draws,
  seed 20261002, with undefined precision recorded and valid draw counts.
  Report SSMU-minus-original/current differences and preserve negative results.
- Do not use these outcomes to tune the model or threshold. Do not deploy it
  automatically. SSMU patient identities and clinical labels remain unverified.
  OpenI negatives are absence of indexed/mentioned pneumonia, not confirmed
  normal lungs; there are only 32 positives. Independent original patient IDs
  are unavailable. This cannot establish clinical reliability or the complete
  leakage-free improvement goal. Data terms remain separate from code licensing.

Local registration/results: `reports/metrics/ssmu_openi_check_v1/`.
