# Fifth experiment: fixed seed ensemble

The v4 capacity/augmentation experiment did not beat the incumbent on validation.
This experiment tests variance reduction from a fixed three-seed ensemble using
the successful ResNet18/CLAHE recipe. Registration is in
`configs/evaluation_policy_v5.json`, before new training and selection.

- Same audited pediatric train/validation split; no new images or test fitting.
- Members: incumbent seed 42, new seed 7, new seed 2026.
- Same mild augmentation, learning rate 0.0005, 16 epochs maximum, patience 5.
- Each member checkpoint uses minimum validation cross entropy.
- Fixed arithmetic average of three probability vectors; no tuned weights.
- Compare only the incumbent and the fixed ensemble on validation specificity at
  recall >=98%, then F1. Individual new seeds are not additional candidates.
- Evaluate only the locked winner. If the incumbent wins, reuse existing test
  results. Neither the pediatric legacy test nor RSNA external v1 is untouched
  now; subsequent results on either are exploratory.

The ensemble checkpoint contains all member weights and source checkpoint hashes.
Export rejects incomplete runs, unequal data splits, or different preprocessing.
Consumers apply softmax to its returned log-probabilities, recovering the fixed
probability average. It costs approximately three backbone forward passes per
image. No automatic deployment is performed during this experiment.

```bash
python -m src.training.export_ensemble --members models/clean_v3/resnet18_clahe models/clean_v5/resnet18_seed7 models/clean_v5/resnet18_seed2026 --output models/clean_v5/ensemble3
python -m src.evaluation.evaluate_clean select --experiments models/clean_v3/resnet18_clahe models/clean_v5/ensemble3 --minimum-recall 0.98 --output models/clean_v5/selected
```

Both runs completed (seed 7: 14 epochs; seed 2026: 10 epochs). The fixed ensemble won validation but did not improve test accuracy/F1 over the clean incumbent. It is not deployed. See [measured results](clean_v5_results.md).
