# RSNA development pretraining

## Rationale

The capacity and seed-ensemble experiments did not improve the deployed clean
pediatric model. This next experiment will learn adult radiographic opacity
features from additional images and then adapt the feature extractor to pediatric
pneumonia. The two targets are distinct; their labels must not be pooled into a
single pneumonia ground-truth dataset. The pediatric classifier will be reset
before adaptation.

## Data reservation

The 1,000 patients in the registered RSNA external cohort are removed before
selecting development images. Every acquisition of those patients is excluded.
Among remaining patients, select one acquisition by the same previously defined
hash ranking, before filtering its class. Use all 1,007 eligible opacity patients
and 2,014 hash-ranked normal patients. Other abnormal cases are outside this
pretraining target.

The deterministic class-stratified patient split contains:

| Split | Normal | Opacity | Total |
|---|---:|---:|---:|
| Train | 1,611 | 806 | 2,417 |
| Validation | 403 | 201 | 604 |

Cohort SHA-256:
`b8f2f92cc5cdfd2b757d82bcc107dff30426e5684e7f580c781cdef769817415`.
The registration and source metadata hashes are retained in
`data/processed/rsna_development_v1/protocol.json`.

## Required checks before training

Verify DICOM identity and pixel encoding, exact and near duplicates within
development and against both the pediatric inventory and reserved RSNA external
cohort. If duplicated patients/images connect development to the external cohort,
exclude the development records. Preserve excluded records and reason codes.
Do not silently change the registered external cohort. Pretraining must use only
the audited development training split; its validation split chooses checkpoints.

Both previously evaluated benchmarks are now observed, and later comparisons on
them are exploratory. Reserved external patients must remain outside training and
calibration even though their benchmark scores have been seen.

## Reproduction

```bash
python -m src.data.prepare_rsna_development
python -m src.data.extract_rsna_cohort --cohort data/processed/rsna_development_v1 --images data/raw/rsna_development_v1
python -m src.data.audit_rsna_development
```

Current status: all 3,021 development images decoded and passed source integrity
checks. The combined audit of development and the 1,000 reserved external images
passed: zero components crossing train, validation or external splits. The
development images also passed exact and screened near-duplicate comparison to
the pediatric inventory.

## Training started

The fixed ImageNet-initialized ResNet18 pretraining uses CLAHE at 224px, mild
augmentation, batch size 24, learning rate 0.0003 (backbone 0.00006), two frozen
backbone warmup epochs, 16 epochs maximum and patience 5. Minimum unweighted
RSNA development validation cross entropy selects the feature checkpoint.
Checkpoint labels remain `Normal` and `Lung Opacity`; it cannot be loaded as a
pneumonia-serving checkpoint. Only the backbone is exported.

The pediatric adaptation is registered in `configs/evaluation_policy_v6.json`:
reset classifier, transfer the complete backbone, then use the same clean
ResNet18/CLAHE recipe and validation policy as the incumbent. Only the locked
winner is eligible for legacy-test evaluation. New runs do not replace deployment
automatically. Pretraining completed in 10 epochs; its minimum validation loss was
0.27828. On the 604-image **RSNA development validation** set, the fixed 0.5
threshold gives accuracy 89.57%, precision 85.94%, recall 82.09%, F1 83.97%, and
ROC-AUC 95.98%. These are development results for adult opacity, not external or
pediatric pneumonia performance. Pediatric adaptation completed in 15 epochs,
selecting epoch 10 (validation loss 0.04480). It won the prespecified validation
comparison but produced test accuracy 89.10%, precision 85.15%, recall 100%, and
F1 91.98%. Compared with the deployed clean model it missed two fewer positives
but generated five more false alarms. It is not deployed; see the
[measured comparison](clean_v6_results.md). No external RSNA retuning was performed.

```bash
python -m src.training.pretrain_rsna
python -m src.training.train_clean --model resnet18 --preprocessing clahe --cache-images --epochs 16 --patience 5 --minimum-recall 0.98 --feature-checkpoint models/clean_v6/rsna_pretrained/features.pt --output models/clean_v6/rsna_then_pediatric
```
