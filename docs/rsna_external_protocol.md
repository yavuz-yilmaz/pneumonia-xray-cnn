# RSNA external transfer evaluation protocol

## Purpose and label boundary

The [official RSNA 2018 release](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
provides images, adjudicated annotations, and mappings to NIH image identifiers.
This allows explicit patient grouping rather than the Kermany filename proxies.
The release has 30,000 mapped images. The final Calculated annotation group labels
29,684: 9,790 normal, 7,106 lung opacity, and 12,788 other abnormal images.
These cover 12,233 NIH patients; 2,973 patients have multiple class labels across
their acquisitions. Images must therefore never be split independently.

Adult **lung opacity** is a different target and population from pediatric
**pneumonia**. Results will be reported as an external transfer benchmark; they
must not be combined with the pediatric metrics or represented as confirmed
pneumonia diagnoses. The current cohort excludes other abnormal images.

## Cohort locked before model predictions

- Seed: 20260916.
- Choose one acquisition per NIH patient by a deterministic SHA-256 ranking,
  before filtering by class. This avoids preferentially taking a patient's
  positive image.
- Then select 500 normal and 500 lung-opacity patients by the same ranking.
- All 1,000 patients are distinct. Reserve their entire acquisition histories
  from future training and calibration, not just the selected images.
- Preserve actual RSNA source labels in the cohort; do not relabel lung opacity
  as confirmed pneumonia.
- Balanced sampling imposes 50% prevalence. Precision and accuracy do not estimate
  performance at real clinical prevalence.
- Cohort SHA-256: `63b14d6aa4d070302d7123d31bce53e8c8a880f8a4fc04b5516dceef54367b12`.
- The metadata cohort is locked in `data/processed/rsna_external_v1/cohort.json`;
  its protocol contains source metadata hashes and registration time.

## Evaluation rules

Complete image extraction, image integrity checks and overlap screening against
the local pediatric training/development data before scoring. Use the original
deployed ImageNet model and the winner locked by pediatric validation in v4;
retain their existing thresholds. External labels and scores must not select the
model, preprocessing or threshold. Report both positive and negative transfer.

The XRV `all` pretraining includes NIH/RSNA. Models initialized from those weights
are **not eligible for an independent external claim** on this cohort.

## Official source files

- [Images](https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip)
- [Adjudicated annotations](https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json)
- [NIH mappings](https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json)

Files are stored locally under `data/raw/rsna_metadata` and are not committed.
The official image archive download completed (3,978,753,654 bytes), SHA-256
`96b97d81eb042c6513196e01079a060a980d5e62502c3266527dd9c0b2a63b50`.
The repository software license does not relicense source medical images; retain
RSNA/NIH attribution and source usage terms when redistributing artifacts.

After obtaining the two metadata files:

```bash
python -m src.data.prepare_rsna_cohort
python -m pip install -e ".[rsna]"
python -m src.data.extract_rsna_cohort
```

The commands refuse to replace an existing cohort or completed image audit.
Extraction verifies the locked cohort and archive hashes, verifies DICOM SOP
identifiers, and preserves the original 8-bit MONOCHROME2 pixels in lossless PNG.
Unexpected pixel encodings or display transforms fail for review. Exact decoded
hashes and perceptual similarity are checked against the pediatric inventory.
External scoring requires a successful completed image audit. No external metrics
have been computed during cohort preparation.

## Completed preparation audit

All 1,000 selected DICOM images decoded successfully with the expected pixel
encoding. The extracted set contains 1,000 distinct NIH patients. The audit found
zero exact decoded duplicates within this cohort and zero exact or screened
near-duplicate matches to the pediatric source inventory. Screening uses the same
pHash distance <=6 and thumbnail correlation >=0.995 rule as the pediatric audit;
this is not a guarantee against every possible related acquisition.

The image manifest SHA-256 is
`aa189ceb7a44f3614d25e83e7eeae744bd81d8bc276677816b626374ce3c3472`.
Machine-readable evidence is in `data/processed/rsna_external_v1/image_audit.json`.
The validation-only v4 selection and external evaluation are now complete.
See [measured results](clean_v4_results.md). The cohort is now observed and must
not be described as untouched for subsequent experiments. Its patients remain
reserved from training and calibration.
