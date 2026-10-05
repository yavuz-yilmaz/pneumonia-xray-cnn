# Additional dataset acquisition and screening — 2026-10-02

Scope: pneumonia classification for a personal portfolio, with adult data
permitted. The user has no approved PhysioNet/CITI access. Raw candidate data and
private audit evidence remain outside Git. SSMU now has a separately registered
development experiment with conservative proxy grouping; it is not a verified
patient-independent clinical cohort. Other candidates remain unused for models.

## BDCXR-3257: acquired and decoded; unresolved terms and patient identity

[Publisher record](https://doi.org/10.6084/m9.figshare.32086362.v1) lists CC BY 4.0.
Its complete 922,993,307-byte ZIP passed publisher MD5, every member CRC and all
3,257 image decodes. Archive/CSV counts agree: 880 normal, 1,477 bacterial and
900 viral images; the publisher description totals only 3,245.

The conservative byte/pixel/near-duplicate screen compared all images with
112,120 NIH, 5,856 legacy and 7,470 OpenI images. It found zero cross-source
matches and one within-source pair, without a conflicting binary label.
Cropping/contrast changes can evade this screen. Zero matches is not proof of
patient independence or authentic hospital provenance. The CSV has age, gender
and class but **no patient identifier**. Most age entries are pediatric; three
numeric entries are 25, 26 and 40, requiring clarification rather than silently
discarding anomalies.

The bundled three-page hospital/university Data Use Agreement says:
“The dataset is provided exclusively for non-commercial academic research
purposes” and prohibits public raw-data release without provider permission.
Its scope concerns named parties; it is not established that it binds every
user of the public CC BY record. The conflicting public license and included
contract leave authority/scope unresolved for this personal portfolio. Therefore
do not train, score or distribute this candidate pending clarification. No
provider was contacted. Acquisition occurred under the public CC BY record
before the bundled conflict was discovered; private evidence is preserved.

Evidence: `reports/data_access_20261002/bdcxr3257_download.json`,
`candidate_metadata_audit.json`, `bdcxr_document_review/`, and
`data/processed/bdcxr_images_v1/image_audit.json`.
Archive SHA256:
`90811e5a1ef2a66287ab38b0044dec9340bcc6d567c9f885bba683976cfe413c`.

## SSMU pneumonia/norm: full acquisition and image audit completed

[Zenodo publisher record](https://doi.org/10.5281/zenodo.5732746) lists CC BY 4.0
and a 4,185,026,327-byte `dataset.tar.gz`. Publisher description attributes
frontal/lateral acquisitions to ApolloDRF and authors affiliated with Siberian
State Medical University; clinical labels and patient grouping need verification.

The original connection broke after 289,406,976 complete bytes. A separate
recovery preserved that prefix and completed all remaining exact ranges at
11:11 UTC. Whole-file publisher MD5 matched. SHA256:
`430127367f26dd7eb4e6a74f19fc2c7649e975a25c3c05df732810e5dc879dba`.
The original partial and recovery records remain preserved.

Full TAR inventory, gzip CRC and **all 1,837 PNG decodes** passed. There are
1,035 images in `pneumonia` and 802 in `norma`. The initial audit's label parser
recognized norm/normal but not norma; it retained null labels for those 802.
A separate semantic readback documents the normal-folder interpretation without
overwriting the original audit or treating source labels as clinically verified.

The filenames indicate **934 frontal images** (518 pneumonia / 416 norma) and
903 lateral images. There are 934 directory/stem pairs, 903 with both views and
31 frontal-only. **401 numeric stems appear in both class directories**. There
is no metadata/patient table in the complete archive. Neither global stems nor
directory/stem pairs are verified patient IDs; numbering may restart by class.
No verified patient-independent split has been established. A subsequent
[development protocol](ssmu_development_protocol.md) conservatively groups equal
numeric stems across both classes and views and all known duplicates. After
removing five redundant frontal images, it reserves 665 train / 133 validation /
131 test images. Known byte/pixel/proxy/component and near-match overlap audits
passed. This does not resolve unknown patient identities.

The conservative NIH/legacy/OpenI screen recorded zero cross-source matches and
11 internal pairs: five exact-byte and six near matches, with no class-directory
conflict. This is not a proof of independence. Six normal lateral images are
16-bit; original L conversion clipped them, so their near/pixel screening is
limited and requires a separate normalization audit before use. None of the 11
recorded pairs involves these six. All 934 frontal filename-hint images are
original 8-bit L mode. No age metadata verifies an adult-only cohort.

Evidence: `pneumonia_norm_2021_resume2.json` and
`ssmu_inventory_readback_v1.json` in `reports/data_access_20261002`, plus
`data/processed/ssmu_images_v1/{registration,image_audit,tar_inventory}.json`.
Acquisition and archive auditing completed; patient identity/clinical label
verification remain separate, unfinished steps. The first frozen-feature model
experiment completed using train/validation only (`models/ssmu_probe_v1`).
A separately registered one-time test now shows higher four-metric point
estimates than both baselines on 131 held-out source images; precision's paired
interval includes zero. Test is now observed/reserved. The full leakage-free
and generalization goal remains unproven. See [model results](ssmu_model_results.md).

## MultiCaRe thoracic subset: metadata acquired; provisional NLP targets

[Subset record](https://doi.org/10.5281/zenodo.20548927) supplies a 34,747,785-byte
ZIP with tables and **no image payloads**. Publisher MD5 and all 18 member CRCs
passed. There are 5,252 cleaned rows / 3,415 patient IDs. The intermediate label
table has 713 pneumonia ones and 4,539 zeros. Its README explicitly describes
**pre-negation NLP labels**, not clinically verified pneumonia labels. Zero is
not confirmed normal. Captions/case text must be linked to the correct image,
patient, view and time point before labels can be assessed.

The subset lists CC BY-NC-SA 4.0 and describes the original similarly, but the
[original publisher record](https://doi.org/10.5281/zenodo.13936721) actually lists
CC BY 4.0. Its nine image ZIPs total roughly 3.2 GB. Original images have not
been downloaded. Retain per-image article licenses; the collection-level entry
must not silently relicense every paper figure. Published case reports also
carry selection bias and may show follow-up images after treatment.

Further linkage screening matched every cleaned label row to exactly one caption
by `(file, patient_id)`, without duplicate keys. Of the 713 positive rows, 640
are tagged frontal. All 5,274 caption rows have mixed article licenses: 1,784
CC BY, 1,566 CC BY-NC-ND, 854 CC BY-NC-SA, 776 CC BY-NC, 289 NO-CC CODE and five
CC BY-SA. The original dictionary says NO-CC CODE requires manual license
retrieval. Its patient ID is article PMC ID plus a sequential case number;
this does not verify that one person never appears in multiple articles.

A local phrase screen found 29 positive rows containing a possible negation
near “pneumonia”. Examples include “no evidence of pneumonia” and “no recurrence
of pneumonia”; other hits negate a different condition, so **29 is not an error
count** and this regex must not relabel data. It illustrates why case-wide
keyword targets need image-specific clinical and temporal review. The subset's
view labels also include implausible combinations such as thoracic X-rays tagged
as ultrasound/intravascular views; metadata alone is insufficient verification.

The current author README calls the project CC0, while this pinned original
record lists CC BY 4.0 and its images retain article-specific terms. These
version/scope signals must be reconciled before any distribution claim.

Evidence: `multicare_subset_download.json`, `candidate_metadata_audit.json`,
`multicare_subset_README.txt`, and `multicare_original_publisher_metadata.json`
under `reports/data_access_20261002`.
The further audit and acquired author README/checksum-verified dictionary are
recorded in `multicare_linkage_v1.json` in that directory. No images were acquired.

## Other candidate records

DataCite discovery returned 145 DOI records, including versions, unrelated
records and derived/model outputs; these are **not 145 eligible cohorts**.
The Pakistan Mendeley release `10.17632/8cm4ffk7b4.1` describes 1,345 pneumonia
images only. SSMU's newer part 2 lists 100 positive images. A retry acquired
[part 1 publisher metadata](https://doi.org/10.5281/zenodo.18241541): 99 PNGs
(50 frontal filename hints and 49 lateral), 231,954,776 bytes, CC BY 4.0,
collected in 2024–2025 according to the publisher. It also supplies only
pneumonia images, with no patient grouping table in the file list. The date
range alone cannot establish patient independence from the 2021 release.
No 2026 images were acquired or scored. DMCH-PneumoX describes 426 annotated
positive and 5,574 unlabeled images with restricted metadata. None supplies a
verified, accessible binary normal/pneumonia cohort for immediate training.

PadChest remains excluded from image acquisition for this personal portfolio
without the required nonacademic approval. VinDr/MIMIC/BRAX require credentialed
access; CheXpert's current destination requires an application. See the
[full access inventory](dataset_access_inventory.md).

## SSMU embedded identity investigation

All 929 selected frontal files matched their recorded source bytes and decoded
successfully for PNG/EXIF inspection, including metadata after image payloads.
None had PNG information keys or EXIF tags. Six deterministic training-only
visual samples showed no apparent identifying text; this is not an inspection
of all image pixels or a proof that burned-in text is absent. No OCR or external
image transmission occurred. The investigation did not resolve patient grouping.

Evidence: `reports/data_access_20261002/ssmu_identity_v1/png_metadata_audit.json`.
A [publisher clarification draft](ssmu_identity_questions.md) asks for anonymous
grouping and label provenance. It remains unsent.
