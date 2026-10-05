# Dataset access inventory — rechecked 2026-10-02

## Revised scope

The user explicitly permits an adult-only project. The original Kermany cohort's
pediatric population is a property of that dataset, not a project requirement.
Do not block adult model development on access to pediatric data. This inventory
records source access and the registered experiments within the user's existing
task scope. Age coverage follows each source's verified metadata; the locked
OpenI cohort has no age filter without reliable age metadata.

## Current experiment status

NIH training and its one-time test, the MIMIC-only frozen-feature probe,
original source-head calibration and both OpenI comparisons have completed.
Saved-artifact consistency reviews passed. None established the requested
substantial improvement across accuracy, precision, recall and F1; deployment
is unchanged. See [completed MIMIC/OpenI results](mimic_openi_results.md) and
[NIH results](nih_v1_results.md). The further joint-metric calibration completed
on NIH validation only with no feasible threshold, independently verified by
exhaustive enumeration. No checkpoint was exported or additional OpenI
evaluation performed.

The 2026-10-02 [MIMIC final-block adaptation](nih_transfer_results.md) completed
all ten epochs on NIH development only. Validation AP increased slightly to
0.0229255, but none of 14,197 independently enumerated thresholds met all four
baseline metric floors. Final artifacts and original train/validation source
hashes passed review. No NIH test/OpenI rescoring or deployment change occurred.

Additional sources were acquired: BDCXR-3257 passed all 3,257 image decodes and
the cross-source screen, but lacks patient IDs and includes academic-only terms
conflicting with its public CC BY record. It is not admitted to training/scoring.
SSMU's CC BY pneumonia/norm archive was fully acquired with publisher checksum
and gzip CRC verification; all 1,837 PNGs decoded. It has 934 frontal filename
hints, zero recorded NIH/legacy/OpenI matches and 11 internal duplicate pairs,
but no patient table. Six 16-bit lateral images have limited near/pixel screening
after L conversion; all frontal images are original 8-bit. No patient-independent
clinical cohort is established. A [registered SSMU development experiment](ssmu_development_protocol.md)
now uses conservative global-number/duplicate grouping, with 665 train / 133
validation / 131 reserved test images after frontal deduplication. Known overlap
checks passed; the frozen MIMIC-feature classifier completed without training-time
test access. Its separately registered 131-image test gives 86.26% accuracy,
95.00% precision, 79.17% recall and 86.36% F1, higher point estimates than both
baselines on that same cohort. Precision's paired difference intervals include
zero; patient identities/clinical truth remain unverified. See
[SSMU results](ssmu_model_results.md). Deployment remains unchanged.

The frozen SSMU model's separate **exploratory** OpenI check subsequently gave
47.61% accuracy / 1.44% precision / 68.75% recall / 2.82% F1, with 1,505 false
positives. It does not establish generalization or all-four-metric improvement.
All 929 SSMU inputs had empty PNG/EXIF metadata; patient grouping remains
unverified. SSMU 2026 part 1 metadata is accessible (99 pneumonia-only PNGs,
50 frontal filename hints, CC BY 4.0); images were not acquired. See the
[candidate review](additional_dataset_candidates_20261002.md).
MultiCaRe subset tables were acquired, without images;
their intermediate NLP targets need negation and image/time linkage review.
See [new candidate details](additional_dataset_candidates_20261002.md).

The user confirmed a **personal portfolio project** and **no approved PhysioNet
account/CITI training**. PadChest-pneumonia images were not acquired because
nonacademic research requires publisher approval. Public mirror visibility does
not grant that permission. Historical acquisition/queue snapshots below retain
their original timing; this section supersedes their pending-status statements.

| Source | Access observed | What it supports | Main caveat |
|---|---|---|---|
| RSNA 2018 | Official public download; archive already downloaded and hash verified | Approximately 30,000 NIH-derived images with opacity/normal/other-abnormal annotations and NIH mappings | Opacity is not confirmed clinical pneumonia; overlaps NIH |
| NIH ChestX-ray14 | Official metadata and all 12 image archives (45.08 GB) downloaded; all archives passed local size/SHA256 provenance, gzip CRC and image decoding checks | 112,120 images / 30,805 patients, including pneumonia labels | Report-derived labels; cannot act as an independent hospital test for RSNA. Split audit passed; both candidates and one-time test completed with weak results; no successful replacement |
| CheXpert / CheXpert Plus | Original CheXpert page currently links to CheXpert Plus on Stanford Redivis; anonymous access shows Metadata and Apply for access | Original CheXpert describes 224,316 images / 65,240 patients. Current Plus page describes 223,462 image/report pairs / 64,725 patients and 14 pathology annotations | An access application is required for current destination files; these releases must not be conflated; images have not been downloaded |
| MIMIC-CXR-JPG 2.1.0 | Public documentation, restricted files | Structured labels including pneumonia; patient/study identifiers | Credentialed PhysioNet account, CITI training and signed DUA required |
| VinDr-CXR 1.0.0 | Public documentation, restricted files | 18,000 images, 15,000 train / 3,000 test; radiologist findings and disease impressions including pneumonia | Credentialed PhysioNet account, CITI training and signed DUA required; inspect patient identity guarantees |
| PadChest | Author repository documentation available; primary website connection failed during audit | >160,000 images, ~67,000 patients; findings, diagnoses and patient metadata | Authors require formal access request; mixed manual/automated annotations; full data roughly 1 TB |
| PadChest-pneumonia / BIMCV subset | Author README and revised metadata acquired; author-linked mirror publicly lists a 2.85 GB resized image archive | Metadata has 23,521 image records / 13,114 anonymized patient IDs; valid nonpediatric frontal candidate screen has 15,471 images / 12,906 patients, 5,083 pneumonia-coded images | Personal portfolio use requires publisher approval under mirror license clause 1. Images not downloaded. Supplied partitions leak patients, including 961 patient IDs among valid records; not clinical ground truth |
| BRAX 1.1.0 | Official PhysioNet documentation accessible; files credentialed | 40,967 images / 19,351 patients from a Brazilian hospital, including report-derived pneumonia labels | CITI and signed DUA required; radiologist privacy review does not imply adjudicated pneumonia labels |
| PadChest-GR | Paper accessible; BSC public share lists image archives, but publisher access/permission not verified | 4,555 manually curated studies with grounded finding descriptions | Paper explicitly excludes differential diagnosis labels such as pneumonia; not a directly suitable pneumonia target |
| MIDRC-RICORD-1C | Official TCIA collection and annotation ZIP publicly accessible; annotation ZIP downloaded and CRC/hash verified | Publisher describes 998 chest radiographs / 361 adult patients, four sites; thoracic radiologist annotations | All patients COVID-positive; labels concern COVID radiographic appearances. Annotation identities and reader/adjudication reconciliation need resolution; not directly general pneumonia truth |
| OpenI / Indiana | All 3,955 official XML reports and the 1.36 GB PNG archive acquired; gzip CRC and all 7,470 image decodes passed | Independent case comparison cohort locked: 2,892 cases / 2,983 unique frontal images, including 32 manually coded pneumonia cases | Small positive count; report codes and secondary manual view labels; no NIH/legacy overlap found by the conservative screen, not proof of all possible independence; CC BY-NC-ND 4.0 |
| Epic Chittagong / Mendeley v5 | CC BY 4.0 confirmed; full ZIP/PDF acquired with publisher checksum verification; all 2,626 images CRC/decode verified | 2,545 images (96.9%) match already acquired NIH/legacy under the fixed screen; publisher/PDF claim 3,355 | Rejected: 62 cross-partition match pairs and 48 identical-byte pairs with opposing labels; only 81 unmatched images, without patient/label provenance. No training/scoring. See [candidate review](epic_candidate_review.md) |

## Source links checked

- [RSNA official release](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
- [NIH official file listing](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [NIH access and attribution documentation](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)
- [CheXpert official description](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [Current Stanford download destination](https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x)
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/)
- [PadChest author repository](https://github.com/auriml/Rx-thorax-automatic-captioning)
- [PadChest-pneumonia subset author documentation](https://github.com/BIMCV-CSUSP/BIMCV-COVID-19/tree/master/padchest-covid)
- [BRAX official release](https://physionet.org/content/brax/1.1.0/)
- [PadChest-GR author paper](https://arxiv.org/html/2411.05085v2)
- [MIDRC-RICORD-1C official collection](https://www.cancerimagingarchive.net/collection/midrc-ricord-1c/)
- [OpenI report archive](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz)
- [OpenI PNG archive](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz)
- [Indiana source paper and patient sampling](https://pmc.ncbi.nlm.nih.gov/articles/PMC5009925/)
- [Epic Chittagong candidate, version 5](https://data.mendeley.com/datasets/wndbd5r26y/5)

NIH's Google Cloud bucket is documented as Requester Pays, not an anonymously
free download route. The Box listing is a separate public access route. RSNA's
official page permits research/education and commercial/noncommercial use subject
to attribution. Repository code licensing does not relicense any source data.

## Recommended next decision

Keep **pneumonia** as the current target. The registered NIH experiment is now
complete as an accessible, report-label baseline and did not establish the
requested improvement. The preregistered MIMIC transfer and OpenI comparisons
also completed without across-metric success. Prioritize VinDr-CXR for expert-annotated pneumonia
impressions and evaluation from a different institution once access is obtained.
Its 18,000 images were annotated by 17 radiologists: three readers per training
image and five-reader consensus for test images. These are radiographic disease
impressions, not independently confirmed clinical diagnoses. Inspect patient
identity and duplicate guarantees before accepting any supplied split.

MIMIC-CXR-JPG and CheXpert are larger report-labeled alternatives; larger size alone
does not guarantee cleaner pneumonia targets. Audit positive/negative/uncertain
labels and source identity before adding them. PadChest-GR's curated finding
annotations deliberately omit pneumonia differential diagnoses, so do not use
its opacity/consolidation findings as pneumonia ground truth.

The openly licensed Epic Chittagong release was fully acquired and audited.
Its 2,626-image archive contradicts the 3,355-image publisher/PDF description;
2,545 images match NIH/legacy under the exact/near screen. It also contains 62
cross-partition match pairs and 48 identical-byte pairs with conflicting labels.
Reject it as new clean data or an independent hospital test. The 81 unmatched
images have no verified patient/label provenance. A source page claim alone is
not sufficient evidence of a cleaner dataset.

RSNA supports an explicitly separate **radiographic lung opacity** task if desired.
It does not supply adjudicated pneumonia truth and overlaps NIH. Preserve the
existing 1,000 reserved external patients and disclose prior exposure. Do not
compare adult cohort accuracy directly with the old pediatric benchmark as a
before/after improvement. For public distribution, check the dataset and derived
weight terms separately from the repository code license.

Previous RSNA work only evaluated 1,000 reserved patients and pretrained features
on 3,021 separate patients for subsequent pediatric adaptation. It did not train
and evaluate the proposed adult-focused model on the broader RSNA cohort.

## Current local evidence and proposed sequence

On 2026-10-01, the local RSNA archive and final Calculated annotations were
inspected. There are 30,000 DICOM images, 29,684 images with final labels and
12,233 patients among those labeled images. Final labels comprise 7,106 Lung
Opacity, 9,790 Normal, and 12,788 No Lung Opacity / Not Normal images. The
remaining 316 images must not be silently treated as negatives.

All 30,000 DICOM headers were read without decoding or changing image pixels.
ViewPosition contains 16,248 PA and 13,752 AP images. PatientAge is present as
bare numeric text, rather than standard DICOM AS values with an explicit unit.
Values span 1–412; 1,464 images have values below 18. Therefore the entire RSNA
archive must not be described as adult-only, and a simple value >=18 filter is
insufficient without handling implausible ages. Corroborate ages with NIH metadata
and record unknown/invalid-age exclusions before claiming an adult-only cohort.
Aggregate observations are saved locally in
`reports/data_access_20261001/rsna_age_inventory.json`.

The RSNA mappings also retain original NIH report-derived labels; 904 images
include Pneumonia in those original labels. Those labels are distinct from the
adjudicated RSNA opacity target and are not an additional independent dataset.

Recommended order:

1. Inventory images, patient identity, age quality, labels and prior experiment
   exposure before choosing the final task. Pediatric coverage is optional, not a
   prerequisite. Adult-only scope is permitted, not required by dataset access.
2. Preserve the completed NIH/MIMIC/OpenI results and previously evaluated
   cohorts. NIH test and OpenI are now observed; future development needs a new
   independent evaluation cohort for a fresh final claim.
3. Audit the next cohort's label quality, uncertainty and patient identity before
   acquiring images or launching another experiment. Do not assume newer labels
   or a bigger dataset are cleaner. RSNA opacity remains a separate possible task.
4. For a specifically pneumonia-labeled model and evaluation from another
   institution, prioritize VinDr-CXR or MIMIC after required access, with
   CheXpert Plus as another application-based option. PadChest remains a
   candidate: its author documentation is accessible, but its primary host could
   not be resolved in this environment on this date.
5. Select architecture and thresholds on development data; report precision,
   recall, F1, ROC-AUC and PR-AUC on a locked patient-disjoint test. Improvement
   on a new adult task cannot be presented as a direct numerical improvement
   over the earlier pediatric benchmark.

The initial recheck performed source/access research and local RSNA metadata
inspection. Subsequent continuations completed acquisition and the registered
experiments. The sections below retain the chronology of those access checks;
the current status and completed reports above describe the final outcomes.

## NIH acquisition and corrected metadata — 2026-10-01 continuation

The official `Data_Entry_2017_v2020.csv` has now been downloaded through NIH's
public Box page. It is 9,003,496 bytes; SHA256 is
`c69a6daca3549af707ca9cacbdf9f9a7b6a9188e8c61157df653520bb72d8eb1`.
The publisher's `batch_download_zips.py` was downloaded and parsed as data to
obtain source URLs; its code was not executed. A 64-byte HTTP range from the first
archive returned 206, the expected total size of 2,008,470,987 bytes, and gzip
magic bytes. The resumable downloader checks Content-Range, saved-tail continuity,
gzip magic and official listing size, then records its own SHA256. It does not
claim a match against a publisher-supplied checksum; extraction must still verify
gzip integrity and image decoding.

`python -m src.data.inspect_nih` produces an immutable, hash-linked inventory.
This inventory is not a training/test split or duplicate audit. Findings:

| Quantity | Count |
|---|---:|
| All NIH images / patients | 112,120 / 30,805 |
| Images aged 18–120 / patients | 106,718 / 29,324 |
| Report-derived pneumonia images (all ages) | 1,431 |
| Adult pneumonia images / patients | 1,319 / 938 |
| Adult pneumonia images already mapped to RSNA | 834 |
| Additional adult pneumonia images outside RSNA | 485 |
| Adult pneumonia images / patients absent from prior RSNA cohorts | 688 / 531 |

All 30,000 RSNA mappings match this official NIH metadata. The updated NIH ages
range from 0 to 95; using these corrected ages, 28,489 RSNA images are aged 18 or
older. Previous raw-DICOM numeric ages must not override this metadata.

The prior 1,000-image RSNA external cohort contains 60 under-18 acquisitions;
the 3,021-image development cohort contains 187. Earlier descriptions of those
cohorts as adult-only were inaccurate. Their existing results are mixed-age
RSNA opacity results, preserved without changing cohort membership or scores.
Their 4,021 patient identities remain tracked to avoid presenting previously
used patients as a newly untouched test. Final model improvements remain unproven.

### Completed image acquisition

All twelve official archives completed their byte-size/SHA256 verification,
full gzip CRC checks and image decoding. Their aggregate extraction audits report:

| Archive | Decoded images | Adult images | Adult report-pneumonia images |
|---|---:|---:|---:|
| images_001.tar.gz | 4,999 | 4,856 | 62 |
| images_002.tar.gz | 10,000 | 9,382 | 107 |
| images_003.tar.gz | 10,000 | 9,507 | 99 |
| images_004.tar.gz | 10,000 | 9,592 | 92 |
| images_005.tar.gz | 10,000 | 9,355 | 119 |
| images_006.tar.gz | 10,000 | 9,414 | 116 |
| images_007.tar.gz | 10,000 | 9,642 | 166 |
| images_008.tar.gz | 10,000 | 9,495 | 153 |
| images_009.tar.gz | 10,000 | 9,657 | 112 |
| images_010.tar.gz | 10,000 | 9,586 | 123 |
| images_011.tar.gz | 10,000 | 9,560 | 108 |
| images_012.tar.gz | 7,121 | 6,672 | 62 |

The first two archives' local SHA256 digests are respectively
`fd8e3542db6351ae9377779033f5d5c5f32fe50eb0830b519fbf1a7e791354b1` and
`c849cfa5504b8cffb301952bf93d53d3d39d7d931bc88bb70427b4a67de0740a`.
The third and fourth archives' digests are respectively
`6a90a979850545e30ed3e9ba96de2f063db13aa2a3c22363820d3f70d8c882c8` and
`1ca953ec37fe9c132ec49743f9651b707884c6c76e6e3a33853940c13372a385`.
The fifth and sixth archives' digests are respectively
`a058c365478454dc7395ee2897c2b63a3bf108739b857046838859227e42e743` and
`9b00987d39e4c9e4ab18b7426db82197c043dacec5a827141446c3900d39d5dd`.
Archive eight's local digest is
`823088934d0ea57b4a7384446560bcdb825e712932b69e8541a84acee10af447`.
The remaining local digests are:

| Archive | SHA256 |
|---|---|
| images_007.tar.gz | `7868b43798cf1bc9ae98d142644864b056e64edda3be12f7504f16485ac382ff` |
| images_009.tar.gz | `bdd5bfe76323b630f5f83a013a0f2f100d1b03fdaea25240844e8763c1e40a51` |
| images_010.tar.gz | `8f18699d3baad03cdf8946220a95e8d2bab555dc8e4be99e13bc0366a72aba1c` |
| images_011.tar.gz | `7e7d190f71b5b792c495acb2eebbbf0563a8c528adb23e9335ec3b79cb5b486f` |
| images_012.tar.gz | `7316ce5f4d5e0154730e592cc2b45b48a2bee8457f6339d36c2ca55ed6e60b26` |

Publisher checksum comparison remains unverified. These per-archive audits prove
decoded-image availability and integrity, not global patient-disjoint splitting.
These twelve audits cover 112,120 decoded images, including 106,718 adult images
and 1,319 adult report-pneumonia images. All four transfer queues and the extraction
watcher completed successfully. The registered coordinator completed the global
patient/duplicate split audit over 117,976 NIH and prior pediatric fingerprints
and ran the ResNet18 candidate. That candidate completed after eleven epochs;
the registered validation-AP policy selected epoch five. Its validation accuracy
is 14.903%, precision 1.146%, recall 90.323%, F1 2.263%, ROC-AUC 0.6092 and AP
0.02430 at the recall-constrained threshold. These are weak validation results,
not final test results or evidence of the requested improvement. The EfficientNet
stage has started; final candidate selection and test scoring remain pending.
Checkpoint/history/protocol consistency passed and is recorded in
`reports/metrics/nih_run_v1/resnet18_stage_review.json`.

### Finalized NIH split — 2026-10-01

| Partition | Images | Original patients | Report-pneumonia images |
|---|---:|---:|---:|
| Train | 65,662 | 20,142 | 700 |
| Validation | 14,212 | 4,402 | 155 |
| Test | 12,158 | 3,828 | 118 |

The audit reports zero shared original patients, source-byte hashes, decoded-pixel
hashes or connected duplicate groups across all three partition pairs. The
documented near-duplicate screen found one pair across the full inventory and
zero crossing evaluation partitions. No previously exposed RSNA patient enters
the fresh test. An independent manifest read verified hashes, counts, full NIH
metadata coverage, adult labels/ages and preserved historical patient roles.
These checks support the stated split guarantees; perceptual screening cannot
prove that every possible near duplicate has been detected.

The excluded manifest holds 20,088 NIH images and all 5,856 previously observed
pediatric images used only for duplicate bridges. NIH exclusions comprise 15,852
images connected to reserved RSNA external patients and 4,236 age-ineligible
images. Validation and test retain natural cohort prevalence. The test contains
only 118 positive images, so patient-cluster uncertainty intervals and PR-AUC are
necessary when assessing any claimed improvement. No test predictions have been
computed during these integrity checks.

### Additional source check — 2026-10-01

[OpenI / Indiana](https://openi.nlm.nih.gov/) was initially checked as another
potential source for evaluation outside NIH. The home page and FAQ returned a
maintenance message, and a linked downloads host could not be resolved. That
initial check did not prove permanent unavailability. A later direct-archive
probe succeeded, as recorded below; this supersedes the earlier download-status
assessment. Third-party copies are not assumed to provide verified provenance
or clean pneumonia labels. The registered NIH protocol remains unchanged.

### Further access and suitability check — 2026-10-01

BRAX's official page returned HTTP 200 and confirms 40,967 images, 24,959 studies
and 19,351 patients. Its fourteen labels include pneumonia, but were extracted
from Portuguese reports with NLP. Image review for privacy is not evidence of
expert adjudication of every disease label. Files require PhysioNet credentialing,
CITI training and a signed DUA. No BRAX images were downloaded.

The PadChest-GR author paper returned HTTP 200. It describes 4,555 curated studies,
including 3,099 abnormal and 1,456 normal, and access by request through BIMCV.
Its methods explicitly state: "Labels for differential diagnoses that rely on
additional clinical context, such as 'pneumonia', were ignored." Therefore this
release is not a clean, direct pneumonia classification cohort. Its finding
grounding annotations may support a future separately defined task.

A third-party README linked to the BSC share
https://b2drop.bsc.es/nextcloud/s/PadChest-GR . Read-only public WebDAV listing
returned HTTP 207 and exposed 37 image archive parts totaling 38,531,867,282 bytes,
plus nine prior-study parts totaling 9,017,509,038 bytes and annotation filenames.
This proves a working public listing, not verified publisher permission or an
appropriate pneumonia label. No terms file was listed; the paper's CC BY license
must not be interpreted as a license for the dataset. The BIMCV publisher site
could not resolve in this environment. No PadChest-GR images or annotations were
downloaded or used. Unofficial mirror license claims remain unverified.

Research evidence is saved locally in
`reports/data_access_20261001/additional_sources_check.json`,
`reports/data_access_20261001/padchest_gr_and_access_check.json` and
`reports/data_access_20261001/padchest_gr_paper_and_listing.json`.
Only this inventory was edited; registered training, split and evaluation files
and their protocol hashes were preserved.

### Account availability and further public source — 2026-10-01

The user confirmed that they do not yet have an approved PhysioNet account/CITI
access. VinDr-CXR, MIMIC and BRAX therefore remain conditional future sources,
not files currently available to the training pipeline. NIH training continues.

The official TCIA MIDRC-RICORD-1C collection returned HTTP 200. It describes 998
chest radiographs from 361 adults at four institutions, all diagnosed with
COVID-19. Its images, annotations and clinical metadata carry CC BY-NC 4.0.
The official annotation ZIP was downloaded without account credentials from
https://www.cancerimagingarchive.net/wp-content/uploads/midrc-ricord-1c-da-other-json.zip .
The downloaded compressed file is 180,675 bytes, SHA256
`26fcd9e360f41e8150ff447e79d073b74c27daa8621dcaf7705a6283e8954489`;
ZIP CRC and JSON decoding passed. This is a local provenance digest, not a
publisher checksum comparison. No RICORD images were downloaded or scored.

The JSON contains 1,000 study records and 5,440 annotation records across six
label groups. Typical, indeterminate and atypical appearances are COVID-specific;
"Negative for Pneumonia" is described as "No lung opacities." These categories
must not silently replace the NIH report-pneumonia target. The three main reader
groups each reference 1,001 study UIDs, and classification conflicts occur in two
studies for each of two reader groups. The annotation/study listings and the
publisher's 998-image count require reconciliation before any image-level use.
Adjudication is present for a subset, not every study; blindly treating every
annotation as an independent labeled image would duplicate cases and readers.
This remains a potential supplemental evaluation resource with explicitly
different label semantics, not a proven cleaner general pneumonia training set.

Evidence and aggregate annotation counts are saved in
`reports/data_access_20261001/tcia_siim_access_check.json`,
`reports/data_access_20261001/ricord_annotation_download.json` and
`reports/data_access_20261001/ricord_annotation_inventory.json`.

### Working official OpenI archives — 2026-10-01

Direct HTTPS requests to NLM's report, PNG and DICOM archive URLs returned HTTP
206, correct gzip magic and total lengths in Content-Range. The report archive
completed with HTTP 200: 1,112,632 bytes, local SHA256
`8fb6de7eec73d8c3665067ad4bb003ccd57f971ae316d2642e1627ac7268667a`.
All tar members were read and all XML records parsed. There are 3,955 unique
report identifiers and 7,470 unique linked image identifiers. Every XML record
identifies Indiana University and CC BY-NC-ND 4.0. These data are not relicensed
by the repository's MIT code license and remain Git-ignored.

The source paper states: "In no case did we take more than one study per patient."
It collected studies from two Indiana hospital systems and manually coded salient
findings and diagnoses reported as present, not those reported absent. This
supports using report IDs as patient groups within this source; it does not prove
there are no duplicate images or overlap with other published cohorts. The
collection has not been verified as exclusively adult, which is not a requirement
for the user's overall project.

There are 42 reports (77 linked images) whose manual major terms include
pneumonia. This is a small report-coded target, not adjudicated clinical truth.
Three such reports do not literally mention pneumonia in FINDINGS/IMPRESSION.
Conversely, 223 reports without a manual pneumonia term do mention it there,
including potentially negated or uncertain statements. The paper explicitly
describes a suspicious-pneumonia report receiving only an airspace-disease code.
Therefore do not call all 3,913 untagged reports confirmed pneumonia negatives;
95 reports also carry "No Indexing." An explicit ambiguity policy is required
before model evaluation. No report mention was automatically converted into a
training or evaluation label during this inventory.

The official PNG archive is 1,360,814,128 bytes. Its resumable local download has
started while NIH GPU training continues. Final archive integrity, image decode,
view mapping and cross-source duplicate checks remain pending. No OpenI data was
used to train, tune thresholds or score a model. The collection is being acquired
as a potential evaluation source outside NIH; no performance gain is established.
The 80.69 GB DICOM archive was only probed, not downloaded.

Evidence is saved under `reports/data_access_20261001/openi_*`. The local download
helper verifies source range/length, gzip magic and resumed-tail continuity, then
records a local SHA256. Archive CRC/image decoding are separate pending gates.

### OpenI candidate screening and acquisition continuation — 2026-10-01

Before observing any OpenI model scores, all 3,955 reports were screened against
explicit metadata rules: require manual indexing, no manual technical-quality
flag, nonempty FINDINGS and IMPRESSION, and linked images. A positive candidate
requires a manual major term whose first hierarchical category is Pneumonia.
An untagged report mentioning pneumonia in FINDINGS/IMPRESSION is excluded pending
review, regardless of whether the mention is negated or uncertain. Negative
candidates mean no coded or mentioned pneumonia, not clinical normality.

This yields 3,011 metadata candidate cases: 36 positives with 68 linked images
and 2,975 negatives. There are 104 reports without any linked images. Exclusion
reason counts overlap; 536 reports have incomplete findings/impression, 223 have
uncoded pneumonia mentions, 86 have technical-quality flags and 95 lack manual
indexing. The initial assertion that every report links an image failed before
outputs were written; it was corrected by explicitly excluding empty-image cases.
Full source coverage, XML hashes, manual-code labels, image links and candidate
counts subsequently passed an independent readback check. These exclusions limit
representativeness and must accompany later results. They were not selected from
model errors or performance measurements.

`data/processed/openi_candidate_v1/metadata_candidates.json` has SHA256
`c919e0b57ce61877923ca91a9d7d6752e26ad757cd9772aef59203252bbaa149`.
It is explicitly not a training or evaluation image manifest: every row's view
and image/cross-source audit flags are false. Many images share the report-level
caption "PA and Lateral", so neither these captions nor filename ordering are
accepted as per-image view labels. No OpenI model inference has occurred.

The single-connection image download was deliberately replaced after confirming
slow transfer, preserving its 65,011,712-byte partial. Four separate HTTP ranges
now acquire the source with exact range/size and resumed-tail verification. The
original partial and range segments are preserved; no training job was stopped
or restarted. Assembly, full gzip CRC, decoding and overlap checks remain gates.

### Public MIMIC-specific pretrained features — 2026-10-01

The [TorchXRayVision author README](https://github.com/mlmed/torchxrayvision)
lists dataset-specific models, including `densenet121-res224-mimic_ch`, separately
from `densenet121-res224-all`. Its installed model registry and public release
identify MIMIC-CXR/CheXpert-labeler weights. This is a potentially useful source
of chest-radiograph features without downloading credentialed MIMIC images.
Publisher-designated pretraining provenance does not independently reveal every
training patient. The mixed `all` weights include NIH and OpenI and remain
unsuitable for independent evaluation on either source.

The author release asset
`mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`
was downloaded: 28,382,012 bytes, matching its GitHub release listing; local SHA256
`23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189`.
No publisher checksum was listed. No MIMIC data images were acquired.

The legacy checkpoint format failed PyTorch's built-in `weights_only` load. After
statically inspecting its globals, a loader restricted to explicitly named,
known PyTorch/XRV module classes loaded it on CPU. A rejection check for an
unlisted global passed. Its finite feature tensors transferred strictly into the
existing torchvision-compatible architecture; a zero-input CPU smoke check
returned features of shape 1x1024 and logits of shape 1x2. Only feature tensors
were saved into `models/pretrained/xrv_mimic_ch/features_safe.pt` and successfully
reloaded with `weights_only=True`; the source disease classifier was not copied.
Its SHA256 is
`7c5df8cf35f00b01d2b90e77f4301027c9d1555395a0c15512aaf2947973e7c0`.

This is cached preparation for a separately registered future transfer experiment,
not a trained or selected NIH pneumonia candidate and not an accuracy result.
Registered NIH candidates/protocol files were preserved. Publisher/package and
underlying data/weight distribution terms must be considered separately before
any public distribution. Provenance and CPU checks are saved under
`models/pretrained/xrv_mimic_ch/`; author-source research is under
`reports/data_access_20261001/xrv_*`.

### Registered follow-up and additional projection metadata — 2026-10-01

The MIMIC-only training path is now implemented and passed 18 focused tests plus
a CPU smoke check on two NIH training images, one per label. These checks measure
implementation readiness, not classifier performance. The bounded recipe in
[the probe protocol](nih_mimic_probe_protocol.md) was registered while the NIH v1
coordinator still reported `test_started: false`. Source files, development
manifests, split audit and feature artifact are pinned by SHA256. A local launcher
holds the original Windows coordinator process handle and starts the follow-up
only after its successful exit and a second hash check. The follow-up performs
NIH train/validation extraction and selection only; independent final evaluation
remains outstanding. Registration/status are in `reports/metrics/nih_mimic_run_v1`.
The user confirmed they have no approved PhysioNet account or completed CITI
training, so credentialed raw-image sources remain unavailable for this run.

The [raddar Indiana collection](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
public API permits downloading `indiana_projections.csv` alone, without downloading
its image repack. The publisher says each image was **manually classified** as
frontal or lateral; the images in that repack originate from DICOM, with clipping,
scaling and resizing. This is secondary manual view metadata, not independent
verification of original DICOM tags and not a new pneumonia diagnosis source.
The API lists the same CC BY-NC-ND 4.0 license as the official collection.

The 289,397-byte CSV has SHA256
`7af9c8a6b4f8ae695c654ecd4208edd193a37892a5607b7cb9e609dc997e6b7a`.
Matching explicit case UID plus full `IM-...` image suffix yields unique mappings
for 7,466 of the 7,470 official image references: 3,818 frontal and 3,648 lateral.
Four official links lack a mapping; no filename-order fallback is used. Among
previously fixed metadata-eligible candidates, 32 of 36 positive cases have a
secondary frontal label (33 images), and 2,860 of 2,975 negative candidate cases
have one (2,951 images). Image availability and overlap exclusions are still
pending; these counts are not a final evaluation cohort.

Three complete PNG members from the in-progress archive prefix decoded and
matched official report links and secondary projection metadata. This prefix
check does not establish full archive integrity. A separate audit has been
registered and started in waiting mode under `data/processed/openi_images_v1`.
After the downloader publishes full gzip verification, it will decode original
PNG bytes without extracting archive paths and screen exact/near duplicates
against all 112,120 NIH acquisition fingerprints, the legacy source inventory,
and other OpenI images. The near-duplicate rule is the existing conservative
pHash distance <=6 plus 64px grayscale correlation >=0.995; it is not a proof
that every possible overlap is detected. No OpenI model inference or final
evaluation manifest has been produced.

Evidence: `reports/data_access_20261001/openi_secondary_projection_*`,
`openi_projection_metadata_routes.json`, `openi_kaggle_description.json`,
`openi_image_precheck.json`, and the registered image audit/status directory.

### Completed OpenI image audit and locked external cohort — 2026-10-01

The official PNG archive finished at 1,360,814,128 bytes, SHA256
`baf3abfe19ba5d58efe69002aed1e71aa2e6d5efb3238db9adcac210ad44bdf2`.
The initial image audit stopped on an unexpected `./Thumbs.db` member. Its failed
registration/status are preserved under `data/processed/openi_images_v1`.
A full member inventory identified precisely 7,470 regular PNGs and one
60,416-byte Windows thumbnail database. The corrected, separately registered v2
audit skipped only that known member after verifying its type, size and hash;
it did not extract or deserialize it. Source images were preserved unchanged.

`data/processed/openi_images_v2/image_audit.json` confirms full gzip CRC, decoding
of all 7,470 PNGs and exact coverage of every official image link. All images are
RGB. The registered view join identifies 3,818 frontal, 3,648 lateral and four
unmapped images. The exact/near-duplicate screen compared all 112,120 NIH and
5,856 legacy acquisitions and detected no cross-source matches. Three exact-byte
matches were found within OpenI, all within the same case; no cross-case match
was detected. This is evidence under the fixed conservative screening rule,
not proof that every possible acquisition overlap is absent.

The [external comparison protocol](openi_external_protocol.md) was registered
before any OpenI inference. Its finalized cohort is stored in
`data/processed/openi_external_v1`: **2,892 cases / 2,983 unique frontal images**,
comprising **32 positive and 2,860 negative report-code cases**. All 3,955 reports
are accounted for by retained cases or 1,063 excluded cases. A duplicate frontal
frame was removed within its case; multiple distinct frontal frames retain one
case-level vote through mean probability aggregation. No age filter is applied
without reliable age metadata.

An independent readback verified the original report linkage, eligibility and
labels, a fresh join to the downloaded projection CSV, every retained image byte
hash, unique decoded-pixel fingerprints, exclusion coverage and absence of
retained recorded source/cross-case duplicates. Its report is
`reports/data_access_20261001/openi_external_independent_verification.json`.

`src/evaluation/evaluate_openi.py` implements the fixed comparison, keeps report
labels explicit, forbids mixed-source XRV pretraining, verifies candidate
provenance and source-manifest equality, preserves source-selected thresholds,
aggregates to case level, and reports paired bootstrap intervals. Undefined
precision is marked explicitly. Existing or failed evaluation markers prevent
silent rescoring. Thirty focused tests covering this evaluator, the MIMIC probe
and image preparation passed; Ruff passed. The four-model external comparison
was concretely registered and queued behind successful MIMIC training, using the
original process handle and pinned source/cohort hashes. Queue state is in
`reports/metrics/openi_comparison_run_v1`; no OpenI model scores exist yet.

### Completed NIH and source-head comparator — 2026-10-01, 14:54 UTC

The original NIH coordinator completed at 14:42 UTC. On its same 12,158-image
test (118 report-pneumonia positives), selected EfficientNetV2-S
accuracy/precision/recall/F1 are 21.13% / 1.15% / 94.07% / 2.26%. The current
classifier gives 61.62% / 1.62% / 64.41% / 3.15%. A saved-artifact review passed
independent metric recomputation, label/group alignment, selection and hash
checks and exact winner tensor equality without rescoring images. See
[NIH results](nih_v1_results.md). No successful replacement is established.

The MIMIC-only probe is preparing its 224px training image cache after the original
job's successful exit. No probe classification results are published yet. The
user has no approved PhysioNet account or CITI training; access to public author
weights does not mean MIMIC raw images were downloaded or used for local training.

The [source-head comparator](nih_mimic_source_head_protocol.md) keeps the original
MIMIC Pneumonia row (index 8), verified against stored targets and author registry.
Its safe head SHA256 is
`2381ed671a42a3c02418752b332660c37ca6c1f8ae686f715e63a9e67073b94e`.
Features/classifier stay frozen; only the threshold is calibrated using existing
NIH validation features after extraction completes. Its queued coordinator is
`reports/metrics/nih_mimic_source_head_run_v1`.

The secondary external evaluator was registered at 14:44:53 UTC before any OpenI
scoring. It waits for successful completion of both the primary comparison and
source-head calibration, scores the fifth model once, reuses four aligned saved
case-prediction arrays, and reports all five models with paired bootstrap
differences. It cannot replace the primary candidate based on OpenI results.
Queue: `reports/metrics/openi_source_comparison_run_v1`; output:
`reports/metrics/openi_source_head_v1`. During preparation, export/probe tests (15)
and secondary evaluator tests (8) passed. These checks establish implementation
readiness, not source-head performance. Deployment and registered recipes are
preserved.

### Additional PadChest-pneumonia access and metadata audit — 2026-10-01

The BIMCV authors publish a separate PadChest-pneumonia subset. The old image
server failed DNS resolution during this check, but the author-linked mirror
returned a public listing, including a 2,850,291,209-byte resized archive and
a 163,860,361,762-byte raw archive. Listing a file is not full-download or decode
verification. No images have been downloaded or used in a model here.

The publisher's revised metadata table was acquired from its public GitHub
repository (8,818,378 bytes, SHA256
`a31e90e691ce03f8e7a2e99bcda3195ae4df3e7dde71f7cea21e8fec613161ce`).
It contains 23,521 unique image IDs and 13,114 anonymized patient IDs, with groups
C (control), N (pneumonia), I (infiltration), and NI (pneumonia plus infiltration).
Infiltration-only must not be silently relabeled as pneumonia. This report-label
metadata does not establish adjudicated clinical pneumonia diagnoses.

An audit of the supplied partitions found 972 patient IDs spanning partitions
in all rows, **961 among publisher-valid rows**, and **913 after the metadata-only
Valid=1 / Pediatric=No / PA, AP or AP_horizontal screen**. The latter has 15,471
images / 12,906 patients, with 5,083 N/NI-coded images. It is a provisional
metadata screen, not a fixed evaluation cohort or a verified adult-age filter.
The original split must not be used directly. The publisher itself documents
duplicate/quality problems and manufacturer, age and image-background bias.
Removing invalid records resolves the table's recorded duplicate-group overlap,
but does not resolve patient overlap or prove all image duplicates are absent.

The mirror's 4,227-byte `LICENSE.txt` (SHA256
`3439b97898658ae89c7e12c2954d35fe7d505b9491f6ff98f86e88c6a1a7d477`)
uses the BIMCV-COVID19 Research Use Agreement. Clause 1 requires publisher
approval for non-academic research. The user confirmed a **personal portfolio**
project, so image acquisition/training is not treated as already licensed.
Clauses 5–6 also restrict redistribution of data and download links. The
repository's separate MIT software license does not supersede these data terms.
Direct mirror/download URLs and original source files remain in ignored research
evidence; they are not added as distribution instructions here. This source
restriction does not stop the registered NIH/MIMIC/OpenI work.

Evidence: `reports/data_access_20261001/bimcv_access_check.json`,
`bimcv_metadata_inventory.json`, the saved publisher metadata, README and mirror
license. An independent pandas readback agreed on image/patient/overlap counts.
