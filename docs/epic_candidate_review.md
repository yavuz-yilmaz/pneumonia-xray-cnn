# Epic Chittagong candidate: source and access review

## Current disposition

This candidate is **rejected as a new clean training or independent hospital
evaluation source**. The full ZIP and two-page PDF were acquired and verified
against their publisher checksums. All 2,626 images passed CRC and decoding.
Earlier HTTP 403/timeouts are historical observations, superseded by successful
file access.

Of 2,626 images, **2,545 (96.9%) match already acquired legacy or NIH images**
under the fixed exact/near-duplicate screen. The supplied training/testing split
has **62 matching image pairs across partitions**, including 27 identical-byte
pairs. **48 identical-byte pairs have opposing normal/pneumonia labels**.
The archive also contradicts the publisher/PDF's 3,355-image count. These issues
make another model experiment on this source unjustified. No model was
trained/scored and deployment is unchanged.

## Publisher claims and license

The [Mendeley release](https://data.mendeley.com/datasets/wndbd5r26y/5),
DOI **10.17632/wndbd5r26y.5**, was published on 18 August 2026. Its current title
is "A Primary Chest X-ray Dataset of Normal Bangladesh"; its description still
claims primary normal/pneumonia chest images from Epic Chittagong, Bangladesh,
collected in 2025. The prior version has a longer normal/pneumonia title.
The hospital provenance is the contributors' claim and is not independently
established here.

The [DataCite record](https://api.datacite.org/dois/10.17632/wndbd5r26y.5) returned
HTTP 200 and confirms open access and **CC BY 4.0**, matching the publisher page.
This differs from the nonacademic-use approval requirement of the audited BIMCV
PadChest subset. The record supplies no direct image content URL.

The description and acquired PDF report these counts:

| Supplied partition | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| Training | 321 | 321 | 642 |
| Testing | 1,363 | 1,350 | 2,713 |
| Total | 1,684 | 1,671 | 3,355 |

The page lists a 500 MB ZIP, a 704 KB PDF named "Dataset Splition File.pdf" and
an 810 KB PNG. The ZIP file-information view records publisher SHA256
`3d07a6f0d3ba37c9a72edba45d05e307ab57ebf2ba2effe2b3373da448dcad0c`.
The PDF file-information view records publisher SHA256
`1630a1333e63c67a0999554c7491f920de10afb009089e2acd8218a2a6fa5129`.
The PDF checksum matches the local 721,366-byte file; both pages were rendered
and inspected. It repeats the counts and primary-source claim, but supplies no
patient IDs, patient-disjoint guarantees, age/view metadata, specialist labeling
process or clinical confirmation. The complete local 524,339,616-byte ZIP
also matches its publisher checksum.

An independently inspected ZIP central directory, pinned to the observed strong
source ETag, contains 2,633 unique entries: seven directories and 2,626 images.
No additional metadata files were found. Complete member reading and image
decoding subsequently confirmed these counts:

| Archive partition | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| Training | 1,063 | 1,050 | 2,113 |
| Testing | 257 | 256 | 513 |
| Total | 1,320 | 1,306 | 2,626 |

The partial-download overlap screen includes `normal-7869.jpg` matching legacy
`IM-0037-0001.jpeg` (pHash distance 0, correlation 0.999994), and
`normal-7891.jpg` matching legacy `IM-0307-0001.jpeg` (distance 0, correlation
0.999996). This evidence concerns the inspected prefix; it does not establish
the origin of every archive image or every possible patient overlap.

Both example pairs were rendered side by side and visually inspected. They
show corresponding anatomy, framing and image marks consistent with the same
acquisitions under reencoding/resizing. The private renders and their source
hashes are retained in `reports/data_access_20261001/epic_prefix_pair_review`.
This confirms those two examples; all 28 prefix matches retain their numerical
evidence, and full-source patient identity remains unresolved.

A full CRC/decode/cross-source/internal overlap audit completed under
`data/processed/epic_images_v2` at 17:00 UTC on 2026-10-01.
The unstarted v1 audit is preserved as superseded: v2 stores uint8 reference
thumbnails to reduce memory use while applying the same screen. This audit
performs no model inference/training and does not adopt the supplied partitions.

## Full overlap and label results

References comprise all 112,120 audited NIH images, 5,856 legacy acquisitions
and all 7,470 OpenI images. NIH fingerprints cover the NIH acquisitions underlying
RSNA; RSNA is not another independent institution in this comparison. The screen
accepts exact bytes/pixels, or pHash Hamming distance <=6 AND centered 64px
grayscale correlation >=0.995.

| Reference | Epic images with a match | Matching pairs |
|---|---:|---:|
| Legacy | 1,517 | 1,527 (64 exact bytes; 1,463 near) |
| NIH | 1,028 | 1,028 (near) |
| OpenI | 0 | 0 |
| Within Epic | — | 167 (73 exact bytes; 94 near) |

The 62 cross-partition pairs consist of 27 exact-byte and 35 near matches.
All 48 conflicting-label pairs are **exact-byte matches**, so these label
contradictions do not depend on a perceptual-similarity assumption. For example,
testing `pneumonia-695.jpg` and training `normal-612.jpg` contain identical bytes.
One within-training example is `normal-721.jpg` versus `pneumonia-392.jpg`.

There are only **81 images without a recorded cross-source match**. These are
unmatched under this conservative rule; they are not verified new patients,
verified hospital acquisitions or reliable clinical labels. No independent
hospital cohort or leakage-free patient split can be recovered from that count.

## Model-use decision

The visible description does not establish patient identifiers, patient-disjoint
partitions, repeat-acquisition handling, ages/views, specialist label review,
clinical confirmation or independent hospital provenance. The inconsistent
publisher split description does not repair the observed archive split leakage.
Repartitioning alone also cannot resolve identical files with conflicting
labels, missing patient identity or questionable independent-source provenance.

Keep this acquisition separate from model development. Do not score this source
to choose a candidate or threshold and then call it an untouched final test.
If patient identity cannot be verified, explicitly retain that limitation and
do not claim complete patient separation based only on duplicate screening.

## Retained local evidence

- `reports/data_access_20261001/epic_datacite_v5.json`: complete public DOI record.
- `reports/data_access_20261001/epic_browser_observation.json`: publisher-visible
  counts, file identities/checksums and separate failed API-route observation.
- `reports/data_access_20261001/epic_acquisition_status.json`: original failed
  acquisition, terminated at the publisher-page HTTP 403; no images acquired.
- `reports/data_access_20261001/epic_download_route_check.json`: separately
  checked published image and Download All routes, both HTTP 403.
- `reports/data_access_20261001/epic_cache_route_check.json`: an additional HEAD
  request to an inferred conventional provider-cache route also returned 403.
  This was not a publisher-observed redirect or a verified image source.

The initial normal browser download link produced no event within 15 seconds. The
documented API route on `api.data.mendeley.com` was also checked for the PDF and
returned HTTP 403. No account, API client or publisher correspondence was
created. Those failed attempts are retained as dated evidence; later browser
downloads did produce files and reveal a working publisher file URL.

- `reports/data_access_20261001/epic_note_inspection.json`: verified PDF and renders.
- `reports/data_access_20261001/epic_zip_directory_inventory.json`: pinned ZIP
  central directory metadata and count discrepancy.
- `reports/data_access_20261001/epic_partial_precheck_v2.json`: bounded member
  size/CRC/decode verification for 177 prefix images.
- `reports/data_access_20261001/epic_prefix_legacy_near_screen.json`: all 28
  prefix-to-legacy near-match pairs.
- `reports/data_access_20261001/epic_range_transfer.json`: completed disjoint-range
  transfer and verified whole-file publisher checksum.
- `data/processed/epic_images_v2`: registered audit, complete image inventory,
  thumbnails, every match pair and artifact hashes. No extracted image paths
  from the ZIP were written to the filesystem.
- `reports/data_access_20261001/epic_full_audit_readback.json`: separate full
  image byte/CRC/decode/pixel-hash and recorded-match consistency readback.

The private research helpers passed Python compilation and focused Ruff E9/F
checks. A full style/type lint run on the audit/prefix-visualization helpers
reported 49 issues (35 line-length, 12 annotation, one import-order and one
comparison-style issue), retained in `epic_helper_full_lint.json`. The registered
executed audit source is preserved unchanged. These style failures must not be
reported as a fully passing repository lint run.

The first separate readback verified all 2,626 image bytes/CRC/decodes/pixel
hashes but stopped on a numerical consistency check: three independently
computed float64 correlations differed from stored float32 results by more than
the initial 1e-6 tolerance, with maximum difference 2.19e-6. Those correlations
are all above 0.99994, comfortably above the unchanged 0.995 matching rule.
The original failed helper and failure/precision diagnosis are retained.
The revised readback allows 3e-6 numerical drift and retains the scientific
threshold; this does not add matches, alter splits or change any model result.
It passed at 17:08 UTC: all 2,626 image byte/CRC/decode/pixel-hash checks and all
2,722 recorded match pairs were recomputed. It also found 80 legacy
filename-derived patient proxies represented in both Epic partitions; these
proxies are not independently verified true patient IDs. Both deployed/original
checkpoint hashes still match their pre-audit values.
