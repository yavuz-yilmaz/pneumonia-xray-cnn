# Independent pediatric evaluation candidate

The official [VinDr-PCXR 1.0.0 page](https://physionet.org/content/vindr-pcxr/1.0.0/)
was inspected on 2026-09-16. It describes 9,125 pediatric studies from a hospital
in Vietnam (2020–2021), with 36 findings and 15 disease labels, split into 7,728
training and 1,397 test studies. This provides a more relevant pediatric source
than the adult RSNA transfer benchmark.

The official access policy says: “Only registered users who sign the specified
data use agreement can access the files.” The listed license and agreement are
PhysioNet Restricted Health Data License / Use Agreement 1.5.0. Access status has
been requested from the user; no restricted files were downloaded or agreement
accepted on the user's behalf.

The description says patient identifiers were removed during de-identification.
Consequently, access alone does not establish verifiable patient-disjointness.
Before using the data, inspect the supplied split guarantees and metadata, confirm
the pneumonia label definition and normal/control definition, and screen image
overlap. Preserve the official test separately. Do not tune thresholds on it.
Do not treat repackaged Kermany copies as independent pediatric validation.

Current status: potential external cohort identified; source files unavailable
locally, patient-disjoint guarantees and label compatibility not yet verified.
