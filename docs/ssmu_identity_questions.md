# SSMU identity/label clarification — unsent draft

This draft is for the publisher of
[X-ray database: pneumonia and norm](https://doi.org/10.5281/zenodo.5732746).
No message has been sent. The request seeks anonymous grouping information;
patient names or other identifying details are not needed. Replying is not
assumed to grant a new data license.

## Verified public contact route

**Suggested recipient:** Vladimir D. Udodov, via the current publisher author section linked below.

The publisher's [2025 article author section](https://jdigitaldiagnostics.com/DD/article/view/633978)
lists this address, Siberian State Medical University affiliation and ORCID
`0000-0002-1321-7861`. The same ORCID appears for Udodov in the
[2021 dataset record](https://doi.org/10.5281/zenodo.5732746) and the
[2026 part 1 record](https://doi.org/10.5281/zenodo.18241541). This verifies the
published contact's identity linkage; current mailbox availability, a response
and willingness to supply grouping data have not been established. The address
was checked on 2026-10-02. Use the current publisher author section if it changes.

## Message

**Subject:** Anonymous patient grouping and labels for Zenodo dataset 5732746

Dear Dr. Udodov,

I am using your public CC BY 4.0 chest X-ray dataset for a personal,
non-commercial pneumonia-classification portfolio project. Thank you for making
the images available. I would like to prevent the same patient's examinations
from appearing in both training and evaluation.

Could you clarify these points?

1. Do the numeric filename stems identify patients, examinations, or image
   pairs? Numbering repeats between `norma` and `pneumonia`; can the same person
   occur under different numbers or in both folders?
2. Could you provide an anonymous filename-to-patient grouping table, or
   documented grouping rule covering repeated examinations? I do not need
   patient names, dates of birth or other identifying information.
3. How were normal and pneumonia labels established—for example, radiologist
   review, clinical diagnosis, CT confirmation or reports? Were the images
   reviewed independently, and were follow-up examinations included?
4. Are your 2026 pneumonia releases (Zenodo 18241541 and 18241637) drawn from
   patients or examinations already present in the 2021 release?

I have retained source attribution and kept images out of the public code
repository. I will describe filename groups as unverified proxies until their
meaning is established.

Kind regards,
[Your name]

## Why this information is needed

The archive has no patient/age/annotation table, and all 929 selected frontal
PNGs have empty PNG/EXIF metadata. Conservative grouping of equal numbers across
both classes and known duplicate pairs removes documented overlap, but cannot
establish that different numbers never refer to the same person. A stronger
identity claim requires source evidence rather than anatomical resemblance.
