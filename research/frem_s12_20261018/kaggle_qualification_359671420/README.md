# FREM-S12 Kaggle qualification 359671420

The private Kaggle saved version `359671420` was launched on 18 October 2026
with accelerator None and internet disabled. Kaggle finalized the run as a
failure after 19.9 seconds and exposed one output file,
`frem-s12-output/preflight.json` (261 B).

`preflight_observed.json` is a semantic transcription from Kaggle's signed
output preview. It is not represented as a byte-for-byte downloaded producer
artifact. `kaggle_output_proof.png` is the retained UI proof with SHA-256
`159b642a196cd53f53781f155e3c2a856621dca30cfb061c04e4db37b12f5e8d`.
The preview reports Python 3.13.15, `scientific_admitted: false`, and
no files at the notebook's expected input path. The notebook then raised:

```text
RuntimeError: Unsupported Kaggle Python 3.13.15; required 3.12; preflight receipt preserved
```

This occurred before wheel installation, dependency import, official archive
access, market parsing, search/fixed consumption, model fitting or protected
outcome access. The platform qualification is therefore rejected and the
scientific status remains `NOT_EVALUATED`. Do not retry this unchanged saved
version and do not weaken the frozen runtime requirement to manufacture an
admission.

Kaggle notebook:
<https://www.kaggle.com/code/ryangomez010/frem-s12-campaign?scriptVersionId=359671420>

Private input dataset:
<https://www.kaggle.com/datasets/ryangomez010/frem-s12-input>
