# Reproduction

Three levels, from open to authorised:

1. **Synthetic demonstration** (`vetnutrilm demo`): runs grouping, folds, parsing and metrics on invented notes. No clinical data or GPU.
2. **Published-figure check** (`vetnutrilm reproduce`): needs the saved run outputs, held privately. Each prediction file is checked against its completion hash, then all 30 published figures are recomputed: binary F1, sensitivity, precision, specificity and accuracy with their intervals; the pretrained comparison and the fine-tuning difference interval; category macro and micro F1 with intervals; and per-category F1. Expected values are in `src/vetnutrilm/published_results.json`.
3. **Retraining** (`vetnutrilm train`): needs the labelled notes and a GPU. The recipe, prompts and seed are fixed, but GPU arithmetic is not bit-for-bit deterministic, so retrained predictions can differ slightly. Retraining is a new run, not a replacement for the published predictions.

## Verification record (25 September 2026)

- `reproduce` against the published run outputs: 30 of 30 figures matched.
- `folds.binary_folds` reproduced the published binary allocation exactly (1,002 records).
- `folds.category_folds` matched the original category allocator on five random datasets.
- The prompts match the published run packages (SHA-256 in `prompts.py`, checked by the tests).
- The GPU training and inference code is a direct port of the published runner.

## Method notes

- Intervals are percentile intervals from 2,000 bootstrap resamples of whole linked groups, seed 42. They are conditional on the fitted fold models and fixed folds.
- Measures pool the held-out predictions from all five folds, rather than averaging fold scores.
- Macro F1 is undefined in a resample where any category has no positives and no predictions; intervals are withheld if under 95% of resamples are defined.
