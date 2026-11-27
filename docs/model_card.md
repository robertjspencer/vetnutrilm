# Model card: VetNutriLM adapters

**Status: not yet released.** Release is subject to written approval by the data custodian (VetCompass Australia, University of Sydney).

## What they are

Two LoRA adapters (rank 8, 22,507,520 trainable parameters each) for `google/gemma-4-31B-it`, loaded in 4-bit NF4:

- **VetNutriLM-Binary**: returns `[1]` or `[0]` for whether a note records feeding.
- **VetNutriLM-FoodCategories**: returns six 0/1 values for commercial dry, commercial wet, raw, home-prepared cooked, therapeutic, and other *(the frozen training prompt and schema retain `Therapeutic / prescription` for reproducibility)*.

They are separate models trained on different reference sets with different rules, not one combined model.

## Intended use

Research: locating and organising feeding information in veterinary clinical notes for review by researchers. Outputs should be checked by a person before being relied on for any individual animal. **Not for clinical decisions.**

## Training data

Australian first-opinion (primary-care) canine consultation notes from VetCompass Australia, 2019 to 2024, drawn from a 5,000-record stratified candidate pool (Dataset B) within a 38,535-consultation lexicon-matched extract across 167 clinics (Dataset X). Binary: 1,002 notes (430 positive, 958 linked groups). Categories: 512 notes from 511 dogs (461 linked groups). Labels from one veterinarian with second-author quality-control review; no second-annotator agreement was measured.

## Performance

See the README. Internal cross-validation only; the folds also informed model choice.

## Known limitations

- Both reference sets were drawn from the lexicon-matched candidate corpus (Dataset X), so performance on notes without lexicon matches was not directly evaluated.
- Single-annotator reference; manual review of binary disagreements suggested possible reference-label problems, so reported error rates may partly reflect reference-label errors.
- Rare categories (raw, therapeutic) have wide intervals.
- Brand names, unspecified food forms, and the difference between feeding and advice are recurring sources of error.
- Untested outside Australian primary care, and on species other than dogs.

## Privacy

Training used answer-token loss only: the adapters were trained to output labels, not to reproduce note text. Release adapters are to be trained on de-identified notes, and a memorisation check (whether the adapter continues training notes word for word more often than the base model) will be reported here before release.
