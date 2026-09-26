# VetNutriLM

**What this is, in plain terms.** VetNutriLM finds nutrition information in the free-text notes vets write during consultations. It is two separately trained add-ons (adapters) for Google's Gemma-4-31B-it language model:

- **VetNutriLM-Binary** answers yes or no: does the note record what the dog is currently fed, or was fed without stopping?
- **VetNutriLM-FoodCategories** marks six overlapping food types: commercial dry, commercial wet, raw, home-prepared cooked, therapeutic or prescription, and other.

This repository holds the code, prompts and settings behind the published results, a command-line tool to train or run either classifier, and a check that recomputes every published figure. It contains **no clinical notes, labels, predictions or adapter weights**.

## Published results (internal validation)

Grouped five-fold cross-validation on Australian primary-care notes from VetCompass Australia. Notes from the same dog, or with identical text, were kept in the same fold. 95% intervals resample whole linked groups (2,000 replicates).

| Task | Notes | Result |
|---|---:|---|
| Binary | 1,002 (430 positive) | F1 94.1% (92.3 to 95.7); sensitivity 95.8%; precision 92.4% |
| Food categories | 512 | Macro F1 92.7% (90.2 to 94.7); micro F1 93.9% |

**Scope.** These are development results: the same folds informed the choice of Gemma. The 1,002 binary reference notes comprised 1,000 records from the nutrition-candidate extract (Dataset B) and two from the counterfactual extract (Dataset C). Performance in other clinics, referral hospitals and countries is untested.

## Install

Python 3.11 or later.

```sh
python -m pip install .            # data handling, folds, metrics, demo
python -m pip install ".[gpu]"     # adds the pinned GPU stack for training and inference
```

Training and inference need a local copy of `google/gemma-4-31B-it` and one GPU with about 80 GB of memory (the published runs used an NVIDIA H100). Use of Gemma is subject to Google's Gemma terms.

## Use

```sh
vetnutrilm demo --output runs/demo-01
vetnutrilm prepare  --task binary --input labelled.csv --output folds.csv
vetnutrilm train    --task binary --input labelled.csv --folds folds.csv --fold 1 --checkpoint /models/gemma-4-31B-it --output runs/fold1
vetnutrilm classify --task both --input notes.csv --checkpoint /models/gemma-4-31B-it --binary-adapter A --category-adapter B --output runs/predict-01
vetnutrilm evaluate --task binary --records labelled.csv --predictions runs/fold1/predictions.csv --output runs/fold1/metrics.json
```

- Use a new output directory for every run; the tool refuses to overwrite.
- Notes are never truncated or cleaned. Inputs longer than the tested 2,322 tokens are refused rather than cut.
- Outputs that are not exactly the expected JSON answer are marked invalid and reported separately.
- The demo uses invented notes. Its output is not an accuracy estimate.

Input formats are in [docs/inputs.md](docs/inputs.md). Slurm templates are in [slurm/](slurm).

## Reproduce the published figures

With authorised access to the saved run outputs:

```sh
vetnutrilm reproduce --binary-dataset B/dataset.json --binary-results B/results \
                     --category-dataset C/dataset.json --category-results C/results
```

All 30 published Gemma figures (point estimates and intervals) must match; any difference exits with an error. See [docs/reproduction.md](docs/reproduction.md).

## Adapters

The trained adapters will be released separately, subject to approval by the data custodian, because they were trained on clinical text. See the [model card](docs/model_card.md) and [data access](docs/data_access.md).

## Tests

```sh
python -m pip install ".[test]"
python -m pytest
```

The reproduction test runs only when the four `VETNUTRILM_*` paths to authorised run outputs are set.

## Licence and citation

Code: Apache-2.0. Adapters, when released, are subject to the Gemma terms and the release conditions in the model card. Please cite the VetNutriLM article (reference to be added on publication).
