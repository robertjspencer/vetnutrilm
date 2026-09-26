# Input formats

All inputs are UTF-8 CSV files.

| Column | Needed for | Meaning |
|---|---|---|
| `id` | all | Unique record identifier |
| `text` | all | Full clinical note, exactly as recorded |
| `patient` | optional | Patient key; records sharing it are kept in one fold |
| `label` | binary training and evaluation | 1 if the note records food the dog is or was fed (not stopped), else 0 |
| `Commercial dry`, `Commercial wet`, `Raw`, `Home-prepared cooked`, `Therapeutic / prescription`, `Other` | category training and evaluation | 0 or 1 per category; categories may overlap |

**Linked groups.** Records sharing a patient, or whose text is identical after ignoring case and repeated whitespace, form one group. `prepare` writes `id,group,fold`.

**Predictions** (written by `train` and `classify`): `id`, the raw model output, the parsed prediction (`binary` or the six category columns) and an `*_invalid` flag. `evaluate` refuses to score invalid binary outputs until they are reported and resolved.

## Label definitions

**Binary.** Yes if the note mentions what the dog is currently fed, or was previously fed with no mention of that stopping. No for: no food mentioned; advice or recommendations only; food the owner is only considering; supplements, chews or probiotics alone; a food word used as a name or a spelling error; a previous diet that has stopped with no current diet.

**Categories.** Current or past feeding, or food supplied at the visit, counts; advice alone and denied feeding do not. See the verbatim prompt in `src/vetnutrilm/prompts.py` for the rules given to the model.
