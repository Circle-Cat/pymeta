# PyMETA Data

> 📦 **The data lives on Hugging Face, not in this repo:**
> **https://huggingface.co/datasets/CircleCat/pymeta**

The split CSVs are large (train is ~110 MB), so — following the pattern of datasets like
[`allenai/c4`](https://huggingface.co/datasets/allenai/c4) — the data is hosted on Hugging
Face Datasets, and this GitHub repo holds only the documentation, prompts, and scripts.

## Download

```python
from datasets import load_dataset
ds = load_dataset("CircleCat/pymeta")                      # v2.0: question-ID split
ds_v1 = load_dataset("CircleCat/pymeta", revision="v1.0")  # v1.0: random split
ds["train"], ds["validation"], ds["test"]
```

`scripts/bootstrap.sh` downloads the **v1.0** CSVs into this folder, because the prompting
scripts and the 97-sample subset index refer to row positions in the v1.0 test split.

Or grab the raw CSVs directly from the "Files and versions" tab of the Hugging Face repo.

## Schema

The core columns are listed below. The
[dataset card](https://huggingface.co/datasets/CircleCat/pymeta) documents all of them,
including the judge status, traceback and error-type columns and, from v2.0, the starter
code (`answer_preload`).

| Column | Description |
|--------|-------------|
| `userId` | Anonymized numeric identifier of the student |
| `name` | Lesson / quiz name in the course |
| `questionId` | Question identifier (153 values; some problems appear under several IDs) |
| `question` | Problem description |
| `exceptedAnswer` | A correct reference code solution |
| `attemptId` | Question-attempt identifier; one attempt can contain several submissions |
| `studentAnswer` | The student's submitted code |
| `testOutcome` | Raw Online Judge (CodeRunner) outcome |
| `attemptstepid` | Submission identifier, unique per row |
| `all_errortype2` | Source of the Task B/C labels (see [`../TAXONOMY.md`](../TAXONOMY.md)) |
| `error_category` | Task A label: `error` / `no_error` |

All identifiers are irreversibly anonymized; the dataset contains no PII.

## Splits

Both versions contain the same 48,646 rows; only the split assignment differs.

| Split | v2.0 (question-ID split) | Question IDs | v1.0 (random split) |
|-------|---------:|---------:|---------:|
| train | 38,918 | 133 | 39,402 |
| validation (`dev`) | 4,863 | 11 | 4,379 |
| test | 4,865 | 9 | 4,865 |
| **total** | **48,646** | **153** | **48,646** |

> Note: the files are named `train.csv`, `dev.csv`, `test.csv` on Hugging Face (the `dev`
> split is surfaced as `validation` by `load_dataset`). The two test splits have the same
> size but are different sets of rows.
