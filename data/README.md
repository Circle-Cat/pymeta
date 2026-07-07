# PyMETA Data

> 📦 **The data lives on Hugging Face, not in this repo:**
> **https://huggingface.co/datasets/CircleCat/pymeta**

The split CSVs are large (train is ~110 MB), so — following the pattern of datasets like
[`allenai/c4`](https://huggingface.co/datasets/allenai/c4) — the data is hosted on Hugging
Face Datasets, and this GitHub repo holds only the documentation, prompts, and scripts.

## Download

```python
from datasets import load_dataset
ds = load_dataset("CircleCat/pymeta")
ds["train"], ds["validation"], ds["test"]
```

Or grab the raw CSVs directly from the "Files and versions" tab of the Hugging Face repo.

## Schema

Each sample has 9 features plus a single-error label:

| Column | Description |
|--------|-------------|
| `userId` | Anonymized numeric identifier of the student |
| `name` | Problem/lesson name (problem type) |
| `questionId` | Unique identifier of the problem |
| `question` | Problem description |
| `exceptedAnswer` | A correct reference code solution |
| `attemptId` | Attempt number |
| `studentAnswer` | The student's submitted code |
| `testOutcome` | Online Judge error message / execution output |
| `attemptstepid` | Step identifier for the attempt |
| `error_category` | Single-error label (see [`../TAXONOMY.md`](../TAXONOMY.md)) |

All identifiers are irreversibly anonymized; the dataset contains no PII.

## Splits

| Split | Examples |
|-------|---------:|
| train | 39,402 |
| validation (`dev`) | 4,379 |
| test | 4,865 |
| **total** | **48,646** |

> Note: the source split files are named `fixed_splitted_data_{train,dev,test}_data_20250504_083254.csv`.
> On Hugging Face they are exposed as `train.csv`, `dev.csv`, `test.csv` (the `dev` split is
> surfaced as `validation` by `load_dataset`).
