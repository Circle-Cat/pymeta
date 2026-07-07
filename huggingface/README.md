---
pretty_name: PyMETA
license: cc-by-nc-4.0
language:
- en
- zh
annotations_creators:
- expert-generated
- machine-generated
language_creators:
- found
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- text-classification
task_ids:
- multi-class-classification
- multi-label-classification
tags:
- code
- python
- code-error-classification
- education
- online-judge
configs:
- config_name: default
  data_files:
  - split: train
    path: train.csv
  - split: validation
    path: dev.csv
  - split: test
    path: test.csv
---

# Dataset Card for PyMETA

## Dataset Description

- **Repository (code, prompts, scripts):** https://github.com/Circle-Cat/pymeta
- **Paper:** *PyMETA: A Benchmark Dataset for Hierarchical Student Code Error Classification with Python-Interpreter-Based Labels*
- **Point of Contact:** CircleCat (`{cyli, ztang}@circlecat.org`)

### Dataset Summary

**PyMETA** (**Py**thon **M**ulti-**E**rror **TA**xonomy) is a large-scale benchmark for
**hierarchical student code error classification**. It contains **48,646** real Python
code submissions from **579** users across **155** distinct problems (22 problem types),
collected from the Circle Cat online learning platform. Every submission has a
**single-error label** derived from Online Judge execution output, organized under a
**three-level hierarchical taxonomy** (binary → three-class → 14 fine-grained error types
grounded in Python's official exception hierarchy).

An **expert-annotated multi-error diagnostic subset** of 97 samples (13 error types) is
provided in the [GitHub repository](https://github.com/Circle-Cat/pymeta) for studying
co-occurring errors.

### Supported Tasks

- **Task A — Binary classification:** `No Error` vs. `Error`.
- **Task B — Three-class classification:** `No Error` / `Logic Error` / `Explicit Error`.
- **Task C — Multi-class classification:** 14 fine-grained error types.
- **Multi-error classification** (on the 97-sample subset): identify *all* concurrent
  errors in a submission.

### Languages

Student code is **Python**. Problem descriptions are primarily in **Chinese** (`zh`); the
evaluation prompts are in **English** (`en`).

## How to use

```python
from datasets import load_dataset

ds = load_dataset("CircleCat/pymeta")
print(ds["train"][0])
```

## Dataset Structure

### Data Fields

Each sample has 9 features plus a single-error label:

| Field | Type | Description |
|-------|------|-------------|
| `userId` | string | Anonymized numeric identifier of the student |
| `name` | string | Problem/lesson name (problem type) |
| `questionId` | string | Unique identifier of the problem |
| `question` | string | Problem description |
| `exceptedAnswer` | string | A correct reference code solution |
| `attemptId` | string | Attempt number |
| `studentAnswer` | string | The student's submitted Python code |
| `testOutcome` | string | Online Judge error message / execution output |
| `attemptstepid` | string | Step identifier for the attempt |
| `error_category` | string | Single-error label (see taxonomy below) |

### Taxonomy (label IDs)

`0` No Error · `1` Logic Error · `2` Syntax Error · `3` Name Error · `4` Type Error ·
`5` Indentation Error · `6` Unbound Local Error · `7` Key Error · `8` Index Error ·
`9` EOF Error · `10` Recursion Error · `11` Value Error · `12` Tab Error ·
`13` Other Errors. Full definitions and per-class counts are in the
[GitHub `TAXONOMY.md`](https://github.com/Circle-Cat/pymeta/blob/main/TAXONOMY.md).

### Data Splits

| Split | Examples |
|-------|---------:|
| train | 39,402 |
| validation | 4,379 |
| test | 4,865 |
| **total** | **48,646** |

## Source Data

Submissions were collected from historical logs of the Circle Cat online learning
platform (a self-hosted Moodle instance with an integrated Online Judge). They were
generated organically by learners of varying proficiency during ordinary coursework, so
they provide a realistic and diverse distribution of student code errors.

## Personal and Sensitive Information

The data were collected from pre-existing educational records under the platform's terms
of use (which inform users that anonymized data may be used for educational and research
purposes), with no recruitment or experimental intervention. **All identifiers have been
irreversibly anonymized**; the dataset contains **no personally identifiable information
(PII)** and no offensive content.

## Licensing Information

Released under the
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
license.

## Citation

```bibtex
@misc{li2026pymeta,
  title        = {PyMETA: A Benchmark Dataset for Hierarchical Student Code Error
                  Classification with Python-Interpreter-Based Labels},
  author       = {Li, Chuyue and Tang, Ziqi and Wang, Jingyi and Wu, Yu and
                  Hashimoto, Kazuma and Gao, Lingyu},
  year         = {2026},
  howpublished = {\url{https://github.com/Circle-Cat/pymeta}},
  note         = {CircleCat}
}
```

<!-- TODO: replace with the final arXiv / ACL citation once available. -->
