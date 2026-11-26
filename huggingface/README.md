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
- **Paper:** *PyMETA: Evaluating Student Code Diagnosis on and Beyond the First Execution Error* (NeurIPS 2026 Trust-AI-Eval Workshop) — https://openreview.net/forum?id=Fz4nqncIlh
- **Point of Contact:** CircleCat (`{cyli, ztang}@circlecat.org`)

### Dataset Summary

**PyMETA** (**Py**thon **M**ulti-**E**rror **TA**xonomy) is a large-scale benchmark for
**hierarchical student code error classification**. It contains **48,646** real Python
code submissions from **579** users on **153** question IDs (75 distinct problems; several
problems were reused across course offerings under different IDs), collected from the
Circle Cat online learning platform. Every submission has a **single-error label** derived
from Online Judge execution output, organized under a **three-level hierarchical taxonomy**
(binary → three-class → 14 fine-grained error types grounded in Python's official exception
hierarchy).

An **expert-annotated multi-error diagnostic subset** of 97 samples (13 error types) is
provided in the [GitHub repository](https://github.com/Circle-Cat/pymeta) for studying
co-occurring errors; `metadata/multi_error_subset_97_ids.csv` in this repository locates
those samples by `attemptstepid`.

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

ds = load_dataset("CircleCat/pymeta")                    # v2.0: question-ID split (this card)
ds_v1 = load_dataset("CircleCat/pymeta", revision="v1.0")  # v1.0: original random split

# Build the labels for Tasks A/B/C from the label columns
LABEL_IDS = {
    "No error": 0, "LogicError": 1, "SyntaxError": 2, "NameError": 3, "TypeError": 4,
    "IndentationError": 5, "UnboundLocalError": 6, "KeyError": 7, "IndexError": 8,
    "EOFError": 9, "RecursionError": 10, "ValueError": 11, "TabError": 12,
}  # any other value -> 13 (Other Errors)

def add_labels(ex):
    c = LABEL_IDS.get(ex["all_errortype2"], 13)
    return {"task_a": int(c != 0), "task_b": 0 if c == 0 else (1 if c == 1 else 2), "task_c": c}

ds = ds.map(add_labels)
```

When showing a problem to a model, combine `question` with `answer_preload`: many
descriptions refer to "the code below" (以下代码), and that code is the starter code in
`answer_preload`, not part of `question`.

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `Unnamed: 0` | int | Row index in the original full dataset (0–48,645); unique and stable across versions. |
| `userId` | int | Numeric identifier of the student. |
| `name` | string | Lesson / quiz name in the course (71 values, e.g. `循环LAB3`). |
| `questionId` | int | Question identifier (153 values). Some problems appear under several IDs; see `metadata/question_groups.csv`. |
| `question` | string | Problem description as shown to students. Does **not** include the starter code (see `answer_preload`). |
| `answer_preload` | string | **New in v2.0.** Starter code pre-filled in the student's code editor when the question was opened (CodeRunner *answer preload*): input-reading scaffolds, fill-in-the-blank skeletons, or buggy code to fix for debugging questions. Empty (loaded as `None`) for the 26 questions without starter code. |
| `exceptedAnswer` | string | Reference solution (column name kept from v1.0). |
| `attemptId` | int | Question-attempt identifier; one attempt can contain several submissions (rows). |
| `state` | string | Platform state of the submission step: `todo`, `complete`, or `invalid`. |
| `studentAnswer` | string | The student's full submitted code. Because the editor starts with the starter code, this usually already contains `answer_preload` (possibly edited by the student). Uses `\r\n` line endings, while `answer_preload` uses `\n`. |
| `testOutcome` | string | Raw Online Judge (CodeRunner) testing outcome, PHP-serialized: per-test results, marks and error messages. |
| `attemptstepid` | int | Submission (attempt-step) identifier; unique per row across all splits and versions. |
| `status` | int | Judge status: `1` code ran, `2` compile-time error, `4` sandbox error (code was not executed). |
| `R_errorcount` | int | Number of failed test cases. |
| `R_traceback` | string | Runtime traceback (`No error` if none). |
| `R_errortype` | string | Runtime exception type (`No error` if none). |
| `C_errormessage` | string | Compile-time error message (`No error` if none). |
| `C_errortype` | string | Compile-time error type: `SyntaxError`, `IndentationError`, `TabError`, `SyntaxWarning`, or `No error`. |
| `all_errortype` | string | Exception type from either stage (`C_errortype` if present, otherwise `R_errortype`). Code that runs without an exception but fails tests is `No error` here. |
| `all_errortype2` | string | **Source of the Task B/C labels.** Same as `all_errortype`, except that submissions which raise no exception but fail at least one test are `LogicError`. The six rare types (`AttributeError`, `RuntimeError`, `SyntaxWarning`, `ZeroDivisionError`, `MemoryError`, `ModuleNotFoundError`; 178 rows) form `Other Errors`. |
| `error_category` | string | **Task A label:** `error` / `no_error`. |

### Taxonomy (label IDs)

`0` No Error · `1` Logic Error · `2` Syntax Error · `3` Name Error · `4` Type Error ·
`5` Indentation Error · `6` Unbound Local Error · `7` Key Error · `8` Index Error ·
`9` EOF Error · `10` Recursion Error · `11` Value Error · `12` Tab Error ·
`13` Other Errors. Task B groups IDs 2–13 as `Explicit Error`. Full definitions and
per-class counts are in the
[GitHub `TAXONOMY.md`](https://github.com/Circle-Cat/pymeta/blob/main/TAXONOMY.md).

### Data Splits

| Split | Examples | Question IDs | Error rate |
|-------|---------:|-------------:|-----------:|
| train | 38,918 | 133 | 49.1% |
| validation | 4,863 | 11 | 61.4% |
| test | 4,865 | 9 | 69.0% |
| **total** | **48,646** | **153** | 52.3% |

Since v2.0 the splits are made at the **question-ID level**: no `questionId` and no
`attemptId` appears in more than one split. All 14 Task C classes occur in test; the
validation split has no `KeyError` samples. Label distributions differ between splits.

Note that the course reused some problems under several question IDs (153 IDs correspond to
75 distinct problems, matched by identical description or reference solution). As a result,
all 9 test question IDs and 7 of the 11 validation question IDs have a duplicate problem in
train (100% of test rows, 82.7% of validation rows), and most students appear in every
split. This does not affect zero-shot prompting evaluation, but matters for models trained
on `train`. For problem-disjoint experiments, re-split using the `problem_group` column of
[`metadata/question_groups.csv`](metadata/question_groups.csv).

### Metadata files

- `metadata/question_groups.csv`: `questionId` → `problem_group` (75 groups), with each
  ID's split, row count and lesson name.
- `metadata/multi_error_subset_97_ids.csv`: the 97 expert-annotated multi-error samples with
  their `attemptstepid`, `questionId`, `attemptId`, `userId`, gold single label, annotated
  labels, and v2.0 split (74 train, 6 validation, 17 test). `Sample_Index` is the original
  position in the **v1.0** test split, as used by the GitHub scripts.

## Known Issues

These come from the original course material or the judge and are documented rather than
changed, so that every row's content and labels are identical to v1.0.

- **Questions 215, 241 (train) and 519 (validation), 134 rows:** `question` and
  `exceptedAnswer` ask for `score_dict` / `status_dict`, while `answer_preload` and almost all
  student answers print `my_dict[test_param]`. The problem appears to have been edited during
  the course; the description and reference solution reflect a later version than the one
  students solved.
- **Question 236 (train, 74 rows):** the starter code uses `input().split('')`, which always
  raises `ValueError: empty separator`. 23 rows labelled `ValueError` come from this
  starter-code line rather than from student-written code.
- **Question 2923 (test, 17 rows):** the example outputs in the description (`2` for
  `1 2 3`, `7` for `6 7 8`) contradict the explanations and the reference solution, which
  compute the sum (`6`, `21`).
- **Sandbox errors (5 rows in train, `status == 4`, `state == invalid`):** the code was not
  executed because the judge sandbox was unavailable, but the rows are labelled `No error`
  (`attemptstepid` 39969, 234690, 234936, 234945, 235723).
- **Formatting artifacts in `question`:** 9 descriptions contain hint text rendered with
  combining strikethrough characters (U+0336), and a few contain leftover HTML artifacts
  such as `&nbsp;` or code-block line numbers.

## Source Data

Submissions were collected from historical logs of the Circle Cat online learning
platform (a self-hosted Moodle instance with an integrated Online Judge). They were
generated organically by learners of varying proficiency during ordinary coursework, so
they provide a realistic and diverse distribution of student code errors.

## Personal and Sensitive Information

The data were collected from pre-existing educational records under the platform's terms
of use (which inform users that anonymized data may be used for educational and research
purposes), with no recruitment or experimental intervention. The dataset contains no
names, e-mail addresses or other direct identifiers; students appear only as numeric
`userId`s. The dataset contains no offensive content.

## Versions

- **v2.0 (current):** question-ID-level re-split (made in February 2026 in response to
  review); added the `answer_preload` column and the `metadata/` files; documented all
  columns, the label derivation and known issues. Row contents are identical to v1.0;
  only the split assignment changed.
- **v1.0** (tag `v1.0`, commit `2279d02`): random submission-level split stratified by
  `error_category` (39,402 / 4,379 / 4,865).

## Licensing Information

Released under the
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
license.

## Citation

```bibtex
@inproceedings{
    tang2026pymeta,
    title={Py{META}: Evaluating Student Code Diagnosis on and Beyond the First Execution Error},
    author={Ziqi Tang and Chuyue Li and Jingyi Wang and Yu Wu and Kazuma Hashimoto and Lingyu Gao},
    booktitle={NeurIPS 2026 Trust-AI-Eval Workshop},
    year={2026},
    url={https://openreview.net/forum?id=Fz4nqncIlh}
}
```
