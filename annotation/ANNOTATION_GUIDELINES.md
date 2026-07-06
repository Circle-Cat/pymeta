# PyMETA Multi-Error Annotation Guidelines

These are the guidelines used to build the **expert-annotated multi-error diagnostic
subset** (`multi_error_subset_97.csv`): 97 samples covering 13 error types, with an
average of 1.91 error types per sample.

## Annotators

Annotation was performed by a team of 15 volunteer researchers familiar with Python
programming and educational assessment (software engineers and researchers with
backgrounds in computer science and educational assessment). All annotators had 3–5
years of Python engineering experience and were familiar with common introductory-level
student programming errors.

## Taxonomy

Annotators used an error taxonomy of 18 error types plus a "No Error" category (ID 0).
Each error type is defined by its associated runtime or compile-time behavior:

- **Logic Error (ID 1)** applies when code executes without explicit exceptions but
  fails one or more test cases.
- All other error types (IDs 2–18) are identified by matching the execution output
  against a specific error string (e.g., `SyntaxError`, `NameError`, `TypeError`).

See [`../TAXONOMY.md`](../TAXONOMY.md) for the full label definitions.

## Annotation Procedure

For each student code sample, identified by a *(Question ID, Attempt ID)* pair,
annotators followed this procedure:

1. Locate the specific programming problem on the online judge using the **Question ID**.
2. Paste the student's answer into the judge, observe the **first** error, and increment
   the corresponding error-type count in the annotation spreadsheet.
3. Fix **only** that single error (referring to the expected answer as a reference), and
   verify that re-execution no longer produces the same error at the same location.
4. Repeat steps 2–3 until no explicit runtime or compile-time errors remain.
5. If the problem includes test cases, run them: failure to pass all test cases indicates
   a **Logic Error**; passing all test cases marks the sample as complete.

## Output Format

The resulting labels are stored in `multi_error_subset_97.csv` with two columns:

| Column | Description |
|--------|-------------|
| `Sample_Index` | Index of the sample in the source dataset |
| `True Label` | All annotated error types, e.g. `"LogicError (#1), TypeError (#4)"` |

The number in `(#n)` is the taxonomy label ID (see `TAXONOMY.md`).

## Error Type Distribution

The distribution of error types across the 97 annotated samples is shown in
[`error_type_dist_97.png`](error_type_dist_97.png).
