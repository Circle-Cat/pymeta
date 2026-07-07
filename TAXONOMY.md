# PyMETA Error Taxonomy

PyMETA uses a **three-level hierarchical taxonomy**, from a coarse binary split down to
14 fine-grained error types grounded in Python's official exception hierarchy. All labels
are derived from Online Judge execution output, including `Logic Error` for code that
executes but fails test cases.

## Three-Level Hierarchy

| Task A (binary) | Task B (three-class) | Task C (multi-class)   | Count  |
|-----------------|----------------------|------------------------|--------|
| No Error        | No Error             | No Error               | 23,207 |
| Error           | Logic Error          | Logic Error            | 11,387 |
| Error           | Explicit Error       | Syntax Error           | 5,618  |
| Error           | Explicit Error       | Name Error             | 2,565  |
| Error           | Explicit Error       | Type Error             | 2,074  |
| Error           | Explicit Error       | Indentation Error      | 1,355  |
| Error           | Explicit Error       | Unbound Local Error    | 482    |
| Error           | Explicit Error       | Key Error              | 417    |
| Error           | Explicit Error       | Index Error            | 400    |
| Error           | Explicit Error       | EOF Error              | 277    |
| Error           | Explicit Error       | Recursion Error        | 334    |
| Error           | Explicit Error       | Value Error            | 190    |
| Error           | Explicit Error       | Tab Error              | 162    |
| Error           | Explicit Error       | Other Errors           | 178    |

*Counts reflect the full 48,646-submission dataset.*

- **Task A — Binary:** No Error vs. Error.
- **Task B — Three-class:** No Error / Logic Error / Explicit Error.
- **Task C — Multi-class:** the 14 fine-grained categories above.

## Label IDs (used in prompts and predictions)

| ID | Label | Definition |
|----|-------|------------|
| 0  | No Error | Code compiles and passes all test cases successfully. |
| 1  | Logic Error | Code compiles but fails one or more test cases due to incorrect logic. |
| 2  | Syntax Error | Parser encounters a syntax error in the code. |
| 3  | Name Error | A local or global name is not found. |
| 4  | Type Error | An operation or function is applied to an object of inappropriate type. |
| 5  | Indentation Error | Incorrect indentation in the code. |
| 6  | Unbound Local Error | A local variable is referenced before assignment. |
| 7  | Key Error | A dictionary key is not found. |
| 8  | Index Error | A sequence subscript is out of range. |
| 9  | EOF Error | `input()` hits an end-of-file condition without reading data. |
| 10 | Recursion Error | Maximum recursion depth exceeded. |
| 11 | Value Error | Argument has the correct type but an inappropriate value. |
| 12 | Tab Error | Indentation inconsistently mixes tabs and spaces. |
| 13 | Other Errors | AttributeError, RuntimeError, SyntaxWarning, ZeroDivisionError, MemoryError, ModuleNotFoundError, etc. |
