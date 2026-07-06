# PyMETA

**PyMETA** (**Py**thon **M**ulti-**E**rror **TA**xonomy) is a large-scale benchmark
dataset for **hierarchical student code error classification**, with labels grounded in
Python's official exception hierarchy and Online Judge execution output.

This repository accompanies the paper *"PyMETA: A Benchmark Dataset for Hierarchical
Student Code Error Classification with Python-Interpreter-Based Labels"*. It hosts the
documentation, annotation guidelines, prompt templates, and evaluation scripts.

> 📦 **The dataset itself is hosted on Hugging Face:**
> **https://huggingface.co/datasets/CircleCat/pymeta**
> ```python
> from datasets import load_dataset
> ds = load_dataset("CircleCat/pymeta")
> ```

## Highlights

- **48,646** real Python student code submissions from **579** users across **155**
  distinct problems (**22** problem types), collected from the Circle Cat online learning
  platform.
- **Single-error labels** for every submission, derived from Online Judge execution
  signals.
- An **expert-annotated multi-error diagnostic subset** of **97 samples** (13 error
  types, avg. 1.91 errors/sample) for studying co-occurring errors.
- A **three-level hierarchical taxonomy** (binary → three-class → 14 fine-grained types).
- Reproducible **prompt templates** and **evaluation scripts** for single-error and
  multi-error classification.

## Repository structure

```
pymeta/
├── README.md                     # this file
├── LICENSE                       # CC BY-NC 4.0
├── TAXONOMY.md                   # full three-level taxonomy + label IDs
├── data/
│   └── README.md                 # schema + link to the Hugging Face dataset
├── huggingface/
│   └── README.md                 # the Hugging Face dataset card (upload with the CSVs)
├── annotation/
│   ├── multi_error_subset_97.csv # 97-sample expert multi-error labels
│   ├── ANNOTATION_GUIDELINES.md  # how the subset was annotated
│   └── error_type_dist_97.png    # error-type distribution of the subset
├── prompts/
│   ├── single_error_prompt.txt   # single-error classification prompt
│   └── multi_error_prompt.txt    # multi-error (Chain-of-Thought) prompt
└── scripts/
    ├── single_error_prompting.py # single-error prompting driver
    ├── multi_error_prompting.py  # multi-error prompting driver
    └── multi_error_metrics.py    # F1 / exact-match metrics
```

## Dataset

Each sample has 9 features (`userId`, `name`, `questionId`, `question`, `exceptedAnswer`,
`attemptId`, `studentAnswer`, `testOutcome`, `attemptstepid`) plus a single-error label
`error_category`. The dataset (48,646 submissions: 39,402 train / 4,379 validation / 4,865
test) is hosted on **[Hugging Face](https://huggingface.co/datasets/CircleCat/pymeta)**.
See [`data/README.md`](data/README.md) for the full schema and [`TAXONOMY.md`](TAXONOMY.md)
for label definitions.

## Tasks

PyMETA supports three nested classification tasks:

| Task | Description | Classes |
|------|-------------|---------|
| A | Binary | No Error / Error |
| B | Three-class | No Error / Logic Error / Explicit Error |
| C | Multi-class | 14 fine-grained error types |

Plus a **multi-error** task on the 97-sample diagnostic subset (identify *all* concurrent
errors per submission).

## Usage

The evaluation scripts read an API key from an environment variable (never hard-coded):

```bash
export OPENAI_API_KEY=...     # or GEMINI_API_KEY / DEEPSEEK_API_KEY as appropriate

python scripts/single_error_prompting.py --data ./data/test.csv --output ./results
python scripts/multi_error_prompting.py  --data ./annotation/multi_error_subset_97.csv --output ./results
python scripts/multi_error_metrics.py    --predictions ./results/preds.csv
```

The exact prompts sent to the models are in [`prompts/`](prompts/).

## Ethics & privacy

The data were collected from historical platform logs during ordinary coursework, with no
recruitment or experimental intervention, under the platform's terms of use (which inform
users that anonymized data may be used for educational and research purposes). All
identifiers have been **irreversibly anonymized**; the dataset contains **no personally
identifiable information (PII)** and no offensive content.

## License

This dataset is released under the
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE)
license.

## Citation

If you use PyMETA, please cite:

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

<!-- TODO: replace with the final arXiv / ACL citation (venue, arXiv id, DOI) once available. -->
