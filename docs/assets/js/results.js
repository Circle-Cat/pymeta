/* PyMETA project page — all numbers rendered on the page live here.
   Sources: camera-ready paper (NeurIPS 2026 TAE workshop) tables, and the
   Hugging Face v2.0 release. Edit this file to update charts and tables. */
window.PYMETA = {
  // Table 2 — single-error Task C on the 4,865-item test set (%)
  rq1: [
    { model: "CodeBERT",               acc: 73.4, f1: 45.2, group: "ft"  },
    { model: "CodeLlama-7B",           acc: 91.7, f1: 80.6, group: "ft"  },
    { model: "GPT-3.5",                acc: 40.3, f1:  8.8, group: "old" },
    { model: "GPT-4o",                 acc: 71.5, f1: 28.0, group: "old" },
    { model: "DeepSeek-V3",            acc: 73.6, f1: 29.1, group: "old" },
    { model: "Gemini 2.5 Pro",         acc: 85.9, f1: 71.9, group: "old" },
    { model: "DeepSeek-V4-Pro",        acc: 91.3, f1: 87.5, group: "new" },
    { model: "Claude Opus 5",          acc: 93.3, f1: 90.8, group: "new" },
    { model: "Gemini 3.1 Pro Preview", acc: 92.1, f1: 91.2, group: "new" },
    { model: "GPT-5.5",                acc: 93.8, f1: 93.8, group: "new" }
  ],
  bestFinetuned: 80.6,

  // Table 3 — Logic Error over-prediction rate, single-error prompting (%)
  lerover: [
    { model: "GPT-3.5",                v: 92.80, group: "old" },
    { model: "GPT-4o",                 v: 57.20, group: "old" },
    { model: "DeepSeek-V3",            v: 51.50, group: "old" },
    { model: "Gemini 2.5 Pro",         v: 17.60, group: "old" },
    { model: "DeepSeek-V4-Pro",        v:  9.89, group: "new" },
    { model: "Claude Opus 5",          v:  6.38, group: "new" },
    { model: "Gemini 3.1 Pro Preview", v:  4.79, group: "new" },
    { model: "GPT-5.5",                v:  4.18, group: "new" }
  ],

  // Table 1 — Task C label distribution, full dataset
  labelDist: [
    ["No Error", 23207], ["Logic Error", 11387], ["Syntax Error", 5618], ["Name Error", 2565],
    ["Type Error", 2074], ["Indentation Error", 1355], ["Unbound Local Error", 482], ["Key Error", 417],
    ["Index Error", 400], ["Recursion Error", 334], ["EOF Error", 277], ["Value Error", 190],
    ["Other Errors", 178], ["Tab Error", 162]
  ],

  // Hugging Face v2.0 release — question-ID split (counted from the released CSVs)
  splits: [
    { split: "train",      rows: 38918, qids: 133, errRate: 49.1 },
    { split: "validation", rows:  4863, qids:  11, errRate: 61.4 },
    { split: "test",       rows:  4865, qids:   9, errRate: 69.0 }
  ],

  // Figure 1 — coverage of CodeBERT's prediction by the expert repair-conditional set,
  // by predicted label, for the two targeted samples
  coverage: {
    a: { title: "Misclassification-based sample", n: 40, overall: 50,
         rows: [["Name Error", 1.00], ["Logic Error", 0.71], ["Type Error", 0.57], ["Index Error", 0.33], ["Syntax Error", 0.33],
                ["EOF Error", 0], ["Indentation Error", 0], ["Key Error", 0], ["Tab Error", 0], ["Unbound Local Error", 0]] },
    b: { title: "High-entropy sample", n: 60, overall: 52,
         rows: [["Logic Error", 1.00], ["Name Error", 1.00], ["Syntax Error", 0.59], ["Key Error", 0.50],
                ["Type Error", 0], ["EOF Error", 0], ["Indentation Error", 0], ["No Error", 0]] }
  },

  // 97-item expert subset — labels per Task C class (187 labels, 13 classes)
  subsetDist: [
    ["Logic Error", 52], ["Syntax Error", 46], ["Type Error", 28], ["Name Error", 21], ["Indentation Error", 9],
    ["Unbound Local Error", 7], ["Index Error", 5], ["Recursion Error", 5], ["Value Error", 4], ["Other Errors", 4],
    ["EOF Error", 3], ["Key Error", 2], ["Tab Error", 1]
  ],

  // Table 4 — multi-error set metrics on the 97-item expert subset (%)
  rq3: [
    { model: "GPT-5.5",                exact: 47.42, sampleF1: 80.84, microF1: 80.94, macroF1: 70.22 },
    { model: "Gemini 3.1 Pro Preview", exact: 48.45, sampleF1: 81.06, microF1: 80.51, macroF1: 67.37 },
    { model: "Claude Opus 5",          exact: 43.30, sampleF1: 79.71, microF1: 78.95, macroF1: 64.06 },
    { model: "DeepSeek-V4-Pro",        exact: 44.33, sampleF1: 78.73, microF1: 78.44, macroF1: 64.06 }
  ],

  // Table 5 — Logic Error on the 45 audited submissions with no expert Logic Error
  le45: [
    { model: "GPT-5.5",                single: 0, any: 21, first: 0 },
    { model: "Gemini 3.1 Pro Preview", single: 0, any: 24, first: 4 },
    { model: "Claude Opus 5",          single: 0, any: 22, first: 0 },
    { model: "DeepSeek-V4-Pro",        single: 0, any: 26, first: 0 }
  ],

  bibtex: `@inproceedings{tang2026pymeta,
  title     = {Py{META}: Evaluating Student Code Diagnosis on and Beyond the First Execution Error},
  author    = {Ziqi Tang and Chuyue Li and Jingyi Wang and Yu Wu and Kazuma Hashimoto and Lingyu Gao},
  booktitle = {NeurIPS 2026 Workshop on Trust-AI-Eval (TAE): Can We Trust AI Evaluation?},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Fz4nqncIlh}
}`
};
