# CLAUDE.md

Guidance for working in this repo. It records decisions and failure modes that cost
real time or money to discover — read it before touching the experiment scripts.

## What this repo is

Code and documentation for the PyMETA paper (student Python error classification).
The dataset lives on Hugging Face (`CircleCat/pymeta`), not here. What lives here:
the taxonomy, the annotation guidelines, the prompt templates, the evaluation
drivers, and small committable run summaries.

Current state: the 2026 re-run with current-generation models is in progress. One
model per machine; `deepseek-v4-pro` is done on both tasks (see
`results_summary/`). Claude / GPT / Gemini are pending.

## Setup

```bash
bash scripts/bootstrap.sh        # venv + pinned deps + test split + 97-sample subset
```

Idempotent. Never commit `data/*.csv`, `results/`, or `.venv/` — all gitignored.

## The prompts are frozen

`prompt_template` in both drivers and the files in `prompts/` are fixed
experimental apparatus. Changing them invalidates comparison with the published
numbers, so they must not be reformatted, "improved", or reflowed — not even
whitespace. When refactoring around them, verify the rendered prompt is unchanged:

```bash
python - <<'PY'
import hashlib, re, subprocess
for p in ("scripts/single_error_prompting.py", "scripts/multi_error_prompting.py"):
    head = subprocess.run(["git","show",f"HEAD:{p}"],capture_output=True,text=True).stdout
    work = open(p, encoding="utf-8").read()
    h = lambda t: hashlib.sha256(re.search(r'prompt_template = """(.*?)"""', t, re.S).group(1).encode()).hexdigest()[:16]
    print(p, h(head) == h(work))
PY
```

Everything else — transport, parsing, concurrency, output files — is fair game.

## Running an experiment

```bash
export DEEPSEEK_API_KEY=...     # or ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
.venv/bin/python scripts/single_error_prompting.py \
    --data ./data/test.csv --output ./results/single_full \
    --model deepseek-v4-pro --temperature 0 \
    --max-output-tokens 32768 --concurrency 8
```

**Always `--limit 16` first on a new model.** Check the parse rate and the label
distribution in the log before committing to a full pass. Every failure mode below
was caught this way.

### Reasoning models eat the output cap

Internal reasoning is billed as output tokens and counts against `--max-output-tokens`
*before* the visible answer. Too low a cap returns an empty string — the driver
reports that explicitly rather than scoring it as an API failure. The 2024 setting
of 50 tokens returns nothing at all on any current model. Use 32768.

`temperature=0` does **not** make these models deterministic: the same sample at the
same settings produced a reply within 16384 tokens on one run and overflowed it on
another. Worth a footnote in the paper — "deterministic decoding" is no longer
comparable across vendors.

### Concurrency: 8

Measured against `deepseek-v4-pro`: sequential 60 output tok/s, **8 in flight 289
tok/s (~4.8x)**, 16 in flight 172 tok/s, 24 worse still. The API throttles by
slowing down and **never returns 429**, so a too-high setting looks like it is
working while halving throughput. Re-measure per provider; do not assume 8 is
optimal elsewhere.

Results are written in sample order even when fetched in parallel. That ordering is
load-bearing: resume reads the highest `Sample <n>` in the log, so an out-of-order
log would make a restart skip unprocessed samples. If you touch `iter_responses`,
keep the ordering guarantee.

### Resume

Re-running the same command continues from the last completed sample. Each model
writes to `outcome-<provider>_<model>-<timestamp>/`, so several models can share one
`--output` without adopting each other's checkpoints. `--force-rerun` starts over.

### Model ids: look them up, don't guess

`deepseek-chat` (the pre-V4 name) does not exist on the current API; the real ids
are `deepseek-v4-pro` and `deepseek-v4-flash`. Query the provider before writing a
default:

```bash
curl -s https://api.deepseek.com/v1/models -H "Authorization: Bearer $DEEPSEEK_API_KEY"
```

### Anthropic: no sampling parameters

Current Claude models reject `temperature` / `top_p` with HTTP 400. Use
`--effort low|medium|high|xhigh|max` instead. Do **not** enable server-side
fallbacks for benchmark runs: a silent substitution would attribute another model's
predictions to the model under test.

Anthropic also has a Batch API at 50% cost with a 24-hour window, which suits a
4,865-sample pass. DeepSeek has no batch endpoint (`/v1/batches` and `/v1/files`
both 404) — concurrency is the only lever there.

## Data alignment trap

`annotation/multi_error_subset_97.csv` stores `Sample_Index` as a **position in the
test split** (420..4733), and `multi_error_metrics.py` joins a run log to the
annotations on that index. Slicing those 97 rows into a standalone CSV breaks the
join *silently*: the log carries 0..96, nothing matches, every metric reads 0.0, and
no error is raised. Use `scripts/make_multi_error_subset.py`, which emits the
submissions and a remapped index table as a consistent pair. The metrics script now
warns when the join rate is low, but the warning is a backstop, not a substitute.

The indices are positions in `test.csv` specifically — verified against all three
splits: dev is too short (4,379 rows), and train scores at chance on both a
label-containment and an `ast.parse` consistency check while test scores 90% / 86%.

## Label taxonomy

Class 13 `Other errors` is the catch-all; its definition names AttributeError,
RuntimeError, SyntaxWarning, ZeroDivisionError, MemoryError, ModuleNotFoundError,
"etc.". The dataset stores those under their own exception names, which are not keys
of `error_type_mapping` — a plain `.get(name, -1)` turned 18 of 4,865 test rows into
`-1` (unscoreable) while class 13 sat at support 0 and dragged the 14-class macro
average. `map_true_label()` folds any exception-shaped name into 13. Keep the driver
and `multi_error_metrics.py` in agreement on this.

Name matching must be space-insensitive: the prompt asks the model for
`Syntax Error (#2)`, so a reply that omits the id has to resolve via the spaced
spelling or a correct answer is scored as a miss.

## Results: what goes in git

Raw outputs stay out (`results/` is gitignored). The per-sample CSV re-serialises
question and student code for every sample — ~13 MB per full run, duplicating the
Hugging Face dataset — and the log holds every reasoning trace.

```bash
.venv/bin/python scripts/export_run_summary.py --run ./results/single_full
```

writes ~140 KB to `results_summary/<task>/<provider>_<model>/`. The task is part of
the path deliberately: one model produces both a single-error and a multi-error run,
and keying on the model alone made the second export overwrite the first. Because
paths are task- and model-scoped, machines running different models can each commit
their own summaries without conflicts.

## Measured cost and runtime (deepseek-v4-pro, for calibration)

| Run | Samples | Wall clock | Cost | Tokens |
|-----|---------|-----------|------|--------|
| multi-error subset | 97 | 73 min (sequential) | ¥3.82 | 0.11M in / 0.26M out |
| single-error full | 4,865 | 6.2 h (concurrency 8) | ¥68.86 | 5.02M in / 2.82M out |

Estimate from a *large* probe, not a small one: 16 samples implied ¥109 for the full
single-error run; it actually cost ¥69, because per-sample reasoning length varies
several-fold across samples.

## Running unattended on a Coder workspace

`coder schedule show <workspace>` — these workspaces default to a 1-hour TTL, and
Coder measures activity by **connections (SSH / IDE / apps), not CPU**. A detached
background job does not extend the deadline, so an unattended multi-hour run gets
killed mid-way. Either keep a session connected or extend explicitly:

```bash
coder schedule extend <workspace> 8h        # current instance, from now
coder schedule stop   <workspace> manual    # future builds (takes effect next build)
```

`/home/<user>` is a persistent volume, so a stop loses the process, not the data —
re-run the same command and it resumes.

## Conventions

- API keys come from environment variables only. Never write one into a file, a
  command that gets committed, or a results artifact.
- Branches: `ztang/pmt-00N-short-description`; PR titles carry `[PMT-00N]`. `PMT-*`
  is a repo-local convention, not a Linear identifier (the Linear team prefix is
  `CIR-`).
- Beware `pgrep -f <pattern>` in a monitoring script: the pattern matches the
  monitor's own command line, so a liveness check on it never fires.
