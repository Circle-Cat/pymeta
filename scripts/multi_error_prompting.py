"""
multi_error_prompting.py
========================

Multi-error prompting driver for Python error classification.

This script reads a CSV of student code submissions (question, expected answer,
student answer, and a ground-truth error label), sends each one to an LLM
(any of OpenAI / Anthropic / Gemini / DeepSeek, selected with --provider and
--model) using a *multi-label* classification prompt (the model may return several
error types per submission), parses the predicted (label-name, label-id) pairs,
and writes per-sample logs, a results JSON (accuracy / precision / recall / F1 and
a classification report), a confusion-matrix PNG, and a sample-index mapping CSV.
It supports resuming from the most recent "outcome-*" folder or forcing a fresh run.

Note: because upstream metrics here consume the Top-1 predicted label, the
confusion matrix / sklearn metrics computed in this script are single-label
(Top-1) approximations. Full multi-label ("contains" / any-hit) metrics are
computed separately by multi_error_metrics.py from the produced log files.

Required environment variable
------------------------------
    One API key for the provider under evaluation:
        OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY | DEEPSEEK_API_KEY
    e.g.  export DEEPSEEK_API_KEY="sk-..."      (Linux/macOS)

Example usage
-------------
    # The 97-sample subset is small enough to run whole, but check the parse rate
    # in the log before trusting the metrics.
    python multi_error_prompting.py --data ./data/multi_error_97.csv --output ./results \
        --provider deepseek --model deepseek-chat

    python multi_error_prompting.py --data ./data/multi_error_97.csv --output ./results \
        --model claude-fable-5

    # Force a fresh run, ignoring any existing outcome folders:
    python multi_error_prompting.py --data ./data/test.csv --output ./results --force-rerun

    # Each model writes to its own outcome-<provider>_<model>-<timestamp> folder,
    # so runs for different models never share or overwrite a resume checkpoint.

Expected input columns
----------------------
    question, exceptedAnswer, studentAnswer, all_errortype2
    (optional metadata: userId, name, questionId, attemptId, state,
     testOutcome, attemptstepid)
"""

import pandas as pd
import json
import numpy as np
import os
import glob
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # render without a display (headless VM / SSH session)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_clients import ChatClient, add_model_arguments

# Default paths (relative). These are overridden by --data / --output in main().
DATA_PATH = "./data/fixed_splitted_data_test_data_20250504_083254.csv"
BASE_PATH = "./results"

# Set in main() once the model is known. RUN_TAG namespaces every output file so
# that runs for different models can share one --output directory safely.
CLIENT = None
RUN_TAG = "model"
MODEL_LABEL = "model"
SYSTEM_PROMPT = "You are a Python error classification expert."

# Markdown emphasis / quote / list markers to strip before matching. "#" is
# deliberately kept: it carries the label id in the "(#2)" form.
_MD_STRIP = re.compile(r'[*_~`>\-]+')

# The prompt asks for "Predicted Label: ...". Current models also emit the plural,
# "Final Answer:", or a bare "Answer:", so accept those spellings too.
_LABEL_PREFIX = re.compile(
    r'^\s*(?:predicted\s*labels?|final\s*answers?|labels?|answers?|classification)\s*:\s*',
    re.IGNORECASE
)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").rstrip(".")


def _pairs_from_line(target_line: str, allow_unknown_names: bool = False):
    """Read [(num, name), ...] out of one already-cleaned label line.

    ``allow_unknown_names`` is only safe on a line that was explicitly marked as
    the label line; on an arbitrary reasoning line it would turn prose into
    bogus labels.
    """
    if not target_line:
        return []

    # Prefer matching "Name (#num)" multi-label form
    pairs = re.findall(
        r'([A-Za-z][A-Za-z ]*?errors?|No\s*error)\s*\(\s*#?\s*(-?\d+)\s*\)',
        target_line,
        flags=re.IGNORECASE
    )
    if pairs:
        return [(int(num), name.strip()) for name, num in pairs]

    # Otherwise match "num Name"
    m = re.match(r'^\s*(-?\d+)\s+(.+?)\s*$', target_line)
    if m:
        return [(int(m.group(1)), m.group(2).strip())]

    # A bare id list: "2, 3, 1"
    if re.match(r'^\s*\d{1,2}(\s*[,;、]\s*\d{1,2})*\s*$', target_line):
        return [(int(n), label_mapping.get(int(n), "UNKNOWN"))
                for n in re.findall(r'\d{1,2}', target_line)]

    # Fall back to matching names only (no numbers). Unlike the 2024 parser this
    # recovers the ids, so a reply that omits "(#n)" still yields a usable Top-1
    # instead of -1 / PARSING_FAILED.
    names = [seg.strip() for seg in re.split(r'[,、;]+', target_line) if seg.strip()]
    if names:
        resolved = [(LABEL_NAME_LOOKUP.get(_normalize_name(nm), -1), nm) for nm in names]
        if allow_unknown_names or all(num != -1 for num, _ in resolved):
            return resolved

    return []


def parse_predicted_pairs(response_text: str):
    """
    Parse [(num, name), ...] pairs from the model reply.
    Supported formats:
      - "Predicted Label: SyntaxError (#2), NameError (#3)"
      - "Label: 0 No error"
      - "2 SyntaxError"

    The label line is taken from the *end* of the reply: reasoning models write
    several lines first and may say "label" mid-reasoning, so the last labelled
    line is the answer. The 2024 parser took the first such line and gave up if
    no line carried the prefix at all.
    """
    if not response_text:
        return []

    lines = [ln for ln in response_text.splitlines() if ln.strip()]

    labelled = []
    for ln in lines:
        # Strip markdown bold/italic/underline/quote/list markers
        clean_ln = _MD_STRIP.sub('', ln).strip()
        if _LABEL_PREFIX.search(clean_ln):
            labelled.append(_LABEL_PREFIX.sub('', clean_ln).strip())

    for target_line in reversed(labelled):
        pairs = _pairs_from_line(target_line, allow_unknown_names=True)
        if pairs:
            return pairs

    # No usable label line: accept an unprefixed answer, but only in the strict
    # forms, so reasoning prose cannot be mistaken for a label.
    for ln in reversed(lines):
        pairs = _pairs_from_line(_MD_STRIP.sub('', ln).strip())
        if pairs:
            return pairs

    return []

def parse_arguments():
  """Parse command line arguments"""
  parser = argparse.ArgumentParser(description='Multi-error LLM classification with optional force rerun')
  add_model_arguments(parser)
  parser.add_argument('--data', default=DATA_PATH,
                     help='Path to the input dataset CSV (default: %(default)s)')
  parser.add_argument('--output', default=BASE_PATH,
                     help='Directory for outcome folders and results (default: %(default)s)')
  parser.add_argument('--force-rerun', action='store_true',
                     help='Force rerun from the beginning, ignoring existing results')
  parser.add_argument('--rerun', action='store_true',
                     help='Alias for --force-rerun')
  return parser.parse_args()

def find_latest_outcome_folder():
  """Find the most recent outcome folder and determine the last completed sample index"""
  outcome_folders = glob.glob(os.path.join(BASE_PATH, f"outcome-{RUN_TAG}-*"))
  if not outcome_folders:
      return None, 0

  # Sort folders by creation time (newest first)
  outcome_folders.sort(key=lambda x: os.path.getctime(x), reverse=True)
  latest_folder = outcome_folders[0]

  print(f"Found latest outcome folder: {latest_folder}")

  # Check all output files in the latest folder to find the last consistently completed sample
  log_files = glob.glob(os.path.join(latest_folder, f"{RUN_TAG}_classification_log_*.txt"))
  json_files = glob.glob(os.path.join(latest_folder, f"{RUN_TAG}_test_results_*.json"))
  csv_files = glob.glob(os.path.join(latest_folder, "sample_index_mapping_*.csv"))

  if not (log_files and json_files and csv_files):
      print("Incomplete output files found in latest folder, starting from beginning")
      return None, 0

  # Find the minimum last sample index across all files
  last_sample_indices = []

  # Check log file
  try:
      with open(log_files[0], 'r', encoding='utf-8') as f:
          content = f.read()
          # Find all sample indices in log
          import re
          sample_matches = re.findall(r'Sample (\d+):', content)
          if sample_matches:
              last_sample_indices.append(max(map(int, sample_matches)))
  except Exception as e:
      print(f"Error reading log file: {e}")
      return None, 0

  # Check CSV file
  try:
      df_mapping = pd.read_csv(csv_files[0])
      if not df_mapping.empty:
          last_sample_indices.append(df_mapping['Sample_Index'].max())
  except Exception as e:
      print(f"Error reading CSV file: {e}")
      return None, 0

  # Check JSON file (if it contains sample-level data)
  try:
      with open(json_files[0], 'r') as f:
          json_data = json.load(f)
          if 'total_processed_samples' in json_data:
              last_sample_indices.append(json_data['total_processed_samples'] - 1)
  except Exception as e:
      print(f"Error reading JSON file: {e}")

  if last_sample_indices:
      # Use the minimum to ensure all files have this sample
      last_completed_sample = min(last_sample_indices)
      print(f"Resuming from sample index: {last_completed_sample + 1}")
      return latest_folder, last_completed_sample + 1
  else:
      print("No completed samples found, starting from beginning")
      return None, 0

def create_outcome_folder():
  """Create a new outcome folder with timestamp"""
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  folder_name = f"outcome-{RUN_TAG}-{timestamp}"
  folder_path = os.path.join(BASE_PATH, folder_name)
  os.makedirs(folder_path, exist_ok=True)
  print(f"Created outcome folder: {folder_path}")
  return folder_path, timestamp

def generate_response_with_retry(prompt):
    """Send one prompt to the configured model.

    Returns:
        tuple: (response_text, error_message) where one will be None

    Retries and backoff live in ChatClient (llm_clients.py).
    """
    return CLIENT.generate(prompt, system=SYSTEM_PROMPT)

def build_input_text(row):
    """Render the per-sample block appended to the prompt template.

    Kept byte-identical to the 2024 driver: the prompt is fixed experimental
    apparatus and must not drift when the transport around it changes.
    """
    return f"""
Question: {row['question']}
Expected Answer: {row['exceptedAnswer']}
Student Answer: {row['studentAnswer']}
"""


def iter_responses(df, start_idx, concurrency):
    """Yield (idx, row, response, error) for every sample, in index order.

    With ``concurrency > 1`` the API calls inside a chunk are issued in parallel,
    but results are handed back strictly in index order. That ordering is not
    cosmetic: resume works by reading the highest "Sample <n>" in the log, so an
    out-of-order log would make a restart skip unprocessed samples.

    Work is issued a chunk at a time rather than as one long sliding window. The
    chunk boundary costs a little throughput when latencies vary, and buys a
    clean checkpoint: everything before it is written, everything after it has
    not been sent yet.
    """
    total = len(df)
    chunk = max(concurrency * 2, 1)
    idx = start_idx
    while idx < total:
        batch = list(range(idx, min(idx + chunk, total)))
        prompts = [prompt_template + build_input_text(df.iloc[i]) for i in batch]
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(generate_response_with_retry, prompts))
        else:
            results = [generate_response_with_retry(p) for p in prompts]
        for i, (response, error) in zip(batch, results):
            yield i, df.iloc[i], response, error
        idx += len(batch)


# Define prompt template
prompt_template = """
You are an expert in Python code error classification, functioning as an automated assessment assistant for a large-scale programming education platform. Your task is to analyze student code submissions, which may contain errors, along with the corresponding coding questions and correct answers. You need to accurately identify whether the student code contains any errors, as well as the specific type of error in each case.


For each submission, you will receive structured information, including:
- The question description
- The expected answer
- The student's code


**Important**: Many student submissions may be completely correct with no errors.


---
## Analysis Process (internal reasoning, not directly output):
1. Examine the student's code for explicit errors (SyntaxError, NameError, TypeError, IndentationError, UnboundLocalError, KeyError, IndexError, EOFError, ValueError, TabError).
2. If no explicit error is found, compare the student's code with the expected answer to check the logic.
3. If the logic is incorrect, classify as LogicError. Otherwise, classify as No error.
Use this reasoning process internally to justify the final prediction.


---
## Definition of Error Type Categories (Label numbers and names)
0: No error — The code runs successfully with no explicit or logic errors.
1: LogicError — The code has no explicit error but has a logic flaw.
2: SyntaxError — The code contains invalid Python syntax and cannot be parsed.
3: NameError — Raised when attempting to access a variable, function, or module name that is not defined.
4: TypeError — Raised when an operation or function is applied to an object of inappropriate type or when types are incompatible.
5: IndentationError — Incorrect indentation. Python relies on indentation to determine code blocks.
6: UnboundLocalError — A local variable is referenced before assignment (typically within a function). Subclass of NameError.
7: KeyError — Raised when accessing a dictionary key that doesn't exist.
8: IndexError — Raised when accessing an index position in a sequence that is out of range.
9: EOFError — Raised when the input() function hits an end-of-file condition without reading data.
10: RecursionError — Raised when the maximum recursion depth is exceeded (infinite recursion).
11: ValueError — Raised when an argument has the correct type but an inappropriate value.
12: TabError — Raised when indentation inconsistently mixes tabs and spaces.
13: Other errors — Includes AttributeError, RuntimeError, SyntaxWarning, ZeroDivisionError, MemoryError, ModuleNotFoundError, etc.


---
## Output Requirements
Please provide output in the following format (consistent and unified):
Reasoning: brief explanation of detected errors
Predicted Label: ErrorName (#n), ErrorName (#m), ...


---
## Example
### Input:
Question description: "Write a function that returns the square of a number."

Expected Answer:
def square(x):
    return x * x

Student Code:
def square(x)
    return x + y

Output:
Reasoning: Firstly, we examine the code for explicit errors in order. The function header is missing a colon (SyntaxError). The variable y is not defined (NameError). While there are explicit errors, we continue to examine the logic. Even if y were defined, the logic is incorrect since it should return x * x instead of x + y (LogicError).
Predicted Label: SyntaxError (#2), NameError (#3), LogicError (#1)

Test Example (for model to fill)
Reasoning:
Predicted Label:
"""

# Define error type mapping dictionary
error_type_mapping = {
  "No error": 0,
  "LogicError": 1,
  "SyntaxError": 2,
  "NameError": 3,
  "TypeError": 4,
  "IndentationError": 5,
  "UnboundLocalError": 6,
  "KeyError": 7,
  "IndexError": 8,
  "EOFError": 9,
  "RecursionError": 10,
  "ValueError": 11,
  "TabError": 12,
  "Other errors": 13
}

# Reverse mapping for label names
label_mapping = {v: k for k, v in error_type_mapping.items()}

# Normalised name -> id, accepting both "LogicError" and "Logic Error" spellings.
LABEL_NAME_LOOKUP = {name.lower().replace(" ", ""): num for name, num in error_type_mapping.items()}

# Class 13 is the taxonomy's catch-all; its definition in the prompt explicitly
# covers AttributeError, RuntimeError, SyntaxWarning, ZeroDivisionError,
# MemoryError, ModuleNotFoundError, "etc.". The dataset stores those under their
# own exception names, which are not keys of error_type_mapping, so a plain
# .get(name, -1) turned them into -1: rows no model can ever be scored right on,
# and an empty "Other errors" class dragging the 14-class macro average down.
OTHER_ERRORS_ID = 13
_EXCEPTION_SUFFIXES = ("error", "errors", "warning", "exception")


def map_true_label(label_str):
    """Map a gold label string to its taxonomy id, or -1 if unrecognisable."""
    name = str(label_str).strip()
    if name in error_type_mapping:
        return error_type_mapping[name]
    key = name.lower().replace(" ", "")
    if key in LABEL_NAME_LOOKUP:
        return LABEL_NAME_LOOKUP[key]
    if key.endswith(_EXCEPTION_SUFFIXES):
        return OTHER_ERRORS_ID
    return -1


def plot_confusion_matrix(y_true, y_pred, label_mapping, output_file):
  """Plot and save confusion matrix"""
  # Create confusion matrix
  label_indices = sorted(label_mapping.keys())
  cm = confusion_matrix(y_true, y_pred, labels=label_indices)
  assert cm.shape == (14, 14)

  # Convert label indices to class names
  class_names = [label_mapping.get(i, f"class_{i}") for i in sorted(label_mapping.keys())]

  # Create figure with larger size for better readability
  plt.figure(figsize=(14, 12))

  # Plot confusion matrix as heatmap
  sns.heatmap(
      cm,
      annot=True,
      fmt='d',
      cmap='Blues',
      xticklabels=class_names,
      yticklabels=class_names
  )

  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title(f'Confusion Matrix - {MODEL_LABEL} Error Classification')


  # Rotate x-axis labels for better readability
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()

  # Save the figure
  plt.savefig(output_file, dpi=300, bbox_inches='tight')
  plt.close()

  return cm

def save_intermediate_results(output_folder, timestamp, predictions, true_labels, sample_details,
                           current_sample_idx, df_total_len):
  """Save intermediate results to prevent data loss"""
  # Save current progress to JSON
  if predictions and true_labels:
      # Calculate metrics for current progress
      accuracy = accuracy_score(true_labels, predictions)
      precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
      recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
      f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

      target_names = [
          "No error", "LogicError", "SyntaxError", "NameError", "TypeError",
          "IndentationError", "UnboundLocalError", "KeyError", "IndexError",
          "EOFError", "RecursionError", "ValueError", "TabError", "Other errors"
      ]

      report = classification_report(
          true_labels, predictions,
          labels=list(range(14)),
          target_names=target_names,
          zero_division=0,
          output_dict=True
      )

      # Save intermediate results
      intermediate_results = {
          "timestamp": timestamp,
          "task": "multi_error",
          "provider": CLIENT.provider,
          "model": CLIENT.model,
          "run_config": CLIENT.describe(),
          "token_usage": dict(CLIENT.usage),
          "parse_failures": int(sum(1 for p in predictions if p == -1)),
          "dataset_path": DATA_PATH,
          "total_samples_in_dataset": df_total_len,
          "total_processed_samples": len(predictions),
          "current_sample_index": current_sample_idx,
          "progress_percentage": (len(predictions) / df_total_len) * 100,
          "accuracy": accuracy,
          "precision": precision,
          "recall": recall,
          "f1_score": f1,
          "classification_report": report,
          "is_complete": False
      }

      json_file = os.path.join(output_folder, f"{RUN_TAG}_test_results_{timestamp}.json")
      with open(json_file, 'w') as f:
          json.dump(intermediate_results, f, indent=4)

  # Save current sample mapping
  if sample_details:
      csv_file = os.path.join(output_folder, f"sample_index_mapping_{timestamp}.csv")
      sample_mapping_df = pd.DataFrame(sample_details)
      sample_mapping_df.to_csv(csv_file, index=False)

# Main execution
def main():
  # Parse command line arguments
  args = parse_arguments()

  # Override the default paths with the values provided on the command line.
  global BASE_PATH, DATA_PATH, CLIENT, RUN_TAG, MODEL_LABEL
  BASE_PATH = args.output
  DATA_PATH = args.data
  os.makedirs(BASE_PATH, exist_ok=True)

  # Build the model client first: RUN_TAG derives from it and namespaces every
  # output file, so resume never mixes two models' checkpoints.
  # The multi-error prompt asks for a Reasoning line before the label, so the
  # output cap has to cover that plus any internal reasoning tokens.
  CLIENT = ChatClient(
      provider=args.provider,
      model=args.model,
      max_output_tokens=args.max_output_tokens or 4096,
      temperature=args.temperature,
      effort=args.effort,
      timeout=args.timeout,
  )
  RUN_TAG = CLIENT.run_tag()
  MODEL_LABEL = CLIENT.model
  print(f"Model: {CLIENT.describe()}")

  force_rerun = args.force_rerun or args.rerun

  if force_rerun:
      print("🔄 Force rerun mode enabled - starting fresh from the beginning")
      output_folder, timestamp = create_outcome_folder()
      start_idx = 0
      sample_details = []
      predictions = []
      true_labels = []
  else:
      # Check for existing work and determine starting point
      latest_folder, start_idx = find_latest_outcome_folder()

      if latest_folder and start_idx > 0:
          # Resume from existing work
          output_folder = latest_folder
          # The folder name is outcome-<RUN_TAG>-<timestamp>; strip the tag too,
          # otherwise resume derives a bogus timestamp and starts a second log.
          timestamp = os.path.basename(latest_folder).replace(f"outcome-{RUN_TAG}-", "")
          print(f"📋 Resuming work from sample index {start_idx}")

          # Load existing data
          existing_log_file = glob.glob(os.path.join(output_folder, f"{RUN_TAG}_classification_log_*.txt"))[0]
          existing_csv_file = glob.glob(os.path.join(output_folder, "sample_index_mapping_*.csv"))[0]

          # Load existing sample details
          existing_df = pd.read_csv(existing_csv_file)
          sample_details = existing_df.to_dict('records')
          predictions = existing_df['predicted_label_num'].tolist()
          true_labels = existing_df['true_label_num'].tolist()

      else:
          # Start fresh
          output_folder, timestamp = create_outcome_folder()
          start_idx = 0
          sample_details = []
          predictions = []
          true_labels = []

  # Set up file paths with timestamp
  LOG_FILE = os.path.join(output_folder, f"{RUN_TAG}_classification_log_{timestamp}.txt")
  OUTPUT_FILE = os.path.join(output_folder, f"{RUN_TAG}_test_results_{timestamp}.json")
  CONFUSION_MATRIX_FILE = os.path.join(output_folder, f"confusion_matrix_{timestamp}.png")
  SAMPLE_MAPPING_FILE = os.path.join(output_folder, f"sample_index_mapping_{timestamp}.csv")

  # Read the complete dataset (not just first 100 rows)
  print("Loading complete dataset...")
  df = pd.read_csv(DATA_PATH)
  print(f"Total samples in dataset: {len(df)}")
  if args.limit:
      df = df.iloc[:args.limit].copy()
      print(f"--limit {args.limit}: evaluating the first {len(df)} samples only (smoke test)")

  # Surface unscoreable gold labels before spending any tokens.
  unmapped = sorted({str(v) for v in df['all_errortype2'] if map_true_label(v) == -1})
  if unmapped:
      n = int(sum(map_true_label(v) == -1 for v in df['all_errortype2']))
      print(f"⚠️  {n} rows carry a gold label outside the taxonomy and will be scored "
            f"as -1 (never correct): {unmapped}")
  other = int(sum(map_true_label(v) == OTHER_ERRORS_ID for v in df['all_errortype2']))
  if other:
      print(f"Gold labels folded into 'Other errors' (#13): {other} rows")

  # Open log file for detailed logging (append mode if resuming, write mode if force rerun)
  log_mode = 'w' if force_rerun else ('a' if start_idx > 0 else 'w')
  with open(LOG_FILE, log_mode, encoding='utf-8') as log_f:
      if start_idx == 0 or force_rerun:
          # Write header for new files or force rerun
          log_f.write(f"{CLIENT.provider}/{CLIENT.model} Error Classification Log\n")
          log_f.write(f"Run config: {CLIENT.describe()}\n")
          if force_rerun:
              log_f.write(f"FORCE RERUN MODE - Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
          else:
              log_f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
          log_f.write(f"Dataset: {DATA_PATH}\n")
          log_f.write(f"Total samples: {len(df)}\n")
          log_f.write("="*80 + "\n\n")
      else:
          # Write resume header
          log_f.write(f"\n{'='*80}\n")
          log_f.write(f"RESUMED at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
          log_f.write(f"Resuming from sample index: {start_idx}\n")
          log_f.write("="*80 + "\n\n")

      # Process samples starting from start_idx. iter_responses handles the
      # parallel fetch and hands results back in index order, so everything
      # below stays sequential and order-dependent state stays correct.
      for idx, row, response, error in iter_responses(df, start_idx, args.concurrency):
          current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

          print(f"Processing sample {idx}/{len(df)} ({((idx+1)/len(df)*100):.1f}%)")

          # If there's a connection error, stop execution. Samples after this one
          # in the chunk are dropped even though they were fetched: recording them
          # would leave a hole at idx, and resume would then skip past it.
          if error and any(keyword in error.lower() for keyword in ['connection', 'timeout', 'network', 'rate limit']):
              print(f"Connection error encountered: {error}")
              print("Stopping execution to prevent corrupted output files.")
              print(f"Progress saved. Resume from sample index {idx} next time.")
              break

          # Get true label
          true_label_str = row['all_errortype2']
          true_label_num = map_true_label(true_label_str)
          true_labels.append(true_label_num)

          if response:
            predicted_pairs = parse_predicted_pairs(response)

            if predicted_pairs:
                # Take Top-1 into predictions (keeps downstream accuracy etc. unchanged)
                predicted_label_num = predicted_pairs[0][0]
                predicted_label_name = predicted_pairs[0][1]
                predictions.append(predicted_label_num)

                # Multi-label display (drop the redundant "Label:" prefix)
                predicted_display = ", ".join(
                    f"{name} (#{num})" if num != -1 else f"{name}"
                    for num, name in predicted_pairs
                )

                # Correct = true label is present anywhere in the predicted set => Yes
                correct_yes = any(num == true_label_num for num, _ in predicted_pairs)
                correct_str = "Yes" if correct_yes else "No"

                log_msg = f"[{current_time}] Sample {idx}:\n"
                log_msg += f"  True Label: {true_label_str} (#{true_label_num})\n"
                log_msg += f"  Predicted Label: {predicted_display}\n"
                log_msg += f"  Correct: {correct_str}\n"
                log_msg += f"  Model Response: {response}\n"

                print(f"Sample {idx}: True={true_label_str}, Pred={predicted_display}, Correct={correct_str}")
                log_f.write(log_msg + "\n")

            else:
                # Parsing failed fallback
                predictions.append(-1)
                log_msg = f"[{current_time}] Sample {idx}:\n"
                log_msg += f"  True Label: {true_label_str} (#{true_label_num})\n"
                log_msg += f"  Predicted Label: PARSING_FAILED\n"
                log_msg += f"  Model Response: {response}\n"
                print(f"Sample {idx}: True={true_label_str}, Pred=PARSING_FAILED")
                log_f.write(log_msg + "\n")
          else:
              predictions.append(-1)
              log_msg = f"[{current_time}] Sample {idx}:\n"
              log_msg += f"  True Label: {true_label_str} (#{true_label_num})\n"
              log_msg += f"  Predicted Label: API_CALL_FAILED\n"
              if error:
                  log_msg += f"  Error: {error}\n"

              print(f"Sample {idx}: True={true_label_str}, Predicted=API_CALL_FAILED")
              log_f.write(log_msg + "\n")

          # Store sample details for mapping file
          sample_details.append({
              'Sample_Index': idx,
              'userId': row.get('userId', 'N/A'),
              'name': row.get('name', 'N/A'),
              'questionId': row.get('questionId', 'N/A'),
              'question': row.get('question', 'N/A'),
              'exceptedAnswer': row.get('exceptedAnswer', 'N/A'),
              'attemptId': row.get('attemptId', 'N/A'),
              'state': row.get('state', 'N/A'),
              'studentAnswer': row.get('studentAnswer', 'N/A'),
              'testOutcome': row.get('testOutcome', 'N/A'),
              'attemptstepid': row.get('attemptstepid', 'N/A'),
              'true_label': true_label_str,
              'true_label_num': true_label_num,
              'predicted_label_num': predictions[-1],
              'predicted_label_name': label_mapping.get(predictions[-1], 'UNKNOWN') if predictions[-1] != -1 else 'FAILED'
          })

          log_f.flush()  # Ensure immediate write to file

          # Save intermediate results every 50 samples
          if (idx + 1) % 50 == 0:
              save_intermediate_results(output_folder, timestamp, predictions, true_labels,
                                      sample_details, idx, len(df))
              print(f"Intermediate results saved at sample {idx}")

  # Final processing and results saving
  if predictions and true_labels:
      print("Computing final metrics...")

      # Compute metrics
      accuracy = accuracy_score(true_labels, predictions)
      precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
      recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
      f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
      # Top-1 macro-F1. The multi-label "contains" / Coverage Rate metrics are
      # computed separately by multi_error_metrics.py from the log file.
      macro_f1 = f1_score(true_labels, predictions, average='macro',
                          labels=list(range(14)), zero_division=0)

      target_names = [
          "No error", "LogicError", "SyntaxError", "NameError", "TypeError",
          "IndentationError", "UnboundLocalError", "KeyError", "IndexError",
          "EOFError", "RecursionError", "ValueError", "TabError", "Other errors"
      ]

      # Generate classification report
      report = classification_report(
          true_labels, predictions,
          labels=list(range(14)),
          target_names=target_names,
          zero_division=0,
          output_dict=True
      )

      report_str = classification_report(
          true_labels, predictions,
          labels=list(range(14)),
          target_names=target_names,
          zero_division=0
      )

      # Plot and save confusion matrix
      cm = plot_confusion_matrix(true_labels, predictions, label_mapping, CONFUSION_MATRIX_FILE)

      # Save final results to JSON file
      results = {
          "timestamp": timestamp,
          "task": "multi_error",
          "provider": CLIENT.provider,
          "model": CLIENT.model,
          "run_config": CLIENT.describe(),
          "token_usage": dict(CLIENT.usage),
          "parse_failures": int(sum(1 for p in predictions if p == -1)),
          "macro_f1_top1": macro_f1,
          "dataset_path": DATA_PATH,
          "total_samples_in_dataset": len(df),
          "total_processed_samples": len(predictions),
          "accuracy": accuracy,
          "precision": precision,
          "recall": recall,
          "f1_score": f1,
          "classification_report": report,
          "confusion_matrix": cm.tolist(),
          "is_complete": True,
          "force_rerun": force_rerun
      }

      with open(OUTPUT_FILE, 'w') as f:
          json.dump(results, f, indent=4)

      # Save final sample mapping file
      sample_mapping_df = pd.DataFrame(sample_details)
      sample_mapping_df.to_csv(SAMPLE_MAPPING_FILE, index=False)

      # Append final summary to log file
      with open(LOG_FILE, 'a', encoding='utf-8') as log_f:
          log_f.write("\n" + "="*80 + "\n")
          log_f.write("FINAL EVALUATION RESULTS\n")
          log_f.write("="*80 + "\n")
          log_f.write(f"Accuracy: {accuracy:.4f}\n")
          log_f.write(f"Precision (weighted): {precision:.4f}\n")
          log_f.write(f"Recall (weighted): {recall:.4f}\n")
          log_f.write(f"F1-score (weighted): {f1:.4f}\n\n")
          log_f.write("Classification Report:\n")
          log_f.write(report_str)
          log_f.write("\n\nConfusion Matrix:\n")
          log_f.write(str(cm))
          log_f.write(f"\n\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
          if force_rerun:
              log_f.write("Mode: FORCE RERUN\n")

      print("\n" + "="*60)
      print("EVALUATION COMPLETED")
      print("="*60)
      print(f"Accuracy: {accuracy:.4f}")
      print(f"Precision: {precision:.4f}")
      print(f"Recall: {recall:.4f}")
      print(f"F1-score (weighted): {f1:.4f}")
      print(f"Macro-F1 (Top-1): {macro_f1:.4f}")
      failed = sum(1 for p in predictions if p == -1)
      print(f"Unusable predictions (parse or API failure): {failed}/{len(predictions)}"
            f" ({failed / len(predictions) * 100:.1f}%)")
      print(f"Tokens: {CLIENT.usage['input_tokens']:,} in / {CLIENT.usage['output_tokens']:,} out"
            f" over {CLIENT.usage['calls']} calls ({CLIENT.usage['retries']} retries)")
      print(f"\nNext: multi-label metrics come from multi_error_metrics.py --logs {LOG_FILE}")
      print(f"\nResults saved to: {OUTPUT_FILE}")
      print(f"Detailed log saved to: {LOG_FILE}")
      print(f"Confusion matrix saved to: {CONFUSION_MATRIX_FILE}")
      print(f"Sample mapping saved to: {SAMPLE_MAPPING_FILE}")
      if force_rerun:
          print("🔄 Completed in FORCE RERUN mode")
  else:
      print("No predictions were made. Check API connection and try again.")

if __name__ == "__main__":
  main()
