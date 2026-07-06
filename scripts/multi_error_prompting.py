"""
multi_error_prompting.py
========================

Multi-error prompting driver for Python error classification (DeepSeek).

This script reads a CSV of student code submissions (question, expected answer,
student answer, and a ground-truth error label), sends each one to the DeepSeek
chat API with a *multi-label* classification prompt (the model may return several
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
    DEEPSEEK_API_KEY   Your DeepSeek API key. Set it before running, e.g.:
                           export DEEPSEEK_API_KEY="sk-..."     (Linux/macOS)
                           setx  DEEPSEEK_API_KEY "sk-..."      (Windows)

Example usage
-------------
    python multi_error_prompting.py --data ./data/test.csv --output ./results

    # Force a fresh run, ignoring any existing outcome folders:
    python multi_error_prompting.py --data ./data/test.csv --output ./results --force-rerun

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
import time
import argparse
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from datetime import datetime
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Read API key from an environment variable (never hard-code secrets).
# Set DEEPSEEK_API_KEY in your shell before running this script.
API_KEY = os.environ["DEEPSEEK_API_KEY"]

# Default paths (relative). These are overridden by --data / --output in main().
DATA_PATH = "./data/fixed_splitted_data_test_data_20250504_083254.csv"
BASE_PATH = "./results"

# DeepSeek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # confirm this is the correct API endpoint
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def parse_predicted_pairs(response_text: str):
    """
    Parse [(num, name), ...] pairs from the model reply.
    Supported formats:
      - "Predicted Label: SyntaxError (#2), NameError (#3)"
      - "Label: 0 No error"
      - "2 SyntaxError"
    """
    if not response_text:
        return []

    target_line = ""
    for ln in response_text.splitlines():
        # Strip markdown bold/italic/underline/quote/list markers
        clean_ln = re.sub(r'[*_~`>\-]+', '', ln).strip()
        if re.search(r'(Predicted\s*Label\s*:|Label\s*:)', clean_ln, flags=re.IGNORECASE):
            target_line = re.sub(
                r'^\s*(Predicted\s*Label\s*:|Label\s*:)\s*',
                '',
                clean_ln,
                flags=re.IGNORECASE
            ).strip()
            break

    if not target_line:
        return []

    # Prefer matching "Name (#num)" multi-label form
    pairs = re.findall(
        r'([A-Za-z][A-Za-z ]*?error|No error|Other errors)\s*\(#\s*(-?\d+)\s*\)',
        target_line,
        flags=re.IGNORECASE
    )
    if pairs:
        return [(int(num), name.strip()) for name, num in pairs if re.match(r'-?\d+', num)]

    # Otherwise match "num Name"
    m = re.match(r'^\s*(-?\d+)\s+(.+?)\s*$', target_line)
    if m:
        return [(int(m.group(1)), m.group(2).strip())]

    # Fall back to matching names only (no numbers)
    names = [seg.strip() for seg in re.split(r'[,、;]+', target_line) if seg.strip()]
    if names:
        return [(-1, nm) for nm in names]

    return []

def parse_arguments():
  """Parse command line arguments"""
  parser = argparse.ArgumentParser(description='deepseek Error Classification with optional force rerun')
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
  outcome_folders = glob.glob(os.path.join(BASE_PATH, "outcome-*"))
  if not outcome_folders:
      return None, 0

  # Sort folders by creation time (newest first)
  outcome_folders.sort(key=lambda x: os.path.getctime(x), reverse=True)
  latest_folder = outcome_folders[0]

  print(f"Found latest outcome folder: {latest_folder}")

  # Check all output files in the latest folder to find the last consistently completed sample
  log_files = glob.glob(os.path.join(latest_folder, "deepseek_classification_log_*.txt"))
  json_files = glob.glob(os.path.join(latest_folder, "deepseek_test_results_*.json"))
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
  folder_name = f"outcome-{timestamp}"
  folder_path = os.path.join(BASE_PATH, folder_name)
  os.makedirs(folder_path, exist_ok=True)
  print(f"Created outcome folder: {folder_path}")
  return folder_path, timestamp

def generate_response_with_retry(prompt, max_retries=10, retry_delay=10):
    """Generate response from DeepSeek API with retry mechanism and error handling.

    Args:
        prompt (str): The input prompt/message for the AI
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Initial delay between retries in seconds (will exponentially increase)

    Returns:
        tuple: (response_text, error_message) where one will be None
    """
    for attempt in range(max_retries):
        try:
            # Prepare the request payload
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a Python error classification expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }

            # Make the API request
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=data,
                timeout=10  # Add timeout to prevent hanging
            )
            response.raise_for_status()

            # Parse and return the successful response
            return response.json()["choices"][0]["message"]["content"].strip(), None

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {error_msg}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return None, f"API request failed: {error_msg}"

        except (KeyError, ValueError) as e:
            # Handle JSON parsing errors
            return None, f"Response parsing failed: {str(e)}"

    return None, "Max retries exceeded"

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
  # plt.title('Confusion Matrix - deepseek Error Classification')
  plt.title('Confusion Matrix - deepseek Error Classification')


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

      json_file = os.path.join(output_folder, f"deepseek_test_results_{timestamp}.json")
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
  global BASE_PATH, DATA_PATH
  BASE_PATH = args.output
  DATA_PATH = args.data
  os.makedirs(BASE_PATH, exist_ok=True)

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
          timestamp = os.path.basename(latest_folder).replace("outcome-", "")
          print(f"📋 Resuming work from sample index {start_idx}")

          # Load existing data
          existing_log_file = glob.glob(os.path.join(output_folder, "deepseek_classification_log_*.txt"))[0]
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
  LOG_FILE = os.path.join(output_folder, f"deepseek_classification_log_{timestamp}.txt")
  OUTPUT_FILE = os.path.join(output_folder, f"deepseek_test_results_{timestamp}.json")
  CONFUSION_MATRIX_FILE = os.path.join(output_folder, f"confusion_matrix_{timestamp}.png")
  SAMPLE_MAPPING_FILE = os.path.join(output_folder, f"sample_index_mapping_{timestamp}.csv")

  # Read the complete dataset (not just first 100 rows)
  print("Loading complete dataset...")
  df = pd.read_csv(DATA_PATH)
  print(f"Total samples in dataset: {len(df)}")

  # Open log file for detailed logging (append mode if resuming, write mode if force rerun)
  log_mode = 'w' if force_rerun else ('a' if start_idx > 0 else 'w')
  with open(LOG_FILE, log_mode, encoding='utf-8') as log_f:
      if start_idx == 0 or force_rerun:
          # Write header for new files or force rerun
          # log_f.write(f"deepseek Error Classification Log\n")
          log_f.write(f"deepseek Error Classification Log\n")
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

      # Process samples starting from start_idx
      for idx in range(start_idx, len(df)):
          row = df.iloc[idx]
          current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

          print(f"Processing sample {idx}/{len(df)} ({((idx+1)/len(df)*100):.1f}%)")

          # Construct input text using actual column names
          input_text = f"""
Question: {row['question']}
Expected Answer: {row['exceptedAnswer']}
Student Answer: {row['studentAnswer']}
"""
          prompt = prompt_template + input_text

          # Generate response with retry mechanism
          response, error = generate_response_with_retry(prompt)

          # If there's a connection error, stop execution
          if error and any(keyword in error.lower() for keyword in ['connection', 'timeout', 'network', 'rate limit']):
              print(f"Connection error encountered: {error}")
              print("Stopping execution to prevent corrupted output files.")
              print(f"Progress saved. Resume from sample index {idx} next time.")
              break

          # Get true label
          true_label_str = row['all_errortype2']
          true_label_num = error_type_mapping.get(true_label_str, -1)
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
                log_msg += f"  deepseek Response: {response}\n"

                print(f"Sample {idx}: True={true_label_str}, Pred={predicted_display}, Correct={correct_str}")
                log_f.write(log_msg + "\n")

            else:
                # Parsing failed fallback
                predictions.append(-1)
                log_msg = f"[{current_time}] Sample {idx}:\n"
                log_msg += f"  True Label: {true_label_str} (#{true_label_num})\n"
                log_msg += f"  Predicted Label: PARSING_FAILED\n"
                log_msg += f"  deepseek Response: {response}\n"
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
      print(f"F1-score: {f1:.4f}")
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
