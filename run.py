import os
import sys
import subprocess
import datetime
import shutil


# --------------- Experiment Configuration ---------------

# EXP_NAME = "Llama-3.1-8B_dev"  # TODO: Change model here
# EXP_NAME = "Ministral-3-8B_dev"
EXP_NAME = "Qwen3-8B_dev"
# EXP_NAME = "gemini-2.5-flash_dev"

TIMESTAMP = datetime.datetime.now().strftime("%m%d_%H%M")
RUN_ID = f"{EXP_NAME}_{TIMESTAMP}"


# --------------- Path Settings ---------------

PROJECT_ROOT = "/root/autodl-tmp/FEHU"  # TODO: Change string here
# PROJECT_ROOT = ""

INFERENCE_SCRIPT = os.path.join(PROJECT_ROOT, "inference.py")  # TODO: Change string here
# INFERENCE_SCRIPT = os.path.join(PROJECT_ROOT, "inference_api.py")

EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "evaluation.py")

# MODEL_NAME = "Llama-3.1-8B"  # TODO: Change model here
# MODEL_NAME = "Ministral-3-8B"
MODEL_NAME = "Qwen3-8B"
# MODEL_NAME = "gemini-2.5-flash"

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, f"outputs/{MODEL_NAME}")

GOLD_LABELS = os.path.join(PROJECT_ROOT, "dataset/gold_labels/dev/dev_human_values.json")  # TODO: Change string here

# Directory for this specific experiment archive
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "logs", RUN_ID)


# --------------- Pipeline Execution ---------------

print(f"Starting Automated Pipeline | Run ID: {RUN_ID}")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# ---------- Step A: Run LLM Inference (Stream logs to terminal & txt) ----------
txt_log_path = os.path.join(ARCHIVE_DIR, f"{RUN_ID}_inference.txt")
print(f"\n[INFO] [Step 1/3] Running inference script...")
print(f"[INFO] Terminal output & warnings are being synced to: {txt_log_path}\n")

# 1. TTY Spoofing: Force tqdm to output \r for dynamic progress bars
my_env = os.environ.copy()
my_env["PYTHONUNBUFFERED"] = "1"
my_env["TQDM_FORCE_TTY"] = "1"
my_env["EXP_ARCHIVE_DIR"] = ARCHIVE_DIR

with open(txt_log_path, "w", encoding="utf-8") as log_file:
    log_file.write(f"=== Inference Runtime Log for {RUN_ID} ===\n\n")

    # 2. Binary Unbuffered Streaming: Read raw bytes to intercept \r
    process = subprocess.Popen(
        ["python", "-u", INFERENCE_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=my_env
    )

    # 3. Byte-level Interceptor & Router
    current_line_bytes = bytearray()
    while True:
        char_byte = process.stdout.read(1)
        if not char_byte and process.poll() is not None:
            break

        if char_byte:
            # A. Real-time dynamic output to the terminal screen
            sys.stdout.buffer.write(char_byte)
            sys.stdout.flush()

            # B. Log File Router: Discard temporary progress bar frames (\r)
            if char_byte == b'\r':
                current_line_bytes = bytearray() # Clear buffer on carriage return
            elif char_byte == b'\n':
                # Write finalized lines to the text log
                log_file.write(current_line_bytes.decode('utf-8', errors='replace') + '\n')
                log_file.flush()
                current_line_bytes = bytearray()
            else:
                current_line_bytes.extend(char_byte)

    process.wait()

if process.returncode != 0:
    print(f"\n[ERROR] Inference failed or was interrupted! Please check the log file: {txt_log_path}")
    sys.exit(1)

# ---------- Step B: Run Evaluation Script (Save metrics to json) ----------
print("\n[INFO] [Step 2/3] Inference complete! Evaluating predictions...")
eval_command = [
    "python", EVAL_SCRIPT,
    "--gold_task1", GOLD_LABELS,
    "--pred_task1a", os.path.join(DEFAULT_OUTPUT_DIR, "task1", "pred_task1a.json"),
    "--pred_task1b", os.path.join(DEFAULT_OUTPUT_DIR, "task1", "pred_task1b.json"),
    "--gold_task2", GOLD_LABELS,
    "--pred_task2a", os.path.join(DEFAULT_OUTPUT_DIR, "task2", "pred_task2a.json"),
    "--pred_task2b", os.path.join(DEFAULT_OUTPUT_DIR, "task2", "pred_task2b.json")
]

result = subprocess.run(eval_command, capture_output=True, text=True)

if result.returncode != 0:
    print(f"[ERROR] Evaluation script encountered an issue:\n{result.stderr}")
else:
    # The evaluation output is pure JSON, save it directly
    json_score_path = os.path.join(ARCHIVE_DIR, f"Score_{RUN_ID}.json")
    with open(json_score_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(f"[SUCCESS] Evaluation finished! Scores saved to: {json_score_path}")

# ---------- Step C: Archive Code and Predictions (Reproducibility) ----------
print("\n[INFO] [Step 3/3] Archiving current code and model outputs for reproducibility...")
# Backup the exact code snapshot used for this run
shutil.copy2(INFERENCE_SCRIPT, os.path.join(ARCHIVE_DIR, "code_snapshot.py"))
# Backup the generated JSON predictions
shutil.copytree(DEFAULT_OUTPUT_DIR, os.path.join(ARCHIVE_DIR, "model_predictions"), dirs_exist_ok=True)

print(f"\nPipeline executed successfully.")
print(f"Experiment Archive Location: {ARCHIVE_DIR}/")
print("=" * 60)