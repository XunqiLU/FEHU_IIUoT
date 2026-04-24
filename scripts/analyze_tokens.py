import json
import numpy as np
from transformers import AutoTokenizer

# =====================================================================
# 1. Path Settings (Adjusted for server environment)
# =====================================================================
MODEL_PATH = "/root/autodl-tmp/FEHU/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"

# Official input feature file (X)
BASE_FILE = "/root/autodl-tmp/FEHU/dataset/train/train_event_base.json"
# Official gold labels file (Y)
VALUES_FILE = "/root/autodl-tmp/FEHU/dataset/gold_labels/train/train_human_values.json"

print("[INFO] Loading Llama-3 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# =====================================================================
# 2. Dataset Loading and Preprocessing
# =====================================================================
print("[INFO] Loading raw datasets...")
with open(BASE_FILE, 'r', encoding='utf-8') as f:
    train_base = json.load(f)
with open(VALUES_FILE, 'r', encoding='utf-8') as f:
    train_values = json.load(f)

# Convert gold labels (Y) to a dictionary keyed by GUID for O(1) lookup speed
values_dict = {item['guid']: item for item in train_values}

input_lengths = []
output_lengths = []
label_counts = []

print("[INFO] Scanning dataset and computing token lengths (this may take a few seconds)...")

# =====================================================================
# 3. Token Calculation
# =====================================================================
for article in train_base:
    guid = article.get("guid")

    # --- Calculate Input (X) Tokens ---
    content = article.get("content", "")
    # Extract subevent texts to simulate the LLM prompt context
    subevents_text = " ".join([se.get("subevent", "") for se in article.get("subevents", [])])
    input_text = f"Content: {content}\nSubevents: {subevents_text}"

    input_len = len(tokenizer.encode(input_text))
    input_lengths.append(input_len)

    # --- Calculate Output JSON (Y) Tokens ---
    # If gold labels exist for this article
    if guid in values_dict:
        gold_data = values_dict[guid]

        # Simulate the standard JSON structure output by the LLM
        gold_answer = {
            "article_human_values": gold_data.get("article_human_values", []),
            "subevents_human_values": gold_data.get("subevents_human_values", [])
        }
        # Add indentation (spaces and newlines consume tokens)
        output_text = json.dumps(gold_answer, indent=4)
        output_len = len(tokenizer.encode(output_text))
        output_lengths.append(output_len)

        # Count the total number of article-level human value labels annotated by experts
        label_counts.append(len(gold_data.get("article_human_values", [])))

# =====================================================================
# 4. Statistical Report Generation
# =====================================================================
print("\n" + "="*50)
print("Dataset Token Statistical Analysis Report")
print("="*50)
print(f"Max Input Tokens:                 {np.max(input_lengths)}")
print(f"95th Percentile Input Tokens:     {np.percentile(input_lengths, 95):.0f}")

if output_lengths:
    print(f"\nMax Output JSON Tokens:           {np.max(output_lengths)}")
    print(f"95th Percentile Output JSON:      {np.percentile(output_lengths, 95):.0f}")

    print(f"\nAverage article-level labels:     {np.mean(label_counts):.2f}")
    print(f"Max labels in a single article:   {np.max(label_counts)}")
else:
    print("\n[WARNING] No matching label data found. Please check file paths.")
print("="*50)

# =====================================================================
# Example Output Log
# =====================================================================
# Dataset Token Statistical Analysis Report
# ==================================================
# Max Input Tokens:                 8943
# 95th Percentile Input Tokens:     2470
#
# Max Output JSON Tokens:           7707
# 95th Percentile Output JSON:      4505
#
# Average article-level labels:     13.77
# Max labels in a single article:   37
# ==================================================