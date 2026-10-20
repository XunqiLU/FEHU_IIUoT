import os
from modelscope import snapshot_download

# Specify the model ID and download path
# MODEL_ID = "LLM-Research/Meta-Llama-3.1-8B-Instruct"  # TODO: Change model here
MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512"
# MODEL_ID = "Qwen/Qwen3-8B"
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

MODEL_SAVE_DIR = "/root/autodl-tmp/FEHU/models"

print("Initializing model download from ModelScope: {MODEL_ID}...")
print(f"Target directory: {MODEL_SAVE_DIR}")
print("Downloading...")

# Download the model snapshot
model_dir = snapshot_download(
    MODEL_ID,
    cache_dir=MODEL_SAVE_DIR
)

print("-" * 50)
print("Download completed successfully.")
print(f"Model saved to local path:\n{model_dir}")