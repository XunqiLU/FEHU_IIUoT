import os
from modelscope import snapshot_download

# Define the target directory for model weights
MODEL_SAVE_DIR = "/root/autodl-tmp/FEHU/models"

print("Initializing model download from ModelScope: Llama-3.1-8B-Instruct...")
print(f"Target directory: {MODEL_SAVE_DIR}")
print("Downloading...")

# Download the model snapshot
model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    cache_dir=MODEL_SAVE_DIR
)

print("-" * 50)
print("Download completed successfully.")
print(f"Model saved to local path:\n{model_dir}")