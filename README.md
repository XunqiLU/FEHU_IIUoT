# NTCIR-19 FEHU (Fact-based Event-centric Human Value Understanding)

> **Team Name:** IIUoT<br>
> **Project Status:** Work in Progress

This repository contains the training and inference pipeline for the NTCIR-19 FEHU shared task. It supports both local inference using the `vLLM` engine and model fine-tuning utilizing `LLaMA-Factory`.

## Project Structure

```text
FEHU/
├── dataset/             # Official FEHU datasets (dev, train, etc.)
├── logs/                # Automated experiment archives (code snapshots, predictions, score reports)
├── models/              # Downloaded base models and exported fine-tuned models
├── outputs/             # Temporary inference outputs (Task 1 & Task 2 JSONs)
├── scripts/             # Utility scripts (e.g., download_model.py, analyze_tokens.py)
├── inference.py         # Local inference script utilizing vLLM (optimized for open-source models)
├── inference_api.py     # Asynchronous API inference script (e.g., Gemini, OpenAI)
├── evaluation.py        # Official evaluation and scoring script
├── run.py               # Main pipeline entry point (Execution -> Evaluation -> Archiving)
└── requirements.txt     # Python dependencies for the inference environment
```

---

## Part 1: Inference & Evaluation Pipeline

### 1. Environment Configuration

It is recommended to use an environment like an RTX 4090 for inference.

```bash
conda create -n fehu_env python=3.10 -y
conda activate fehu_env
pip install -r requirements.txt
```

### 2. Run the Automated Pipeline

Once your environment and model paths are correctly configured, you can execute the end-to-end workflow (Inference -> Evaluation -> Archiving) with a single command:

```bash
python run.py
```

*Note: If you are cloning this repository to a local machine or a different server path, you MUST manually update the `PROJECT_ROOT` variable in `run.py`, as well as the path configurations at the top of `inference.py` / `inference_api.py` to match your local paths.*

---

## Part 2: Fine-Tuning Workflow (LoRA)

The following guide details the process of fine-tuning models (e.g., Qwen3, Ministral) using a high-VRAM node (e.g., A800-80GB on AutoDL) and transferring the exported weights back for inference.

### 1. Training Environment Setup (A800 Node)

Initialize the `LLaMA-Factory` training environment:

```bash
# Create the training environment
conda create -n fehu_train python=3.11 -y
conda activate fehu_train

# Clone and install LLaMA-Factory
cd /root/autodl-tmp
git clone --depth 1 [https://github.com/hiyouga/LLaMA-Factory.git](https://github.com/hiyouga/LLaMA-Factory.git)
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 2. Download Base Models

Use `modelscope` to download the required base models:

```bash
conda activate fehu_env

# Download Qwen3-8B
modelscope download --model Qwen/Qwen3-8B \
    --local_dir /root/autodl-tmp/FEHU/models/Qwen/Qwen3-8B

# Download Ministral
modelscope download --model mistralai/Ministral-3-8B-Instruct-2512 \
    --local_dir /root/autodl-tmp/FEHU/models/mistralai/Ministral-3-8B-Instruct-2512
```

### 3. Register Datasets

Run `python scripts\prepare_ft_data.py` and upload training data (`fehu_train.json` and `fehu_dev.json`) to `/root/autodl-tmp/FEHU/dataset/finetune/`. Then, register them in LLaMA-Factory by editing `/root/autodl-tmp/LLaMA-Factory/data/dataset_info.json`:

```json
  "fehu_train": {
    "file_name": "/root/autodl-tmp/FEHU/dataset/finetune/fehu_train.json",
    "columns": {
      "system": "system",
      "prompt": "instruction",
      "response": "output"
    }
  },
  "fehu_dev": {
    "file_name": "/root/autodl-tmp/FEHU/dataset/finetune/fehu_dev.json",
    "columns": {
      "system": "system",
      "prompt": "instruction",
      "response": "output"
    }
  },
```

### 4. Execute Fine-Tuning

Upload `fehu_qwen3_lora_a800.yaml` and `fehu_ministral_lora_a800.yaml` to `/root/autodl-tmp/LLaMA-Factory/`. Start the training process using the configured YAML files.

```bash
cd /root/autodl-tmp/FEHU/LLaMA-Factory
conda activate fehu_train

# Train Qwen3
llamafactory-cli train fehu_qwen3_lora_a800.yaml

# Train Ministral
llamafactory-cli train fehu_ministral_lora_a800.yaml
```

### 5. Export and Merge LoRA Weights

Once training is complete, merge the LoRA adapters with the base models to export the final SFT models:

```bash
# Export Qwen3 SFT Model
llamafactory-cli export \
  --model_name_or_path /root/autodl-tmp/FEHU/models/Qwen/Qwen3-8B \
  --adapter_name_or_path /root/autodl-tmp/FEHU/models/Qwen/Qwen3-8B-FEHU-LoRA \
  --template qwen3_nothink \
  --finetuning_type lora \
  --export_dir /root/autodl-tmp/FEHU/models/Qwen/Qwen3-8B-FEHU-SFT \
  --export_size 2 \
  --export_device cpu \
  --export_legacy_format false

# Export Ministral SFT Model
llamafactory-cli export \
    --model_name_or_path /root/autodl-tmp/FEHU/models/mistralai/Ministral-3-8B-Instruct-2512 \
    --adapter_name_or_path /root/autodl-tmp/FEHU/models/mistralai/Ministral3-8B-FEHU-LoRA \
    --template mistral \
    --finetuning_type lora \
    --export_dir /root/autodl-tmp/FEHU/models/mistralai/Ministral3-8B-FEHU-SFT \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format false
```

### 7. Fix Ministral SFT Model

```bash
cp \
/root/autodl-tmp/FEHU/models/mistralai/Ministral-3-8B-Instruct-2512/tekken.json \
/root/autodl-tmp/FEHU/models/mistralai/Ministral3-8B-FEHU-SFT/

python3 << 'EOF'
import json

# Check the values in the original model config
with open('/root/autodl-tmp/FEHU/models/mistralai/Ministral-3-8B-Instruct-2512/config.json') as f:
    orig = json.load(f)
orig_arch = orig.get('text_config', {}).get('architectures')
print(f"original text_config.architectures = {orig_arch}")

# Fix SFT config
with open('/root/autodl-tmp/FEHU/models/mistralai/Ministral3-8B-FEHU-SFT/config.json') as f:
    sft = json.load(f)

arch = orig_arch if orig_arch else ['MistralForCausalLM']
sft['text_config']['architectures'] = arch
print(f"write in text_config.architectures = {arch}")

with open('/root/autodl-tmp/FEHU/models/mistralai/Ministral3-8B-FEHU-SFT/config.json', 'w') as f:
    json.dump(sft, f, indent=2)
EOF
```

### 7. Transfer Models for Inference

Transfer the merged models back to inference node.

After the transfer is complete, you can shut down the A800 instance and use `inference.py` on your 4090 node to test the newly fine-tuned models.
