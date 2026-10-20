# NTCIR-19 FEHU (Fact-based Event-centric Human Value Understanding)

> **Team Name:** IIUoT<br>

## Quick Start

**1. Environment Configuration**

```bash
conda create -n fehu_env python=3.10
conda activate fehu_env
pip install -r requirements.txt
```

**2. Run the Automated Pipeline**

Once your environment and paths are correctly configured, you can execute the end-to-end workflow (Inference -> Evaluation -> Archiving) with a single command:

```bash
python run.py
```

*Note: If you are cloning this repository to a local machine or a different server path, you MUST manually update the `PROJECT_ROOT` variable in `run.py`, as well as the path configurations at the top of `inference.py` / `inference_api.py` to match your paths.*

## Project Structure
* `run.py`: **Main Pipeline Entry Point**. Executes the inference script, triggers the official evaluation, and automatically archives code snapshots, JSON predictions, and score reports into the `logs/` directory.
* `inference.py`: Local inference script utilizing the vLLM engine, optimized for open-source models (e.g., Llama-3.1-8B).
* `inference_api.py`: Asynchronous concurrency script designed for commercial APIs (e.g., Gemini, OpenAI).
* `evaluation.py`: The official evaluation and scoring script.
* `dataset/`: The official FEHU datasets.
