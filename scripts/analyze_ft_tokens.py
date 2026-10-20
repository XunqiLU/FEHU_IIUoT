import json
import numpy as np
from transformers import AutoTokenizer

MODEL_PATH = "/root/autodl-tmp/FEHU/models/Qwen/Qwen3-8B"
TRAIN_FILE = "/root/autodl-tmp/FEHU/dataset/finetune/fehu_train.json"
DEV_FILE   = "/root/autodl-tmp/FEHU/dataset/finetune/fehu_dev.json"

print("[INFO] Loading Qwen3 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

total_lengths = []

for filepath, split in [(TRAIN_FILE, "train"), (DEV_FILE, "dev")]:
    print(f"[INFO] Scanning {split} set...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lengths = []
    for sample in data:
        # 拼接完整训练文本（instruction + input + output）
        full_text = (
            sample.get("system", "")
            + "\n"
            + sample.get("instruction", "")
            + "\n"
            + sample.get("output", "")
        )
        length = len(tokenizer.encode(full_text))
        lengths.append(length)

    total_lengths.extend(lengths)
    print(f"  [{split}] samples: {len(lengths)}")
    print(f"  [{split}] Max tokens:        {np.max(lengths)}")
    print(f"  [{split}] 50th percentile:   {np.percentile(lengths, 50):.0f}")
    print(f"  [{split}] 90th percentile:   {np.percentile(lengths, 90):.0f}")
    print(f"  [{split}] 95th percentile:   {np.percentile(lengths, 95):.0f}")
    print(f"  [{split}] 99th percentile:   {np.percentile(lengths, 99):.0f}")

print("\n" + "="*50)
print(f"Overall max tokens:       {np.max(total_lengths)}")
print(f"Overall 95th percentile:  {np.percentile(total_lengths, 95):.0f}")
print(f"Overall 99th percentile:  {np.percentile(total_lengths, 99):.0f}")
print("="*50)


# [INFO] Loading Qwen3 Tokenizer...
# [INFO] Scanning train set...
#   [train] samples: 1228
#   [train] Max tokens:        10761
#   [train] 50th percentile:   3132
#   [train] 90th percentile:   4278
#   [train] 95th percentile:   4937
#   [train] 99th percentile:   8443
# [INFO] Scanning dev set...
#   [dev] samples: 177
#   [dev] Max tokens:        9597
#   [dev] 50th percentile:   3089
#   [dev] 90th percentile:   4387
#   [dev] 95th percentile:   5115
#   [dev] 99th percentile:   8510

# ==================================================
# Overall max tokens:       10761
# Overall 95th percentile:  5004
# Overall 99th percentile:  8506
# ==================================================