"""
现在的llm调用方式都是借用公司内部的 所以没有贴出来
能参考的好像只有instruction和json_repair
"""

import sys
import json
import os
import subprocess
import concurrent.futures
import time

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


# Add apipe to path
sys.path.append("/vepfs/home/zhuangzhiying/sigma/code")
sys.path.append("/vepfs/home/zhuangzhiying/sigma/code/apipe")

from ops.op_prompt_enhance import OpPromptEnhance


def main():
    print("Loading data...")
    # dev datasets
    data_path = "/vepfs/home/zhuangzhiying/sigma/code/ntcir/ntcir19_fehu/dataset/dev/dev_event_base.json"
    gold_data_path = (
        "/vepfs/home/zhuangzhiying/sigma/code/ntcir/ntcir19_fehu/dataset/gold_labels/dev/dev_human_values.json"
    )

    with open(data_path, "r") as f:
        events = json.load(f)
    with open(gold_data_path, "r") as f:
        golds = json.load(f)

    print(f"Loaded {len(events)} events from Dev Set.")

    # 跑全量验证集
    test_events = events

    op = OpPromptEnhance({})

    prompt_template = """
You are an expert sociologist and data annotator specializing in the Schwartz Theory of Basic Human Values. 
Your task is to exhaustively analyze factual news articles and their subevents to identify ALL human values expressed by the main actors.

### 1. Task Definition & Extraction Guidelines
You will be provided with a news article (`content`), a list of main `actors` (with their IDs), and a list of `subevents` (with their IDs and descriptions).
You need to perform two levels of analysis:

**Level 1: Article-level (Macro)**
- Based on the ENTIRE article, identify the human values expressed by each actor.
- **CRITICAL**: Be EXHAUSTIVE. An actor often expresses multiple values simultaneously through different actions or statements across the article. Do not limit yourself to only the most obvious value. Extract ALL applicable values.
- If an actor's wording slightly differs from the 
**Level 2: Subevent-level (Micro)**
- Focus STRICTLY on the text of each specific subevent. 
- Identify the human values expressed by the actors *within that specific subevent*. 
- **CRITICAL**: Only extract a value if it is clearly implied by the actor's action/words in THAT specific sentence. Do not hallucinate values from the broader article context into a specific subevent.

For each identified value, you must provide:
1. `l1_value`: The ID of the Level-1 value (0 to 53).
2. `l2_value`: The ID of the corresponding Level-2 value (0 to 19).
3. `direction`: "1" (Aligned) if the actor's actions/words support, uphold, or protect the value. "0" (Contradictory) if the actor's actions/words violate, oppose, or undermine the value.
4. `explanation`: A brief reasoning explaining WHAT the actor did/said, WHY it reflects this specific value, and WHY the direction is 1 or 0.

### 2. Value Taxonomy (Label Space)
You MUST ONLY use the following IDs for Level-1 and Level-2 values. 
[Level-2 Values]:
0: Self-direction: thought, 1: Self-direction: action, 2: Stimulation, 3: Hedonism, 4: Achievement, 5: Power: dominance, 6: Power: resources, 7: Face, 8: Security: personal, 9: Security: societal, 10: Tradition, 11: Conformity: rules, 12: Conformity: interpersonal, 13: Humility, 14: Benevolence: caring, 15: Benevolence: dependability, 16: Universalism: concern, 17: Universalism: nature, 18: Universalism: tolerance, 19: Universalism: objectivity.

[Level-1 Values and their corresponding Level-2 IDs]:
0: Be creative (L2: 0), 1: Be curious (L2: 0), 2: Have freedom of thought (L2: 0)
3: Be choosing own goals (L2: 1), 4: Be independent (L2: 1), 5: Have freedom of action (L2: 1), 6: Have privacy (L2: 1)
7: Have an exciting life (L2: 2), 8: Have a varied life (L2: 2), 9: Be daring (L2: 2)
10: Have pleasure (L2: 3)
11: Be ambitious (L2: 4), 12: Have success (L2: 4), 13: Be capable (L2: 4), 14: Be intellectual (L2: 4), 15: Be courageous (L2: 4)
16: Have influence (L2: 5), 17: Have the right to command (L2: 5)
18: Have wealth (L2: 6)
19: Have social recognition (L2: 7), 20: Have a good reputation (L2: 7)
21: Have a sense of belonging (L2: 8), 22: Have good health (L2: 8), 23: Have no debts (L2: 8), 24: Be neat and tidy (L2: 8), 25: Have a comfortable life (L2: 8)
26: Have a safe country (L2: 9), 27: Have a stable society (L2: 9)
28: Be respecting traditions (L2: 10), 29: Be holding religious faith (L2: 10)
30: Be compliant (L2: 11), 31: Be self-disciplined (L2: 11), 32: Be behaving properly (L2: 11)
33: Be polite (L2: 12), 34: Be honoring elders (L2: 12)
35: Be humble (L2: 13), 36: Have life accepted as is (L2: 13)
37: Be helpful (L2: 14), 38: Be honest (L2: 14), 39: Be forgiving (L2: 14), 40: Have the own family secured (L2: 14), 41: Be loving (L2: 14)
42: Be responsible (L2: 15), 43: Have loyalty towards friends (L2: 15)
44: Have equality (L2: 16), 45: Be just (L2: 16), 46: Have a world at peace (L2: 16)
47: Be protecting the environment (L2: 17), 48: Have harmony with nature (L2: 17), 49: Have a world of beauty (L2: 17)
50: Be broadminded (L2: 18), 51: Have the wisdom to accept others (L2: 18)
52: Be logical (L2: 19), 53: Have an objective view (L2: 19)

### 3. Demonstration (Few-Shot Example)
**Input:**
{
  "guid": "17-27710-2-3",
  "content": "The Russian ransomware group responsible for the Colonial Pipeline hack, Darkside, reaped just over $90 million in Bitcoin ransom payments... Colonial Pipeline paid a nearly $5 million ransom to Darkside to get back online... The hack has caused consumers to panic buying gasoline...",
  "actors": [
    {"Darkside and Affiliates (Ransomware Operators)": "17-27710-2-3-0"},
    {"Victim Organizations (e.g., Colonial Pipeline)": "17-27710-2-3-1"},
    {"General Public / Consumers": "17-27710-2-3-2"}
  ],
  "subevents": [
    {"id": "0", "subevent": "Darkside, a Russian ransomware group, received over $90 million in Bitcoin ransom payments from 47 victims before announcing it would cease operations."},
    {"id": "8", "subevent": "Colonial Pipeline paid nearly $5 million ransom to Darkside to restore operations."}
  ]
}

**Expected Output:**
{
  "article_human_values": [
    {
      "actor": "17-27710-2-3-0",
      "l1_value": "18",
      "l2_value": "6",
      "explanation": "The group's primary motivation is financial gain, having successfully extorted over $90 million. This aligns with the value of accumulating wealth.",
      "direction": "1"
    },
    {
      "actor": "17-27710-2-3-0",
      "l1_value": "38",
      "l2_value": "14",
      "explanation": "Darkside's ransomware operations involve extortion and deception, directly violating principles of honesty.",
      "direction": "0"
    },
    {
      "actor": "17-27710-2-3-1",
      "l1_value": "42",
      "l2_value": "15",
      "explanation": "Colonial Pipeline's efforts to restore operations demonstrate corporate responsibility to mitigate public harm.",
      "direction": "1"
    }
  ],
  "subevents_human_values": [
    {
      "subevent_id": "0",
      "subevent_human_values": [
        {
          "actor": "17-27710-2-3-0",
          "l1_value": "18",
          "l2_value": "6",
          "explanation": "Receiving over $90 million in ransom payments is a direct and successful pursuit of wealth.",
          "direction": "1"
        },
        {
          "actor": "17-27710-2-3-0",
          "l1_value": "27",
          "l2_value": "9",
          "explanation": "Large-scale cybercrime operations erode trust in digital infrastructure essential for societal stability.",
          "direction": "0"
        }
      ]
    },
    {
      "subevent_id": "8",
      "subevent_human_values": [
        {
          "actor": "17-27710-2-3-1",
          "l1_value": "18",
          "l2_value": "6",
          "explanation": "The victims paying a nearly $5 million ransom represents a significant financial loss, opposing their interest in preserving wealth.",
          "direction": "0"
        }
      ]
    }
  ]
}

### 4. Output Constraints
- Output ONLY a valid JSON object matching the structure of the Expected Output above.
- Do NOT include any markdown formatting (like ```json), HTML, or conversational text.
- Ensure all `l1_value`, `l2_value`, and `direction` fields are STRINGS (e.g., "18", not 18).
- Ensure that the `l2_value` correctly corresponds to the `l1_value` according to the Taxonomy provided.

### 5. Input Data to Analyze
Now, thoroughly analyze the following input data and extract ALL applicable values. Generate the JSON output:

{INPUT_JSON_DATA_HERE}
"""

    def process_event(ev, max_retries=3):
        prompt = prompt_template.replace("{INPUT_JSON_DATA_HERE}", json.dumps(ev, ensure_ascii=False))

        for attempt in range(max_retries):
            try:
                # 调用qwen-max，op_prompt_enhance中会自动处理
                res = op(
                    {"model": "doubao-seed-1-6-251015", "format_type"timeout": 120, "retries": 3},
                    " ",
                    None,
                    prompt,
                )

                if res is None:
                    continue

                # Post-process res
                import json_repair

                pred_obj = json_repair.repair_json(res, return_objects=True)
                if not isinstance(pred_obj, dict):
                    raise ValueError("Repaired JSON is not a dictionary")

                pred_obj["guid"] = ev["guid"]
                return pred_obj
            except Exception as e:
                # Log and retry
                time.sleep(1)
                if attempt == max_retries - 1:
                    print(f"Failed processing {ev['guid']} after {max_retries} attempts: {e}")
                    return None
        return None

    preds = []
    # 线程数
    max_workers = 10
    print(f"Start concurrent predicting with {max_workers} workers on {len(test_events)} events...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor    futures = {executor.submit(process_event, ev): ev for ev in test_events}

        # 显示进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(test_events)):
            pred_obj = future.result()
            if pred_obj is not None:
                preds.append(pred_obj)

 th = "./pred_dev.json"
    with open(pred_path, "w") as f:
        json.dump(preds, f)

    print(f"Done predicting. Total successful predictions: {len(preds)}/{len(test_events)}")
    print("Running evaluation...")

    # Run evaluation script (使用全量 dev_human_values.json)
    cmd = [
        "python3",
        "./evaluation.py",
        "--gold_task1",
        gold_data_path,
        "--gold_task2",
        gold_data_path,
        "--pred_task1a",
        pred_path,
        "--pred_task1b",
        pred_path,
        "--pred_task2a",
        pred_path, "--pred_task2b",
        pred_path,
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

