
"""
FEHU Fine-tuning Data Preprocessor v2
=======================================
Changes from v1:
  - system/instruction/output field structure (aligns with inference.py message roles)
  - Shorter system instruction: removed few-shot example (training samples ARE the few-shot)
  - Removed reasoning_process from output (not evaluated by FEHU; was noise from concatenated annotations)
  - Added level_1_values deduplication (fixes duplicate (value, direction) pairs)
  - Compact JSON output via separators=(',', ':') instead of indent=2 (saves ~15-20% tokens)
  - Removed leading whitespace from user_query f-string (minor token saving + cleaner)

Usage:
  python scripts/prepare_ft_data.py `
      --base_file  dataset/train/train_event_base.json `
      --gold_file  dataset/gold_labels/train/train_human_values.json `
      --output_file dataset/finetune/fehu_train.json

  python scripts/prepare_ft_data.py `
      --base_file  dataset/dev/dev_event_base.json `
      --gold_file  dataset/gold_labels/dev/dev_human_values.json `
      --output_file dataset/finetune/fehu_dev.json
"""

import json
import os
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIERARCHY_FILE = os.path.join(PROJECT_ROOT, "dataset/hv_categories/level_2_to_level_1.json")
L1_DICT_FILE   = os.path.join(PROJECT_ROOT, "dataset/hv_categories/human_value_level1_values.json")
L2_DICT_FILE   = os.path.join(PROJECT_ROOT, "dataset/hv_categories/human_value_level2_values.json")
DIR_DICT_FILE  = os.path.join(PROJECT_ROOT, "dataset/hv_categories/direction_mapping.json")


def deduplicate_l1(l1_list: list) -> list:
    """Remove duplicate (value, direction) pairs from a level_1_values list."""
    seen = set()
    unique = []
    for item in l1_list:
        key = (item["value"], item["direction"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file",   type=str,
        default=os.path.join(PROJECT_ROOT, "dataset/dev/dev_event_base.json"))
    parser.add_argument("--gold_file",   type=str,
        default=os.path.join(PROJECT_ROOT, "dataset/gold_labels/dev/dev_human_values.json"))
    parser.add_argument("--output_file", type=str,
        default=os.path.join(PROJECT_ROOT, "dataset/finetune/fehu_dev.json"))
    args = parser.parse_args()

    # ---- Load taxonomy and build reverse mappings (ID → Name) ----
    with open(L1_DICT_FILE,  'r', encoding='utf-8') as f: l1_name_to_id = json.load(f)
    with open(L2_DICT_FILE,  'r', encoding='utf-8') as f: l2_name_to_id = json.load(f)
    with open(DIR_DICT_FILE, 'r', encoding='utf-8') as f: dir_name_to_id = json.load(f)

    l1_rev  = {str(v): k for k, v in l1_name_to_id.items()}
    l2_rev  = {str(v): k for k, v in l2_name_to_id.items()}
    dir_rev = {str(v): k for k, v in dir_name_to_id.items()}

    with open(HIERARCHY_FILE, 'r', encoding='utf-8') as f:
        value_hierarchy = json.load(f)["level_2_to_level_1"]

    hierarchy_text = ""
    for l2_value, l1_list in value_hierarchy.items():
        hierarchy_text += f"- Level 2: {l2_value}\n"
        for l1_value in l1_list:
            hierarchy_text += f"  * Level 1: {l1_value}\n"

    # -----------------------------------------------------------------------
    # System instruction (v2):
    #   - No few-shot example: the 1228 training samples serve as implicit few-shot
    #   - Includes output format schema so model knows exact key names
    #   - Includes full taxonomy so model can produce exact string matches
    # -----------------------------------------------------------------------
    system_instruction = f"""You are a human value analysis expert.

Analyze the provided news article, subevents, and actors. Identify the human values demonstrated by each actor at both the article level and subevent level.

[OUTPUT FORMAT]
Output exactly ONE JSON object with the following structure:
{{
  "guid": "<guid>",
  "article_level_evaluation": [
    {{
      "actor": "<actor_id>",
      "level_1_values": [{{"value": "<exact_l1_name>", "direction": "<aligned|contradictory>"}}],
      "level_2_values": ["<exact_l2_name>"]
    }}
  ],
  "subevent_level_evaluation": [
    {{
      "subevent_id": "<subevent_id_string>",
      "actors": [
        {{
          "actor": "<actor_id>",
          "level_1_values": [{{"value": "<exact_l1_name>", "direction": "<aligned|contradictory>"}}],
          "level_2_values": ["<exact_l2_name>"]
        }}
      ]
    }}
  ]
}}

[RULES]
1. Use EXACT strings from the taxonomy below. Do NOT paraphrase, abbreviate, or invent values.
2. Each level_1 entry requires a direction: "aligned" or "contradictory".
3. Identify all values that are reasonably supported by the article.
4. Use the provided actor IDs exactly as given.
5. Output JSON only. No text, explanation, or markdown outside the JSON block.

[TAXONOMY]
{hierarchy_text}"""

    # ---- Load data ----
    with open(args.base_file, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    gold_by_guid = {str(item["guid"]): item for item in gold_data}

    # Detect subevent field name (handles official dataset typo "subvent_human_values")
    SE_FIELD = None
    for item in gold_data:
        for se in item.get("subevents_human_values", []):
            if "subvent_human_values" in se:
                SE_FIELD = "subvent_human_values"
            elif "subevent_human_values" in se:
                SE_FIELD = "subevent_human_values"
            break
        if SE_FIELD:
            break
    if SE_FIELD is None:
        SE_FIELD = "subvent_human_values"
    print(f"[INFO] Detected subevent field name: '{SE_FIELD}'")

    ft_dataset = []
    skipped = 0

    for item in base_data:
        guid = str(item.get("guid", ""))
        if guid not in gold_by_guid:
            skipped += 1
            continue

        gold_item = gold_by_guid[guid]
        content   = item.get("content", "")
        actors    = item.get("actors", [])
        subevents = item.get("subevents", [])

        actor_info = ""
        for actor_dict in actors:
            for name, actor_id in actor_dict.items():
                actor_info += f"- Actor Name: {name} (ID: {actor_id})\n"

        subevents_text = "\n".join(
            [f"[Subevent ID: {se['id']}] {se['subevent']}" for se in subevents]
        )

        # Note: no leading whitespace (avoids f-string indentation bleeding into string)
        user_query = f"""--- News Article ---
{content}

--- Subevents ---
{subevents_text}

--- Actors to Evaluate ---
{actor_info}
Output the JSON evaluation for the actors above. Ensure "guid" is strictly "{guid}"."""

        # ---- Article-level processing ----
        article_eval_map = defaultdict(lambda: {"level_1_values": [], "level_2_values": []})

        for hv in gold_item.get("article_human_values", []):
            actor_id = str(hv["actor"])

            # Level 2 (deduplicated via 'not in' check)
            l2_val = l2_rev.get(str(hv.get("l2_value", "")))
            if l2_val and l2_val not in article_eval_map[actor_id]["level_2_values"]:
                article_eval_map[actor_id]["level_2_values"].append(l2_val)

            # Level 1 (deduplicated at end via deduplicate_l1)
            l1_val  = l1_rev.get(str(hv.get("l1_value", "")))
            dir_val = dir_rev.get(str(hv.get("direction", "")))
            if l1_val and dir_val:
                article_eval_map[actor_id]["level_1_values"].append(
                    {"value": l1_val, "direction": dir_val}
                )

        article_eval_list = []
        for actor_id, data in article_eval_map.items():
            article_eval_list.append({
                "actor": actor_id,
                "level_1_values": deduplicate_l1(data["level_1_values"]),
                "level_2_values": data["level_2_values"],
            })

        # ---- Subevent-level processing ----
        subevent_eval_map = {}
        for se in gold_item.get("subevents_human_values", []):
            se_id = str(se["subevent_id"])
            actor_se_map = defaultdict(lambda: {"level_1_values": [], "level_2_values": []})

            for hv in se.get(SE_FIELD, []):
                actor_id = str(hv["actor"])

                l2_val = l2_rev.get(str(hv.get("l2_value", "")))
                if l2_val and l2_val not in actor_se_map[actor_id]["level_2_values"]:
                    actor_se_map[actor_id]["level_2_values"].append(l2_val)

                l1_val  = l1_rev.get(str(hv.get("l1_value", "")))
                dir_val = dir_rev.get(str(hv.get("direction", "")))
                if l1_val and dir_val:
                    actor_se_map[actor_id]["level_1_values"].append(
                        {"value": l1_val, "direction": dir_val}
                    )

            actors_list = []
            for actor_id, data in actor_se_map.items():
                actors_list.append({
                    "actor": actor_id,
                    "level_1_values": deduplicate_l1(data["level_1_values"]),
                    "level_2_values": data["level_2_values"],
                })

            if actors_list:
                subevent_eval_map[se_id] = {"subevent_id": se_id, "actors": actors_list}

        output_json = {
            "guid": guid,
            "article_level_evaluation": article_eval_list,
            "subevent_level_evaluation": list(subevent_eval_map.values()),
        }

        ft_dataset.append({
            "system":      system_instruction,
            "instruction": user_query,
            # compact JSON: no indent, minimal separators → ~15-20% fewer tokens
            "output":      json.dumps(output_json, ensure_ascii=False, separators=(',', ':')),
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(ft_dataset, f, ensure_ascii=False, indent=4)

    print(f"[SUCCESS] {len(ft_dataset)} samples saved to {args.output_file}")
    print(f"[INFO]    Skipped (no gold match): {skipped}")


if __name__ == "__main__":
    main()
