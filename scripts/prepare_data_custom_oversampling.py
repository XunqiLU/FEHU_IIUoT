import json
import os
import argparse
from pathlib import Path

# =====================================================================
# FEHU Data Preprocessor: Custom Oversampling Version
# =====================================================================
# This script is rewritten from scratch to clearly demonstrate the 
# Data Balancing and Oversampling logic, independently of the original 
# author's formatting code.
# =====================================================================

# The 10 rarest Level-1 IDs based on data distribution analysis
RARE_L1_IDS = {"31", "51", "8", "39", "35", "48", "34", "36", "23", "24"}
CONTRADICTORY_DIR_ID = "0"

def load_and_reverse_dict(filepath):
    """Loads a JSON dictionary and reverses its keys and values."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {str(v): str(k) for k, v in data.items()}

def build_taxonomy_text(hierarchy_file):
    """Builds a Markdown representation of the taxonomy tree."""
    with open(hierarchy_file, 'r', encoding='utf-8') as f:
        hierarchy = json.load(f)["level_2_to_level_1"]
    
    lines = []
    for l2, l1_list in hierarchy.items():
        lines.append(f"- Level 2: {l2}")
        for l1 in l1_list:
            lines.append(f"  * Level 1: {l1}")
    return "\n".join(lines)

def remove_duplicates(l1_values):
    """Removes duplicate Level-1 value/direction pairs."""
    unique_pairs = []
    seen = set()
    for item in l1_values:
        identifier = (item["value"], item["direction"])
        if identifier not in seen:
            seen.add(identifier)
            unique_pairs.append(item)
    return unique_pairs

def parse_labels_for_actors(raw_labels, l1_map, l2_map, dir_map):
    """
    Transforms flat label lists into a grouped structure per actor.
    (Rewritten to replace the original author's complex defaultdict logic).
    """
    actor_groups = {}
    
    for label in raw_labels:
        actor_id = str(label.get("actor"))
        if actor_id not in actor_groups:
            actor_groups[actor_id] = {
                "actor": actor_id,
                "level_1_values": [],
                "level_2_values": set() # Using set to auto-deduplicate Level 2
            }
            
        # Map IDs to human-readable strings
        l1_name = l1_map.get(str(label.get("l1_value")))
        l2_name = l2_map.get(str(label.get("l2_value")))
        direction_name = dir_map.get(str(label.get("direction")))
        
        if l2_name:
            actor_groups[actor_id]["level_2_values"].add(l2_name)
            
        if l1_name and direction_name:
            actor_groups[actor_id]["level_1_values"].append({
                "value": l1_name,
                "direction": direction_name
            })
            
    # Convert sets back to lists and deduplicate Level 1
    final_list = []
    for actor_id, data in actor_groups.items():
        final_list.append({
            "actor": actor_id,
            "level_1_values": remove_duplicates(data["level_1_values"]),
            "level_2_values": list(data["level_2_values"])
        })
        
    return final_list

# ---------------------------------------------------------------------
# CORE CONTRIBUTION: OVERSAMPLING LOGIC
# ---------------------------------------------------------------------
def calculate_sample_weight(gold_item, apply_oversampling):
    """
    Analyzes the gold labels of an article to determine its training weight.
    This is the core of the statistical intervention to balance the dataset.
    
    Returns: The number of times this sample should be duplicated.
    """
    if not apply_oversampling:
        return 1
        
    has_rare_label = False
    has_contradiction = False
    
    # Check Article-level labels
    for hv in gold_item.get("article_human_values", []):
        if str(hv.get("l1_value")) in RARE_L1_IDS:
            has_rare_label = True
        if str(hv.get("direction")) == CONTRADICTORY_DIR_ID:
            has_contradiction = True
            
    # Check Subevent-level labels (handling dataset typo)
    for se in gold_item.get("subevents_human_values", []):
        se_key = "subvent_human_values" if "subvent_human_values" in se else "subevent_human_values"
        for hv in se.get(se_key, []):
            if str(hv.get("l1_value")) in RARE_L1_IDS:
                has_rare_label = True
            if str(hv.get("direction")) == CONTRADICTORY_DIR_ID:
                has_contradiction = True
                
    # Determine copies based on priority
    if has_rare_label:
        return 4  # 1 original + 3 copies
    elif has_contradiction:
        return 2  # 1 original + 1 copy
        
    return 1 # Standard sample


def process_dataset(base_file, gold_file, out_file, dataset_dir, do_oversample):
    print("Loading data and dictionaries...")
    # Load dictionaries
    l1_map = load_and_reverse_dict(f"{dataset_dir}/hv_categories/human_value_level1_values.json")
    l2_map = load_and_reverse_dict(f"{dataset_dir}/hv_categories/human_value_level2_values.json")
    dir_map = load_and_reverse_dict(f"{dataset_dir}/hv_categories/direction_mapping.json")
    taxonomy_text = build_taxonomy_text(f"{dataset_dir}/hv_categories/level_2_to_level_1.json")
    
    # Load JSON files
    with open(base_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_labels = {str(item["guid"]): item for item in json.load(f)}

    system_prompt_template = f"""You are a human value analysis expert. Analyze the provided news article, subevents, and actors.

[OUTPUT FORMAT]
Output ONE JSON object matching this structure:
{{
  "guid": "<guid>",
  "article_level_evaluation": [
    {{"actor": "<actor_id>", "level_1_values": [{{"value": "<l1>", "direction": "<aligned|contradictory>"}}], "level_2_values": ["<l2>"]}}
  ],
  "subevent_level_evaluation": [
    {{"subevent_id": "<id>", "actors": [ ... ]}}
  ]
}}

[RULES]
1. Use EXACT strings from the taxonomy below.
2. Output JSON only.

[TAXONOMY]
{taxonomy_text}"""

    training_dataset = []
    
    print("Processing articles and applying Oversampling weights...")
    for article in articles:
        guid = str(article.get("guid"))
        if guid not in gold_labels:
            continue
            
        gold_item = gold_labels[guid]
        
        # 1. Format the Input Prompt
        actor_str = "\n".join([f"- {list(a.keys())[0]} (ID: {list(a.values())[0]})" for a in article.get("actors", [])])
        subevent_str = "\n".join([f"[{se['id']}] {se['subevent']}" for se in article.get("subevents", [])])
        
        user_prompt = f"--- News ---\n{article.get('content')}\n\n--- Subevents ---\n{subevent_str}\n\n--- Actors ---\n{actor_str}\n\nEnsure guid is strictly \"{guid}\"."

        # 2. Format the Output JSON
        target_json = {
            "guid": guid,
            "article_level_evaluation": parse_labels_for_actors(
                gold_item.get("article_human_values", []), l1_map, l2_map, dir_map
            ),
            "subevent_level_evaluation": []
        }
        
        # Parse subevents
        for se in gold_item.get("subevents_human_values", []):
            se_id = str(se["subevent_id"])
            se_key = "subvent_human_values" if "subvent_human_values" in se else "subevent_human_values"
            actors_parsed = parse_labels_for_actors(se.get(se_key, []), l1_map, l2_map, dir_map)
            if actors_parsed:
                target_json["subevent_level_evaluation"].append({
                    "subevent_id": se_id,
                    "actors": actors_parsed
                })

        # Compress JSON to save tokens
        compressed_output = json.dumps(target_json, ensure_ascii=False, separators=(',', ':'))
        
        sample = {
            "system": system_prompt_template,
            "instruction": user_prompt,
            "output": compressed_output
        }
        
        # 3. Calculate Weight and Append (OVERSAMPLING)
        copies_to_make = calculate_sample_weight(gold_item, do_oversample)
        training_dataset.extend([sample] * copies_to_make)

    # Save to file
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"[SUCCESS] Processed {len(articles)} unique articles into {len(training_dataset)} weighted training samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", required=True)
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--dataset_dir", default="/Users/lijx/paper/ntcir19_fehu/dataset")
    parser.add_argument("--oversample", action="store_true")
    args = parser.parse_args()
    
    process_dataset(args.base_file, args.gold_file, args.output_file, args.dataset_dir, args.oversample)