import json
import os
import argparse
from typing import Dict, List, Any

def load_json(filepath: str) -> Any:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_fewshot_database(
    train_event_base_path: str,
    train_human_values_path: str,
    l1_dict_path: str,
    l2_dict_path: str,
    dir_dict_path: str,
    output_path: str
):
    print("Loading datasets and dictionaries...")
    # 1. Load data
    event_base = load_json(train_event_base_path)
    human_values = load_json(train_human_values_path)
    
    # 2. Load dictionaries and reverse mapping
    l1_id_to_name = {v: k for k, v in load_json(l1_dict_path).items()}
    l2_id_to_name = {v: k for k, v in load_json(l2_dict_path).items()}
    # Direction is typically 1 (aligned) or 2 (contradictory) in official data, need to map back to string
    # Assuming dir_dict maps string to id (e.g., {"aligned": "1"}), we reverse it
    dir_id_to_name = {str(v): k for k, v in load_json(dir_dict_path).items()}

    print(f"Loaded {len(event_base)} articles from event base.")

    # 3. Create a lookup dictionary for human values by guid
    values_lookup: Dict[str, Dict] = {item["guid"]: item for item in human_values}
    
    fewshot_database = []
    
    print("Processing and transforming data into unified few-shot format...")
    for article in event_base:
        guid = article.get("guid")
        if not guid or guid not in values_lookup:
            continue
            
        gold_labels = values_lookup[guid]
        
        # We will build the exact JSON structure expected by the model
        transformed_labels = {
            "guid": guid,
            "article_level_evaluation": [],
            "subevent_level_evaluation": []
        }
        
        # --- Process Article Level ---
        article_evals = gold_labels.get("article_human_values", [])
        # Group by actor
        actor_article_map: Dict[str, Dict] = {}
        for ev in article_evals:
            actor_id = ev.get("actor")
            if not actor_id:
                continue
            if actor_id not in actor_article_map:
                actor_article_map[actor_id] = {
                    "actor": actor_id,
                    "reasoning_process_parts": [],
                    "level_1_values": [],
                    "level_2_values": set()
                }
            
            l1_id = ev.get("l1_value")
            l2_id = ev.get("l2_value")
            direction_id = ev.get("direction")
            explanation = ev.get("explanation", "")
            
            if explanation:
                actor_article_map[actor_id]["reasoning_process_parts"].append(explanation)
                
            if l1_id and direction_id:
                l1_name = l1_id_to_name.get(str(l1_id))
                dir_name = dir_id_to_name.get(str(direction_id), "aligned")
                if l1_name:
                    # Avoid duplicates
                    if not any(v["value"] == l1_name for v in actor_article_map[actor_id]["level_1_values"]):
                         actor_article_map[actor_id]["level_1_values"].append({
                             "value": l1_name,
                             "direction": dir_name
                         })
                         
            if l2_id:
                l2_name = l2_id_to_name.get(str(l2_id))
                if l2_name:
                    actor_article_map[actor_id]["level_2_values"].add(l2_name)
                    
        # Finalize article level formatting
        for actor_id, data in actor_article_map.items():
            # Synthesize reasoning process
            reasoning = " ".join(data["reasoning_process_parts"]) if data["reasoning_process_parts"] else "Derived from context."
            
            transformed_labels["article_level_evaluation"].append({
                "actor": actor_id,
                "reasoning_process": reasoning,
                "level_1_values": data["level_1_values"],
                "level_2_values": list(data["level_2_values"])
            })
            
        # --- Process Subevent Level ---
        subevent_evals = gold_labels.get("subevents_human_values", [])
        subevent_map: Dict[str, Dict[str, Dict]] = {} # subevent_id -> {actor_id -> data}
        
        for ev in subevent_evals:
            se_id = str(ev.get("subevent_id", ""))
            actor_id = ev.get("actor")
            if not actor_id:
                continue
            
            if se_id not in subevent_map:
                subevent_map[se_id] = {}
                
            if actor_id not in subevent_map[se_id]:
                subevent_map[se_id][actor_id] = {
                    "actor": actor_id,
                    "reasoning_process_parts": [],
                    "level_1_values": [],
                    "level_2_values": set()
                }
                
            l1_id = ev.get("l1_value")
            l2_id = ev.get("l2_value")
            direction_id = ev.get("direction")
            explanation = ev.get("explanation", "")
            
            if explanation:
                subevent_map[se_id][actor_id]["reasoning_process_parts"].append(explanation)
                
            if l1_id and direction_id:
                l1_name = l1_id_to_name.get(str(l1_id))
                dir_name = dir_id_to_name.get(str(direction_id), "aligned")
                if l1_name:
                    if not any(v["value"] == l1_name for v in subevent_map[se_id][actor_id]["level_1_values"]):
                         subevent_map[se_id][actor_id]["level_1_values"].append({
                             "value": l1_name,
                             "direction": dir_name
                         })
                         
            if l2_id:
                l2_name = l2_id_to_name.get(str(l2_id))
                if l2_name:
                    subevent_map[se_id][actor_id]["level_2_values"].add(l2_name)
                    
        # Finalize subevent level formatting
        for se_id, actors_dict in subevent_map.items():
            se_actors_list = []
            for actor_id, data in actors_dict.items():
                reasoning = " ".join(data["reasoning_process_parts"]) if data["reasoning_process_parts"] else "Derived from context."
                se_actors_list.append({
                    "actor": actor_id,
                    "reasoning_process": reasoning,
                    "level_1_values": data["level_1_values"],
                    "level_2_values": list(data["level_2_values"])
                })
                
            transformed_labels["subevent_level_evaluation"].append({
                "subevent_id": se_id,
                "actors": se_actors_list
            })
            
        # Add to final database
        fewshot_database.append({
            "guid": guid,
            "input_context": {
                "content": article.get("content", ""),
                "actors": article.get("actors", []),
                "subevents": article.get("subevents", [])
            },
            "target_output": transformed_labels
        })
        
    print(f"Successfully processed {len(fewshot_database)} unified examples.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fewshot_database, f, ensure_ascii=False, indent=2)
        
    print(f"Few-shot database saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FEHU Few-Shot Database")
    parser.add_argument("--dataset_root", type=str, default="/Users/lijx/paper/ntcir19_fehu/dataset")
    parser.add_argument("--output_path", type=str, default="/Users/lijx/paper/FEHU_IIUoT/dataset/fewshot_database.json")
    
    args = parser.parse_args()
    
    build_fewshot_database(
        train_event_base_path=os.path.join(args.dataset_root, "train/train_event_base.json"),
        train_human_values_path=os.path.join(args.dataset_root, "gold_labels/train/train_human_values.json"),
        l1_dict_path=os.path.join(args.dataset_root, "hv_categories/human_value_level1_values.json"),
        l2_dict_path=os.path.join(args.dataset_root, "hv_categories/human_value_level2_values.json"),
        dir_dict_path=os.path.join(args.dataset_root, "hv_categories/direction_mapping.json"),
        output_path=args.output_path
    )
