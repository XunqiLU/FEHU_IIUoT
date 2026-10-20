import json
import os
from collections import Counter

def analyze_data(base_file, gold_file):
    with open(base_file, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    gold_by_guid = {str(item["guid"]): item for item in gold_data}

    total_articles = len(base_data)
    total_actors_in_articles = 0
    total_subevents = 0
    total_actors_in_subevents = 0

    article_actors_with_values = 0
    subevent_actors_with_values = 0

    l1_counter = Counter()
    l2_counter = Counter()
    direction_counter = Counter()

    for item in base_data:
        guid = str(item.get("guid", ""))
        
        # Count actors in base
        actors = item.get("actors", [])
        total_actors_in_articles += len(actors)
        
        subevents = item.get("subevents", [])
        total_subevents += len(subevents)
        total_actors_in_subevents += len(subevents) * len(actors) # roughly, each actor could participate

        if guid not in gold_by_guid:
            continue

        gold_item = gold_by_guid[guid]

        # Article Level analysis
        article_evals = gold_item.get("article_human_values", [])
        actors_seen = set()
        for hv in article_evals:
            actor = str(hv.get("actor", ""))
            actors_seen.add(actor)
            
            l1 = hv.get("l1_value")
            l2 = hv.get("l2_value")
            dir_val = hv.get("direction")
            
            if l1: l1_counter[str(l1)] += 1
            if l2: l2_counter[str(l2)] += 1
            if dir_val: direction_counter[str(dir_val)] += 1
            
        article_actors_with_values += len(actors_seen)

        # Subevent Level analysis
        se_key = "subvent_human_values"
        for se in gold_item.get("subevents_human_values", []):
            if "subevent_human_values" in se:
                se_key = "subevent_human_values"
                break
                
        for se in gold_item.get("subevents_human_values", []):
            actors_seen_se = set()
            for hv in se.get(se_key, []):
                actor = str(hv.get("actor", ""))
                actors_seen_se.add(actor)
                
                l1 = hv.get("l1_value")
                l2 = hv.get("l2_value")
                dir_val = hv.get("direction")
                
                if l1: l1_counter[str(l1)] += 1
                if l2: l2_counter[str(l2)] += 1
                if dir_val: direction_counter[str(dir_val)] += 1
                
            subevent_actors_with_values += len(actors_seen_se)

    print("=== FEHU Data Distribution Analysis ===")
    print(f"Total Articles: {total_articles}")
    print(f"Total Article-Level Actors Evaluated (with values): {article_actors_with_values} / {total_actors_in_articles} possible")
    print(f"Total Subevent-Level Actors Evaluated (with values): {subevent_actors_with_values} / {total_actors_in_subevents} possible")
    print(f"\nDirections Distribution:")
    for k, v in direction_counter.items():
        print(f"  {k} (1=aligned, 0=contradictory): {v} ({(v/sum(direction_counter.values())*100):.2f}%)")
        
    print(f"\nTop 5 Level-1 Values:")
    for k, v in l1_counter.most_common(5):
        print(f"  ID {k}: {v}")
    
    print(f"\nBottom 10 Level-1 Values (Long tail):")
    for k, v in l1_counter.most_common()[-10:]:
        print(f"  ID {k}: {v}")

if __name__ == "__main__":
    analyze_data(
        "/Users/lijx/paper/ntcir19_fehu/dataset/train/train_event_base.json",
        "/Users/lijx/paper/ntcir19_fehu/dataset/gold_labels/train/train_human_values.json"
    )
