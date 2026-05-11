#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Official evaluation script (simple):
- Each task reports: micro-F1 and macro-F1
- Macro-F1 is averaged over labels that appear at least once in GOLD (gold-supported macro).

Tasks:
- Task-1a: Article-level, Level-2 values (actor-conditioned, no direction)
- Task-1b: Article-level, Level-1 values + direction (actor-conditioned)
- Task-2a: Subevent-level, Level-2 values (actor-conditioned, no direction)
- Task-2b: Subevent-level, Level-1 values + direction (actor-conditioned)

Label spaces:
- Level-2: "0"..."19"
- Level-1 w/ direction: "direction:l1_value", direction in {"0","1"}, l1_value in "0"..."53"
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import sys

Label = str
InstanceId = Tuple[str, ...]  # (guid, actor) or (guid, subevent_id, actor)


# ---------------- Direction Reverse Rate (DRR) for task-xb ----------------

def split_dir_l1(label: str) -> Tuple[str, str]:
    """
    label format: "{direction}:{l1_value}" e.g., "0:42"
    direction: "0" (aligned) or "1" (contradictory)
    """
    d, v = label.split(":", 1)
    return d, v

def direction_reverse_rate_gold_excl(
    gold: Dict[InstanceId, Set[Label]],
    pred: Dict[InstanceId, Set[Label]],
) -> Tuple[float, int, int, int]:
    """
    DRR (gold-exclusive):
    - For each instance, if a value appears in BOTH directions in GOLD, it is ambiguous and excluded from DRR.
    - Count reversals only on direction-unique (exclusive) gold values.

    Returns:
      drr,
      reverse_count,
      denom_gold_excl_count,
      ambiguous_gold_value_count
    """
    instances = set(gold.keys()) | set(pred.keys())
    reverse_total = 0
    denom_total = 0
    ambiguous_total = 0

    for iid in instances:
        g_labels = gold.get(iid, set())
        p_labels = pred.get(iid, set())

        G_aligned: Set[str] = set()
        G_contra: Set[str] = set()
        for lab in g_labels:
            d, v = split_dir_l1(lab)
            if d == "0":
                G_aligned.add(v)
            else:
                G_contra.add(v)

        P_aligned: Set[str] = set()
        P_contra: Set[str] = set()
        for lab in p_labels:
            d, v = split_dir_l1(lab)
            if d == "0":
                P_aligned.add(v)
            else:
                P_contra.add(v)

        # ambiguous: same value has both directions in GOLD for this instance
        ambiguous = G_aligned & G_contra
        ambiguous_total += len(ambiguous)

        # exclusive gold values (direction is uniquely defined)
        G_only_aligned = G_aligned - G_contra
        G_only_contra = G_contra - G_aligned

        # reverse happens when prediction hits the opposite direction for an exclusive gold value
        flips = (P_aligned & G_only_contra) | (P_contra & G_only_aligned)

        reverse_total += len(flips)
        denom_total += (len(G_only_aligned) + len(G_only_contra))

    drr = safe_div(reverse_total, denom_total)
    return drr, reverse_total, denom_total, ambiguous_total


# ---------------- Metrics ----------------

def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d

def f1_from_pr(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)

def micro_counts(
    gold: Dict[InstanceId, Set[Label]],
    pred: Dict[InstanceId, Set[Label]],
    instances: Set[InstanceId],
) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for iid in instances:
        g = gold.get(iid, set())
        p = pred.get(iid, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return tp, fp, fn

def macro_f1_gold_supported(
    gold: Dict[InstanceId, Set[Label]],
    pred: Dict[InstanceId, Set[Label]],
    instances: Set[InstanceId],
    label_universe: Set[Label],
) -> float:
    # compute tp/fp/fn per label
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for iid in instances:
        g = gold.get(iid, set())
        p = pred.get(iid, set())
        for lab in (g & p):
            tp[lab] += 1
        for lab in (p - g):
            fp[lab] += 1
        for lab in (g - p):
            fn[lab] += 1

    # gold-supported labels: labels with gold positives > 0
    labels = [lab for lab in label_universe if (tp[lab] + fn[lab]) > 0]
    if not labels:
        return 0.0

    f1s: List[float] = []
    for lab in labels:
        p = safe_div(tp[lab], tp[lab] + fp[lab])
        r = safe_div(tp[lab], tp[lab] + fn[lab])
        f1s.append(f1_from_pr(p, r))
    return sum(f1s) / len(f1s)

def evaluate(
    gold: Dict[InstanceId, Set[Label]],
    pred: Dict[InstanceId, Set[Label]],
    label_universe: Set[Label],
) -> dict:
    instances = set(gold.keys()) | set(pred.keys())

    tp, fp, fn = micro_counts(gold, pred, instances)
    micro_p = safe_div(tp, tp + fp)
    micro_r = safe_div(tp, tp + fn)
    micro_f1 = f1_from_pr(micro_p, micro_r)

    macro_f1 = macro_f1_gold_supported(gold, pred, instances, label_universe)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "support_instances": len(instances),
        "support_gold_labels": sum(len(v) for v in gold.values()),
        "support_pred_labels": sum(len(v) for v in pred.values()),
    }


# ---------------- Parsers (match your output formats) ----------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_task1a(path: str) -> Dict[InstanceId, Set[Label]]:
    data = load_json(path)
    out: Dict[InstanceId, Set[Label]] = defaultdict(set)
    for item in data:
        guid = str(item["guid"])
        for hv in item.get("article_human_values", []):
            actor = str(hv["actor"])
            l2 = str(hv["l2_value"])
            out[(guid, actor)].add(l2)
    return dict(out)

def parse_task1b(path: str) -> Dict[InstanceId, Set[Label]]:
    data = load_json(path)
    out: Dict[InstanceId, Set[Label]] = defaultdict(set)
    for item in data:
        guid = str(item["guid"])
        for hv in item.get("article_human_values", []):
            actor = str(hv["actor"])
            direction = str(hv["direction"])
            l1 = str(hv["l1_value"])
            out[(guid, actor)].add(f"{direction}:{l1}")
    return dict(out)

def parse_task2a(path: str) -> Dict[InstanceId, Set[Label]]:
    data = load_json(path)
    out: Dict[InstanceId, Set[Label]] = defaultdict(set)
    for item in data:
        guid = str(item["guid"])
        for se in item.get("subevents_human_values", []):
            sid = str(se["subevent_id"])
            for hv in se.get("subevent_human_values", []):
                actor = str(hv["actor"])
                l2 = str(hv["l2_value"])
                out[(guid, sid, actor)].add(l2)
    return dict(out)

def parse_task2b(path: str) -> Dict[InstanceId, Set[Label]]:
    data = load_json(path)
    out: Dict[InstanceId, Set[Label]] = defaultdict(set)
    for item in data:
        guid = str(item["guid"])
        for se in item.get("subevents_human_values", []):
            sid = str(se["subevent_id"])
            for hv in se.get("subevent_human_values", []):
                actor = str(hv["actor"])
                direction = str(hv["direction"])
                l1 = str(hv["l1_value"])
                out[(guid, sid, actor)].add(f"{direction}:{l1}")
    return dict(out)


# ---------------- Label universes ----------------

def l2_universe() -> Set[Label]:
    return {str(i) for i in range(20)}  # "0"..."19"

def l1_dir_universe() -> Set[Label]:
    return {f"{d}:{i}" for d in ("0", "1") for i in range(54)}  # "0:0"... "1:53"


# ---------------- CLI ----------------

def main():

    ap = argparse.ArgumentParser("Simple micro/macro F1 evaluator (macro is gold-supported).")
    ap.add_argument("--gold_task1", type=str)
    ap.add_argument("--pred_task1a", type=str)
    ap.add_argument("--pred_task1b", type=str)
    ap.add_argument("--gold_task2", type=str)
    ap.add_argument("--pred_task2a", type=str)
    ap.add_argument("--pred_task2b", type=str)
    ap.add_argument("--out", type=str, default=None)

    # DEBUG = True: local run in pycharm/vscode
    # DEBUG = False: remote run
    DEBUG = False
    if DEBUG:
        sys.argv = [sys.argv[0]] + [
            "--gold_task1", "/tmp/ntcir19_fehu/dataset/gold_labels/test/test_article_human_values.json",
            "--pred_task1a", "/tmp/ntcir19_fehu/dataset/evaluation_test_data_examples/pred/task1/pred_task1a.json",
            "--pred_task1b", "/tmp/ntcir19_fehu/dataset/evaluation_test_data_examples/pred/task1/pred_task1b.json",
            "--pred_task2a", "/tmp/ntcir19_fehu/dataset/evaluation_test_data_examples/pred/task2/pred_task2a.json",
            "--gold_task2", "/tmp/ntcir19_fehu/dataset/gold_labels/test/test_subevent_human_values.json",
            "--pred_task2b", "/tmp/ntcir19_fehu/dataset/evaluation_test_data_examples/pred/task2/pred_task2b.json",
            "--out", "/tmp/ntcir19_fehu/output/evaluation_results.txt",
        ]

    args = ap.parse_args()

    results = {}

    # ---------------- (1) Parse all provided files ----------------
    gold_1a = pred_1a = None
    gold_1b = pred_1b = None
    gold_2a = pred_2a = None
    gold_2b = pred_2b = None

    if args.gold_task1 and args.pred_task1a:
        gold_1a = parse_task1a(args.gold_task1)
        pred_1a = parse_task1a(args.pred_task1a)

    if args.gold_task1 and args.pred_task1b:
        gold_1b = parse_task1b(args.gold_task1)
        pred_1b = parse_task1b(args.pred_task1b)

    if args.gold_task2 and args.pred_task2a:
        gold_2a = parse_task2a(args.gold_task2)
        pred_2a = parse_task2a(args.pred_task2a)

    if args.gold_task2 and args.pred_task2b:
        gold_2b = parse_task2b(args.gold_task2)
        pred_2b = parse_task2b(args.pred_task2b)

    # ---------------- (2) Compute F1 metrics (all tasks) ----------------
    if gold_1a is not None:
        results["task1a"] = evaluate(gold_1a, pred_1a, l2_universe())

    if gold_1b is not None:
        results["task1b"] = evaluate(gold_1b, pred_1b, l1_dir_universe())

    if gold_2a is not None:
        results["task2a"] = evaluate(gold_2a, pred_2a, l2_universe())

    if gold_2b is not None:
        results["task2b"] = evaluate(gold_2b, pred_2b, l1_dir_universe())

    # ---------------- (3) Compute DRR metrics (only task-xb) ----------------
    # DRR is computed on gold-exclusive values (filter ambiguous values with both directions in gold).
    if gold_1b is not None:
        drr, rev_cnt, denom_cnt, amb_cnt = direction_reverse_rate_gold_excl(gold_1b, pred_1b)
        results["task1b"].update({
            "direction_reverse_rate": drr,
            "direction_reverse_count": rev_cnt,
            "direction_reverse_denom_gold_excl": denom_cnt,
            "ambiguous_gold_values_filtered": amb_cnt
        })

    if gold_2b is not None:
        drr, rev_cnt, denom_cnt, amb_cnt = direction_reverse_rate_gold_excl(gold_2b, pred_2b)
        results["task2b"].update({
            "direction_reverse_rate": drr,
            "direction_reverse_count": rev_cnt,
            "direction_reverse_denom_gold_excl": denom_cnt,
            "ambiguous_gold_values_filtered": amb_cnt
        })

    if not results:
        raise SystemExit("No tasks evaluated. Provide at least one (gold, pred) pair.")

    text = json.dumps(results, ensure_ascii=False, indent=2)
    # if args.out:
    #     with open(args.out, "w", encoding="utf-8") as f:
    #         f.write(text)
    print(text)

if __name__ == "__main__":
    main()
