"""Evaluation and Benchmark Suite for Multi-label Content Tagging."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.taxonomy import TaxonomyManager
from src.preprocess import clean_article
from src.engine import TaggingEngine


def calculate_topk_precision_recall(
    y_true: List[List[str]],
    y_pred_ranked: List[List[Tuple[str, float]]],
    k: int = 3
) -> Tuple[float, float]:
    """Computes Precision@K and Recall@K over a list of ranked tag predictions."""
    p_scores = []
    r_scores = []

    for true_tags, ranked_preds in zip(y_true, y_pred_ranked):
        if not true_tags:
            continue
        top_k_preds = [tag for tag, _ in ranked_preds[:k]]
        true_set = set(true_tags)
        pred_set = set(top_k_preds)

        hits = len(true_set.intersection(pred_set))
        prec = hits / max(1, len(top_k_preds))
        rec = hits / max(1, len(true_set))

        p_scores.append(prec)
        r_scores.append(rec)

    return float(np.mean(p_scores) if p_scores else 0.0), float(np.mean(r_scores) if r_scores else 0.0)


def evaluate_model_on_dataset(
    engine: TaggingEngine,
    model_name: str,
    test_df: pd.DataFrame,
    threshold: float = 0.35,
    k: int = 3
) -> Dict[str, Any]:
    """Runs full evaluation on a held-out dataset."""
    mlb = MultiLabelBinarizer(classes=engine.taxonomy.all_approved_tags())
    all_approved = set(engine.taxonomy.all_approved_tags())

    y_true_list = []
    y_pred_binary_list = []
    y_pred_ranked_list = []

    inference_times = []
    review_routed_count = 0
    short_draft_count = 0

    error_examples = []

    for _, row in test_df.iterrows():
        title = row["title"]
        body = row["body"]
        raw_true = row.get("tags", [])
        if isinstance(raw_true, str):
            raw_true = json.loads(raw_true) if raw_true.startswith("[") else [raw_true]
        
        # Ground truth filtered to approved taxonomy
        true_tags = [t for t in raw_true if t in all_approved]
        y_true_list.append(true_tags)

        t0 = time.perf_counter()
        res = engine.predict(title, body, model_name=model_name, custom_threshold=threshold)
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

        if res["needs_review"]:
            review_routed_count += 1
            if res["status"] == "flagged_short_draft":
                short_draft_count += 1

        suggested = [item["tag"] for item in res["suggested_tags"]]
        ranked = [(item["tag"], item["score"]) for item in res["suggested_tags"]]

        # Binary multi-label prediction vector
        y_pred_binary = [1 if t in suggested else 0 for t in engine.taxonomy.all_approved_tags()]
        y_pred_binary_list.append(y_pred_binary)
        y_pred_ranked_list.append(ranked)

        # Collect error cases for analysis
        false_positives = set(suggested) - set(true_tags)
        false_negatives = set(true_tags) - set(suggested)
        if false_positives or false_negatives:
            if len(error_examples) < 5:
                error_examples.append({
                    "id": row.get("id", "N/A"),
                    "title": title,
                    "ground_truth": true_tags,
                    "predicted": suggested,
                    "false_positives": list(false_positives),
                    "false_negatives": list(false_negatives),
                    "status": res["status"]
                })

    Y_true_binary = mlb.fit_transform(y_true_list)
    Y_pred_binary = np.array(y_pred_binary_list)

    # Standard metrics
    micro_f1 = float(f1_score(Y_true_binary, Y_pred_binary, average="micro", zero_division=0))
    macro_f1 = float(f1_score(Y_true_binary, Y_pred_binary, average="macro", zero_division=0))
    micro_p = float(precision_score(Y_true_binary, Y_pred_binary, average="micro", zero_division=0))
    micro_r = float(recall_score(Y_true_binary, Y_pred_binary, average="micro", zero_division=0))

    p_at_k, r_at_k = calculate_topk_precision_recall(y_true_list, y_pred_ranked_list, k=k)

    total_posts = len(test_df)
    avg_latency_ms = float(np.mean(inference_times) * 1000)

    # Per-tag metrics
    per_tag_f1 = {}
    for i, tag in enumerate(engine.taxonomy.all_approved_tags()):
        tag_f1 = f1_score(Y_true_binary[:, i], Y_pred_binary[:, i], zero_division=0)
        support = int(np.sum(Y_true_binary[:, i]))
        per_tag_f1[tag] = {
            "f1": round(float(tag_f1), 4),
            "support": support
        }

    return {
        "model": model_name,
        "threshold": threshold,
        "total_test_articles": total_posts,
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        f"precision@{k}": round(p_at_k, 4),
        f"recall@{k}": round(r_at_k, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "review_coverage_rate": round(review_routed_count / max(1, total_posts), 4),
        "review_routed_count": review_routed_count,
        "short_draft_count": short_draft_count,
        "per_tag_metrics": per_tag_f1,
        "error_examples": error_examples
    }


def analyze_threshold_curve(
    engine: TaggingEngine,
    model_name: str,
    val_df: pd.DataFrame,
    thresholds: List[float] = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
) -> List[Dict[str, Any]]:
    """Analyzes trade-offs across different confidence thresholds on validation set."""
    curve = []
    for th in thresholds:
        res = evaluate_model_on_dataset(engine, model_name, val_df, threshold=th, k=3)
        curve.append({
            "threshold": th,
            "micro_f1": res["micro_f1"],
            "macro_f1": res["macro_f1"],
            "precision@3": res["precision@3"],
            "recall@3": res["recall@3"],
            "review_coverage_rate": res["review_coverage_rate"]
        })
    return curve
