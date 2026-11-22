"""Master training, validation threshold tuning, and testing pipeline using official dataset."""

import csv
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.taxonomy import TaxonomyManager
from src.preprocess import clean_article
from src.engine import TaggingEngine


def calculate_precision_recall_at_k(y_true_list, y_pred_list, k=3):
    p_list = []
    r_list = []
    for true_tags, pred_tags in zip(y_true_list, y_pred_list):
        if not true_tags:
            continue
        top_k = pred_tags[:k]
        hits = len(set(true_tags) & set(top_k))
        # denominator is defined as max(1, len(top_k))
        p = hits / max(1, len(top_k))
        r = hits / max(1, len(true_tags))
        p_list.append(p)
        r_list.append(r)
    return float(np.mean(p_list) if p_list else 0.0), float(np.mean(r_list) if r_list else 0.0)


def run_pipeline():
    print("=" * 60)
    print("1. Loading Official Dataset and Taxonomy")
    print("=" * 60)

    taxonomy = TaxonomyManager("data")
    print(f"Loaded {len(taxonomy.all_categories())} categories, {len(taxonomy.all_approved_tags())} canonical tags.")

    articles_path = "data/articles.csv"
    labels_path = "data/labels.csv"

    with open(articles_path, "r", encoding="utf-8") as f:
        articles = list(csv.DictReader(f))
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_map = {r["article_id"]: r for r in csv.DictReader(f)}

    train_arts = [a for a in articles if a["split"] == "train"]
    val_arts = [a for a in articles if a["split"] == "validation"]
    test_arts = [a for a in articles if a["split"] == "test"]

    print(f"Dataset split: Train={len(train_arts)}, Validation={len(val_arts)}, Test={len(test_arts)}")

    print("\n" + "=" * 60)
    print("2. Training Multi-Label TF-IDF & Semantic Models")
    print("=" * 60)

    engine = TaggingEngine(taxonomy)

    print("Training TF-IDF Multi-label model on Train split...")
    engine.tfidf_model.fit(articles_path, labels_path)
    engine.tfidf_model.save("models/tfidf_model.joblib")
    print("Saved TF-IDF model to models/tfidf_model.joblib")

    print("Training Semantic Embedding model on Train split...")
    engine.semantic_model.fit(articles_path, labels_path)
    engine.semantic_model.save("models/semantic_model.joblib")
    print("Saved Semantic model to models/semantic_model.joblib")

    print("\n" + "=" * 60)
    print("3. Validation Set Threshold Tuning & Calibration")
    print("=" * 60)

    # Grid search threshold on validation split
    best_th = 0.35
    best_f1 = -1.0
    val_y_true = [labels_map[a["article_id"]]["category_ids"].split("|") for a in val_arts]

    for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        val_preds = []
        for a in val_arts:
            p = engine.predict(a["title"], a["content"], model_name="tfidf", custom_threshold=th)
            val_preds.append(p["category_ids"])

        all_cats = taxonomy.all_categories()
        y_true_bin = [[1 if c in cats else 0 for c in all_cats] for cats in val_y_true]
        y_pred_bin = [[1 if c in cats else 0 for c in all_cats] for cats in val_preds]
        f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"Optimal validation threshold: {best_th} (Micro-F1: {best_f1:.4f})")
    engine.default_threshold = best_th

    print("\n" + "=" * 60)
    print("4. Evaluating Models on Validation and Test Splits")
    print("=" * 60)

    metrics_output = {}

    for split_name, split_arts in [("validation", val_arts), ("test", test_arts)]:
        y_true_cats = [labels_map[a["article_id"]]["category_ids"].split("|") for a in split_arts]
        y_true_tags = [labels_map[a["article_id"]]["tags"].split("|") for a in split_arts]

        # Evaluate TF-IDF model
        pred_cats = []
        pred_tags = []
        exact_matches = 0
        latencies = []

        for a in split_arts:
            t0 = time.perf_counter()
            res = engine.predict(a["title"], a["content"], model_name="tfidf", article_id=a["article_id"])
            latencies.append(time.perf_counter() - t0)

            cats = res["category_ids"]
            tags = res["tags"]
            pred_cats.append(cats)
            pred_tags.append(tags)

            true_c = set(labels_map[a["article_id"]]["category_ids"].split("|"))
            if set(cats) == true_c:
                exact_matches += 1

        all_cats = taxonomy.all_categories()
        Y_true_bin = np.array([[1 if c in cats else 0 for c in all_cats] for cats in y_true_cats])
        Y_pred_bin = np.array([[1 if c in cats else 0 for c in all_cats] for cats in pred_cats])

        micro_f1 = float(f1_score(Y_true_bin, Y_pred_bin, average="micro", zero_division=0))
        macro_f1 = float(f1_score(Y_true_bin, Y_pred_bin, average="macro", zero_division=0))
        exact_match = exact_matches / len(split_arts)
        p_at_3, r_at_3 = calculate_precision_recall_at_k(y_true_tags, pred_tags, k=3)

        metrics_output[split_name] = {
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "exact_match": round(exact_match, 4),
            "precision@3": round(p_at_3, 4),
            "recall@3": round(r_at_3, 4),
            "articles": len(split_arts),
            "avg_latency_ms": round(float(np.mean(latencies)) * 1000, 2)
        }

        print(f"\n--- TF-IDF Model ({split_name.upper()} SPLIT) ---")
        print(f"Micro-F1:     {micro_f1:.4f}")
        print(f"Macro-F1:     {macro_f1:.4f}")
        print(f"Exact Match:  {exact_match * 100:.1f}%")
        print(f"Precision@3:  {p_at_3:.4f}")
        print(f"Recall@3:     {r_at_3:.4f}")

    # Write official predictions.csv for all 96 articles
    prediction_rows = []
    feedback_rows = []

    for a in articles:
        art_id = a["article_id"]
        res = engine.predict(a["title"], a["content"], model_name="tfidf", article_id=art_id)
        prediction_rows.append({
            "article_id": art_id,
            "category_ids": "|".join(res["category_ids"]),
            "tags": "|".join(res["tags"])
        })

        # Format sample editorial feedback template row
        gold_tags = labels_map[art_id]["tags"].split("|")
        suggested = res["tags"]
        accepted = [t for t in suggested if t in gold_tags]
        rejected = [t for t in suggested if t not in gold_tags]
        added = [t for t in gold_tags if t not in suggested]

        feedback_rows.append({
            "post_id": art_id,
            "model_version": "tfidf_v1.0",
            "suggested_tags": "|".join(suggested),
            "suggestion_scores": json.dumps(res["scores"]),
            "review_status": res["status"],
            "accepted_tags": "|".join(accepted),
            "rejected_tags": "|".join(rejected),
            "added_approved_tags": "|".join(added),
            "final_tags": "|".join(gold_tags),
            "reviewer_id": "editor_1",
            "reviewed_at": "2026-11-22T19:00:00Z"
        })

    with open("predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["article_id", "category_ids", "tags"])
        w.writeheader()
        w.writerows(prediction_rows)

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    with open("editor_feedback.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["post_id", "model_version", "suggested_tags", "suggestion_scores", "review_status", "accepted_tags", "rejected_tags", "added_approved_tags", "final_tags", "reviewer_id", "reviewed_at"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(feedback_rows)

    print("\nSaved predictions.csv, metrics.json, and editor_feedback.csv successfully.")


if __name__ == "__main__":
    run_pipeline()
