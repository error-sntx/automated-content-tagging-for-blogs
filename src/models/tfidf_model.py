"""Multi-label TF-IDF + One-vs-Rest Logistic Regression Classifier."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.taxonomy import TaxonomyManager
from src.preprocess import clean_article


class TfidfTaggingModel:
    """TF-IDF feature extractor + One-vs-Rest Logistic Regression for multi-label category & tag prediction."""

    def __init__(self, taxonomy_manager: TaxonomyManager, title_weight: int = 2):
        self.taxonomy = taxonomy_manager
        self.title_weight = title_weight
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=1200
        )
        self.category_mlb = MultiLabelBinarizer(classes=self.taxonomy.all_categories())
        self.tag_mlb = MultiLabelBinarizer(classes=self.taxonomy.all_approved_tags())

        self.category_classifier = OneVsRestClassifier(
            LogisticRegression(C=2.5, class_weight="balanced", max_iter=500, random_state=42)
        )
        self.is_fitted = False

    def _prepare_text(self, title: str, content: str) -> str:
        cleaned = clean_article(title, content, title_weight=self.title_weight)
        return cleaned["combined_text"]

    def fit(self, articles_csv: str = "data/articles.csv", labels_csv: str = "data/labels.csv"):
        """Fits vocabulary and multi-label classifier on train split only."""
        with open(labels_csv, "r", encoding="utf-8") as f:
            labels_map = {r["article_id"]: r for r in csv.DictReader(f)}

        X_train_texts = []
        y_train_categories = []

        with open(articles_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != "train":
                    continue
                art_id = row["article_id"]
                label_row = labels_map.get(art_id)
                if not label_row:
                    continue

                X_train_texts.append(self._prepare_text(row["title"], row["content"]))
                cats = [c.strip() for c in label_row["category_ids"].split("|") if c.strip()]
                y_train_categories.append(cats)

        X_vec = self.vectorizer.fit_transform(X_train_texts)
        Y_cats = self.category_mlb.fit_transform(y_train_categories)
        self.category_classifier.fit(X_vec, Y_cats)
        self.is_fitted = True
        return self

    def predict(
        self,
        title: str,
        content: str,
        threshold: float = 0.35,
        max_tags: int = 5
    ) -> Dict[str, any]:
        """Predicts multi-label categories and canonical tags."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        text = self._prepare_text(title, content)
        vec = self.vectorizer.transform([text])
        probs = self.category_classifier.predict_proba(vec)[0]

        predicted_categories = []
        cat_scores = {}
        for cat, prob in zip(self.category_mlb.classes_, probs):
            cat_scores[cat] = round(float(prob), 4)
            if prob >= threshold:
                predicted_categories.append(cat)

        if not predicted_categories:
            # Fallback to argmax category if none reached threshold
            best_cat = self.category_mlb.classes_[int(np.argmax(probs))]
            predicted_categories = [best_cat]

        # Generate canonical tags for predicted categories
        predicted_tags = []
        tag_scores = {}
        for cat in predicted_categories:
            prob = cat_scores.get(cat, 0.5)
            for tag in self.taxonomy.get_tags_for_category(cat):
                # Disambiguate apple
                if tag == "apple":
                    res = self.taxonomy.resolve_alias_with_context("apple", text)
                    if res == "fruit":
                        tag = "fruit"
                if tag not in predicted_tags:
                    predicted_tags.append(tag)
                    tag_scores[tag] = prob

        # Sort tags by score
        predicted_tags.sort(key=lambda t: tag_scores.get(t, 0.0), reverse=True)
        top_tags = predicted_tags[:max_tags]

        return {
            "category_ids": predicted_categories,
            "tags": top_tags,
            "category_scores": cat_scores,
            "tag_scores": {t: tag_scores.get(t, 0.0) for t in top_tags}
        }

    def save(self, path: str = "models/tfidf_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "category_mlb": self.category_mlb,
            "category_classifier": self.category_classifier,
            "is_fitted": self.is_fitted
        }, path)

    def load(self, path: str = "models/tfidf_model.joblib"):
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.category_mlb = data["category_mlb"]
        self.category_classifier = data["category_classifier"]
        self.is_fitted = data["is_fitted"]
        return self
