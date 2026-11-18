"""Baseline 2: TF-IDF + Multi-label Logistic Regression Classifier."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

from src.taxonomy import TaxonomyManager


class TfidfTaggingModel:
    """Multi-label classifier using TF-IDF features and One-vs-Rest Logistic Regression."""

    def __init__(self, taxonomy_manager: TaxonomyManager, title_weight: int = 3):
        self.taxonomy = taxonomy_manager
        self.title_weight = title_weight
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True,
            stop_words="english"
        )
        self.mlb = MultiLabelBinarizer(classes=self.taxonomy.all_approved_tags())
        self.model = OneVsRestClassifier(
            LogisticRegression(C=2.0, class_weight="balanced", max_iter=500, random_state=42)
        )
        self.is_fitted = False

    def _prepare_texts(self, titles: List[str], bodies: List[str]) -> List[str]:
        """Combine titles and bodies with title weighting."""
        combined = []
        for t, b in zip(titles, bodies):
            weighted_title = " ".join([t.strip()] * self.title_weight) if t else ""
            combined.append(f"{weighted_title} {b.strip()}".strip())
        return combined

    def fit(self, titles: List[str], bodies: List[str], tag_lists: List[List[str]]):
        """Fit the TF-IDF vectorizer and multi-label classifier."""
        X_text = self._prepare_texts(titles, bodies)
        X_vec = self.vectorizer.fit_transform(X_text)

        # Normalize ground truth tags according to taxonomy
        normalized_tags = [self.taxonomy.normalize_tag_list(tags) for tags in tag_lists]
        Y = self.mlb.fit_transform(normalized_tags)

        self.model.fit(X_vec, Y)
        self.is_fitted = True
        return self

    def predict_scores(self, title: str, body: str) -> Dict[str, float]:
        """Predict confidence probability scores for each tag in the taxonomy."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        X_text = self._prepare_texts([title], [body])
        X_vec = self.vectorizer.transform(X_text)
        probs = self.model.predict_proba(X_vec)[0]

        tag_scores = {}
        for tag, prob in zip(self.mlb.classes_, probs):
            tag_scores[tag] = round(float(prob), 4)
        return tag_scores

    def suggest_tags(
        self,
        title: str,
        body: str,
        threshold: float = 0.40,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return top-k suggestions meeting the confidence threshold."""
        scores = self.predict_scores(title, body)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(t, s) for t, s in ranked if s >= threshold][:top_k]

    def save(self, filepath: str | Path):
        """Serialize model artifacts."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "mlb": self.mlb,
            "model": self.model,
            "title_weight": self.title_weight,
            "is_fitted": self.is_fitted
        }, filepath)

    def load(self, filepath: str | Path):
        """Load serialized model artifacts."""
        data = joblib.load(filepath)
        self.vectorizer = data["vectorizer"]
        self.mlb = data["mlb"]
        self.model = data["model"]
        self.title_weight = data["title_weight"]
        self.is_fitted = data["is_fitted"]
        return self
