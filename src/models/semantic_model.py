"""Semantic Embedding Model with chunking and multi-label classification."""

import csv
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.taxonomy import TaxonomyManager
from src.preprocess import clean_article


class SemanticTaggingModel:
    """Multi-label classifier on top of chunked sentence transformer embeddings."""

    def __init__(
        self,
        taxonomy_manager: TaxonomyManager,
        model_name: str = "models/all-MiniLM-L6-v2",
        chunk_size: int = 120,
        chunk_overlap: int = 30,
        title_weight: float = 0.40
    ):
        self.taxonomy = taxonomy_manager
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.title_weight = title_weight
        self.encoder = None
        self._load_encoder()

        self.category_mlb = MultiLabelBinarizer(classes=self.taxonomy.all_categories())
        self.category_classifier = OneVsRestClassifier(
            LogisticRegression(C=2.0, class_weight="balanced", max_iter=500, random_state=42)
        )
        self.is_fitted = False

    def _load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            p = Path(self.model_name)
            if p.exists():
                self.encoder = SentenceTransformer(str(p))
            else:
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2", token=False)
        except Exception:
            self.encoder = None

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words or len(words) <= self.chunk_size:
            return [" ".join(words)] if words else [""]
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i : i + self.chunk_size]))
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def encode_article(self, title: str, content: str) -> np.ndarray:
        if self.encoder is None:
            # Fallback projection
            np.random.seed(42)
            v = np.zeros(384, dtype=np.float32)
            for w in (title + " " + content).lower().split():
                v[hash(w) % 384] += 1.0
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        e_title = self.encoder.encode([title], show_progress_bar=False, normalize_embeddings=True)[0]
        chunks = self._chunk_text(content)
        chunk_embs = self.encoder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        e_body = np.mean(chunk_embs, axis=0)

        combined = (self.title_weight * e_title) + ((1.0 - self.title_weight) * e_body)
        norm = np.linalg.norm(combined)
        return combined / norm if norm > 0 else combined

    def fit(self, articles_csv: str = "data/articles.csv", labels_csv: str = "data/labels.csv"):
        with open(labels_csv, "r", encoding="utf-8") as f:
            labels_map = {r["article_id"]: r for r in csv.DictReader(f)}

        X_embeddings = []
        y_train_categories = []

        with open(articles_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != "train":
                    continue
                art_id = row["article_id"]
                label_row = labels_map.get(art_id)
                if not label_row:
                    continue

                cleaned = clean_article(row["title"], row["content"])
                emb = self.encode_article(cleaned["cleaned_title"], cleaned["cleaned_content"])
                X_embeddings.append(emb)
                cats = [c.strip() for c in label_row["category_ids"].split("|") if c.strip()]
                y_train_categories.append(cats)

        X = np.vstack(X_embeddings)
        Y_cats = self.category_mlb.fit_transform(y_train_categories)
        self.category_classifier.fit(X, Y_cats)
        self.is_fitted = True
        return self

    def predict(
        self,
        title: str,
        content: str,
        threshold: float = 0.35,
        max_tags: int = 5
    ) -> Dict[str, any]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        cleaned = clean_article(title, content)
        emb = self.encode_article(cleaned["cleaned_title"], cleaned["cleaned_content"]).reshape(1, -1)
        probs = self.category_classifier.predict_proba(emb)[0]

        predicted_categories = []
        cat_scores = {}
        for cat, prob in zip(self.category_mlb.classes_, probs):
            cat_scores[cat] = round(float(prob), 4)
            if prob >= threshold:
                predicted_categories.append(cat)

        if not predicted_categories:
            best_cat = self.category_mlb.classes_[int(np.argmax(probs))]
            predicted_categories = [best_cat]

        predicted_tags = []
        tag_scores = {}
        for cat in predicted_categories:
            prob = cat_scores.get(cat, 0.5)
            for tag in self.taxonomy.get_tags_for_category(cat):
                if tag == "apple":
                    res = self.taxonomy.resolve_alias_with_context("apple", f"{title} {content}")
                    if res == "fruit":
                        tag = "fruit"
                if tag not in predicted_tags:
                    predicted_tags.append(tag)
                    tag_scores[tag] = prob

        predicted_tags.sort(key=lambda t: tag_scores.get(t, 0.0), reverse=True)
        top_tags = predicted_tags[:max_tags]

        return {
            "category_ids": predicted_categories,
            "tags": top_tags,
            "category_scores": cat_scores,
            "tag_scores": {t: tag_scores.get(t, 0.0) for t in top_tags}
        }

    def save(self, path: str = "models/semantic_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "category_mlb": self.category_mlb,
            "category_classifier": self.category_classifier,
            "is_fitted": self.is_fitted
        }, path)

    def load(self, path: str = "models/semantic_model.joblib"):
        data = joblib.load(path)
        self.category_mlb = data.get("category_mlb", data.get("mlb"))
        self.category_classifier = data.get("category_classifier", data.get("classifier"))
        self.is_fitted = data.get("is_fitted", True)
        self._load_encoder()
        return self
