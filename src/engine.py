"""Inference and Decision Engine for Automated Content Tagging."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.taxonomy import TaxonomyManager
from src.preprocess import clean_article
from src.models.keyword_model import KeywordMatcherModel
from src.models.tfidf_model import TfidfTaggingModel
from src.models.semantic_model import SemanticTaggingModel
from src.models.recommender import RelatedArticleRecommender


class TaggingEngine:
    """Multi-label inference engine with editorial review routing and article recommendation."""

    def __init__(
        self,
        taxonomy_manager: TaxonomyManager,
        default_model: str = "tfidf",
        min_word_count: int = 15,
        default_threshold: float = 0.35,
        max_suggestions: int = 5
    ):
        self.taxonomy = taxonomy_manager
        self.min_word_count = min_word_count
        self.default_threshold = default_threshold
        self.max_suggestions = max_suggestions

        self.keyword_model = KeywordMatcherModel(self.taxonomy)
        self.tfidf_model = TfidfTaggingModel(self.taxonomy)
        self._semantic_model = None
        self.recommender = RelatedArticleRecommender()
        self.active_model_name = default_model

    @property
    def semantic_model(self):
        if self._semantic_model is None:
            self._semantic_model = SemanticTaggingModel(self.taxonomy)
            model_file = Path("models/semantic_model.joblib")
            if model_file.exists():
                self._semantic_model.load(str(model_file))
        return self._semantic_model

    def predict(
        self,
        title: str,
        content: str,
        model_name: Optional[str] = None,
        custom_threshold: Optional[float] = None,
        article_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs multi-label tagging, thresholding, and recommendation on a draft article."""
        cleaned = clean_article(title, content)
        word_count = cleaned["word_count"]
        active_model = model_name or self.active_model_name
        threshold = custom_threshold if custom_threshold is not None else self.default_threshold

        # Guardrail 1: Empty or very short draft
        if word_count < self.min_word_count:
            return {
                "title": cleaned["cleaned_title"],
                "word_count": word_count,
                "model_used": active_model,
                "category_ids": [],
                "tags": [],
                "related_article_ids": [],
                "scores": {},
                "needs_review": True,
                "review_reason": f"Input contains too little text ({word_count} words < {self.min_word_count} minimum). Requires editorial review.",
                "status": "flagged_short_draft"
            }

        # Model Inference
        if active_model == "keyword":
            pred = self.keyword_model.predict(cleaned["cleaned_title"], cleaned["cleaned_content"])
            category_ids = pred["category_ids"]
            tags = pred["tags"][:self.max_suggestions]
            scores = pred["scores"]
        elif active_model == "semantic":
            pred = self.semantic_model.predict(cleaned["cleaned_title"], cleaned["cleaned_content"], threshold=threshold, max_tags=self.max_suggestions)
            category_ids = pred["category_ids"]
            tags = pred["tags"]
            scores = pred["tag_scores"]
        else:
            pred = self.tfidf_model.predict(cleaned["cleaned_title"], cleaned["cleaned_content"], threshold=threshold, max_tags=self.max_suggestions)
            category_ids = pred["category_ids"]
            tags = pred["tags"]
            scores = pred["tag_scores"]

        # Suggest 3 related articles (excluding self and formatting sibling)
        related = self.recommender.recommend(cleaned["cleaned_title"], cleaned["cleaned_content"], exclude_article_id=article_id, top_k=3)

        # Guardrail 2: Low confidence check
        if not tags or max(scores.values(), default=0.0) < threshold:
            return {
                "title": cleaned["cleaned_title"],
                "word_count": word_count,
                "model_used": active_model,
                "category_ids": category_ids,
                "tags": tags,
                "related_article_ids": related,
                "scores": scores,
                "needs_review": True,
                "review_reason": "Confidence scores did not meet editorial threshold. Review recommended.",
                "status": "flagged_low_confidence"
            }

        return {
            "title": cleaned["cleaned_title"],
            "word_count": word_count,
            "model_used": active_model,
            "category_ids": category_ids,
            "tags": tags,
            "related_article_ids": related,
            "scores": scores,
            "needs_review": False,
            "review_reason": "Confidence thresholds met.",
            "status": "confident"
        }
