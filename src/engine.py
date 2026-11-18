"""Inference and Decision Engine for Automated Content Tagging."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.taxonomy import TaxonomyManager
from src.preprocess import clean_article
from src.models.keyword_model import KeywordMatcherModel
from src.models.tfidf_model import TfidfTaggingModel
from src.models.semantic_model import SemanticTaggingModel


class TaggingEngine:
    """End-to-end inference and decision engine.
    
    Applies text cleaning, confidence thresholding, review routing, and tag ranking.
    """

    def __init__(
        self,
        taxonomy_manager: TaxonomyManager,
        default_model: str = "tfidf",
        min_word_count: int = 25,
        default_threshold: float = 0.35,
        max_suggestions: int = 5
    ):
        self.taxonomy = taxonomy_manager
        self.min_word_count = min_word_count
        self.default_threshold = default_threshold
        self.max_suggestions = max_suggestions
        self.per_tag_thresholds: Dict[str, float] = {}

        self.keyword_model = KeywordMatcherModel(self.taxonomy)
        self.tfidf_model = TfidfTaggingModel(self.taxonomy)
        self._semantic_model = None
        self.active_model_name = default_model

    @property
    def semantic_model(self):
        """Lazy load semantic model only when needed."""
        if self._semantic_model is None:
            self._semantic_model = SemanticTaggingModel(self.taxonomy)
            model_file = Path("models/semantic_model.joblib")
            if model_file.exists():
                self._semantic_model.load(model_file)
        return self._semantic_model

    def set_per_tag_thresholds(self, thresholds: Dict[str, float]):
        """Set tuned per-tag thresholds calibrated on the validation set."""
        self.per_tag_thresholds = thresholds

    def get_threshold_for_tag(self, tag: str) -> float:
        """Returns the specific threshold for a tag, falling back to default."""
        return self.per_tag_thresholds.get(tag, self.default_threshold)

    def predict(
        self,
        title: str,
        body: str,
        model_name: Optional[str] = None,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs the complete tagging and review decision pipeline on a draft article."""
        cleaned = clean_article(title, body)
        word_count = cleaned["word_count"]
        active_model = model_name or self.active_model_name

        # Route check 1: Thin / very short draft
        if word_count < self.min_word_count:
            return {
                "title": cleaned["cleaned_title"],
                "word_count": word_count,
                "model_used": active_model,
                "suggested_tags": [],
                "scores": {},
                "needs_review": True,
                "review_reason": f"Input contains too little text ({word_count} words < {self.min_word_count} minimum). Requires editorial drafting.",
                "status": "flagged_short_draft"
            }

        # Model prediction
        if active_model == "keyword":
            scores = self.keyword_model.predict_scores(cleaned["cleaned_title"], cleaned["cleaned_body"])
        elif active_model == "semantic":
            scores = self.semantic_model.predict_scores(cleaned["cleaned_title"], cleaned["cleaned_body"])
        else:
            scores = self.tfidf_model.predict_scores(cleaned["cleaned_title"], cleaned["cleaned_body"])

        # Filter by threshold and rank
        suggestions = []
        for tag, score in scores.items():
            thresh = custom_threshold if custom_threshold is not None else self.get_threshold_for_tag(tag)
            if score >= thresh:
                suggestions.append({
                    "tag": tag,
                    "score": score,
                    "threshold_applied": thresh
                })

        suggestions.sort(key=lambda x: x["score"], reverse=True)
        top_suggestions = suggestions[: self.max_suggestions]

        # Route check 2: No tag met threshold
        if not top_suggestions:
            # Provide top fallback recommendation for editor review with flag
            all_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            fallback = [{"tag": t, "score": s, "threshold_applied": custom_threshold or self.get_threshold_for_tag(t)} for t, s in all_ranked[:2]]
            return {
                "title": cleaned["cleaned_title"],
                "word_count": word_count,
                "model_used": active_model,
                "suggested_tags": fallback,
                "scores": scores,
                "needs_review": True,
                "review_reason": "No tag met the confidence threshold. Model scores are uncertain.",
                "status": "flagged_low_confidence"
            }

        return {
            "title": cleaned["cleaned_title"],
            "word_count": word_count,
            "model_used": active_model,
            "suggested_tags": top_suggestions,
            "scores": scores,
            "needs_review": False,
            "review_reason": "Confidence thresholds met.",
            "status": "confident"
        }
