"""Baseline 1: Keyword and Taxonomy Alias Matcher."""

import re
from typing import Dict, List, Tuple
from src.taxonomy import TaxonomyManager


class KeywordMatcherModel:
    """Multi-label tagging baseline using regex matching against taxonomy aliases."""

    def __init__(self, taxonomy_manager: TaxonomyManager, title_weight: float = 2.5):
        self.taxonomy = taxonomy_manager
        self.title_weight = title_weight
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile case-insensitive regex word boundary patterns for all aliases."""
        self.patterns: Dict[str, List[re.Pattern]] = {}
        for canonical_tag in self.taxonomy.all_approved_tags():
            info = self.taxonomy.get_tag_info(canonical_tag)
            terms = [canonical_tag] + (info.get("aliases", []) if info else [])

            compiled = []
            for term in terms:
                # Escape special regex chars like '+' or '.'
                escaped = re.escape(term)
                # Word boundaries
                pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
                compiled.append(pattern)
            self.patterns[canonical_tag] = compiled

    def predict_scores(self, title: str, body: str) -> Dict[str, float]:
        """Compute match scores for all taxonomy tags."""
        raw_scores: Dict[str, float] = {}
        max_possible_score = 0.0

        for tag, pattern_list in self.patterns.items():
            score = 0.0
            for pat in pattern_list:
                title_matches = len(pat.findall(title))
                body_matches = len(pat.findall(body))

                if title_matches > 0:
                    score += title_matches * self.title_weight
                if body_matches > 0:
                    score += min(body_matches, 6) * 1.0  # cap repeated mentions

            if score > 0:
                raw_scores[tag] = score

        if not raw_scores:
            return {tag: 0.0 for tag in self.taxonomy.all_approved_tags()}

        # Normalize scores to 0.0 - 1.0 range
        highest = max(raw_scores.values())
        normalized = {}
        for tag in self.taxonomy.all_approved_tags():
            score = raw_scores.get(tag, 0.0)
            normalized[tag] = min(1.0, round(score / (highest + 1.0), 4))

        return normalized

    def suggest_tags(
        self,
        title: str,
        body: str,
        threshold: float = 0.35,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return top-k ranked tags meeting the threshold."""
        scores = self.predict_scores(title, body)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(tag, score) for tag, score in ranked if score >= threshold][:top_k]
