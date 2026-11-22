"""Baseline 2: Contextual Keyword & Alias Matcher."""

import re
from typing import Dict, List, Set, Tuple
from src.taxonomy import TaxonomyManager


class KeywordMatcherModel:
    """Multi-label baseline using regex matching, alias resolution, and contextual disambiguation."""

    def __init__(self, taxonomy_manager: TaxonomyManager, title_weight: float = 2.0):
        self.taxonomy = taxonomy_manager
        self.title_weight = title_weight

    def predict(self, title: str, content: str) -> Dict[str, any]:
        """Predicts categories and tags from title and content."""
        full_text = f"{title} {content}".lower()
        title_lower = title.lower()

        matched_categories: Set[str] = set()
        matched_tags: Set[str] = set()
        tag_scores: Dict[str, float] = {}

        # 1. Check all canonical tags and aliases
        for cat_id in self.taxonomy.all_categories():
            tags = self.taxonomy.get_tags_for_category(cat_id)
            cat_score = 0.0

            for tag in tags:
                pattern = rf"(?<!\w){re.escape(tag)}(?!\w)"
                in_title = len(re.findall(pattern, title_lower))
                in_body = len(re.findall(pattern, full_text))

                if in_title > 0 or in_body > 0:
                    # Disambiguation check for "apple"
                    if tag == "apple":
                        resolved = self.taxonomy.resolve_alias_with_context("apple", full_text)
                        if resolved == "fruit":
                            # It is the fruit! Map to nutrition
                            matched_categories.add("nutrition")
                            matched_tags.add("fruit")
                            tag_scores["fruit"] = 0.90
                            continue
                        elif resolved == "apple":
                            matched_categories.add("devices")
                            matched_tags.add("apple")
                            tag_scores["apple"] = 0.90
                            continue

                    score = min(1.0, 0.40 + (in_title * 0.30) + (in_body * 0.15))
                    tag_scores[tag] = round(score, 3)
                    matched_tags.add(tag)
                    matched_categories.add(cat_id)

        # 2. Check aliases (e.g. ReactJS -> react, Apple Inc. -> apple, ML -> model)
        for alias, canonical_tag in self.taxonomy.alias_to_tag.items():
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            if re.search(pattern, full_text):
                if canonical_tag == "apple":
                    resolved = self.taxonomy.resolve_alias_with_context("apple", full_text)
                    canonical_tag = resolved or "apple"
                cat_id = self.taxonomy.tag_to_category.get(canonical_tag)
                if cat_id:
                    matched_categories.add(cat_id)
                matched_tags.add(canonical_tag)
                tag_scores[canonical_tag] = max(tag_scores.get(canonical_tag, 0.0), 0.85)

        # Build ordered outputs
        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "category_ids": sorted(list(matched_categories)),
            "tags": [t for t, _ in sorted_tags],
            "scores": tag_scores
        }
