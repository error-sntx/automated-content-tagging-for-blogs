"""Taxonomy management, alias resolution, and contextual disambiguation."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class TaxonomyManager:
    """Manages the official approved tag taxonomy, aliases, categories, and contextual disambiguation."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.categories: Dict[str, dict] = {}
        self.category_to_tags: Dict[str, List[str]] = {}
        self.tag_to_category: Dict[str, str] = {}
        self.alias_to_tag: Dict[str, str] = {}
        self.approved_tags: List[str] = []
        self._load()

    def _load(self):
        taxonomy_csv = self.data_dir / "taxonomy.csv"
        aliases_csv = self.data_dir / "aliases.csv"

        if taxonomy_csv.exists():
            with open(taxonomy_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cat_id = row["category_id"].strip()
                    tags = [t.strip() for t in row["canonical_tags"].split("|") if t.strip()]
                    self.categories[cat_id] = {
                        "category_id": cat_id,
                        "parent_category": row.get("parent_category", ""),
                        "category": row.get("category", ""),
                        "canonical_tags": tags
                    }
                    self.category_to_tags[cat_id] = tags
                    for t in tags:
                        if t not in self.approved_tags:
                            self.approved_tags.append(t)
                        self.tag_to_category[t] = cat_id

        if aliases_csv.exists():
            with open(aliases_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    alias = row["alias"].strip().lower()
                    canonical = row["canonical_tag"].strip()
                    self.alias_to_tag[alias] = canonical

    def is_valid_tag(self, tag: str) -> bool:
        """Check if a tag is an approved canonical tag."""
        return tag in self.tag_to_category

    def all_approved_tags(self) -> List[str]:
        """Return list of all approved canonical tags."""
        return list(self.approved_tags)

    def all_categories(self) -> List[str]:
        """Return list of all category IDs."""
        return list(self.categories.keys())

    def get_tags_for_category(self, category_id: str) -> List[str]:
        """Get canonical tags belonging to a category."""
        return self.category_to_tags.get(category_id, [])

    def resolve_alias_with_context(self, term: str, context_text: str) -> Optional[str]:
        """Resolves alias or ambiguous term to a canonical tag using contextual cues.
        
        Specifically resolves 'apple':
        - If fruit/recipe/lunch/meal/nutrition/salad/fresh/ingredients -> 'fruit'
        - If device/battery/screen/hardware/company/charging/tech -> 'apple' (device)
        """
        cleaned = term.strip().lower()
        context_lower = context_text.lower()

        if cleaned == "apple":
            fruit_clues = {"fruit", "meal", "nutrition", "recipe", "lunch", "salad", "vegetables", "ingredient", "flavour", "flavor", "eat", "food"}
            device_clues = {"device", "battery", "screen", "charging", "hardware", "phone", "mac", "ios", "company", "watch", "consumer"}

            words = set(context_lower.split())
            fruit_score = len(words & fruit_clues)
            device_score = len(words & device_clues)

            if fruit_score > device_score:
                return "fruit"
            elif device_score > 0:
                return "apple"
            return "fruit" if "lunch" in context_lower or "flavour" in context_lower else "apple"

        # Check standard alias dictionary
        if cleaned in self.alias_to_tag:
            return self.alias_to_tag[cleaned]

        # Check canonical tags
        for t in self.approved_tags:
            if cleaned == t.lower():
                return t

        return None
