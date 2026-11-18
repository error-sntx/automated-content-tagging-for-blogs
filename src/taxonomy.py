"""Taxonomy management and alias resolution module."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class TaxonomyManager:
    """Manages the approved tag taxonomy, aliases, and validation."""

    def __init__(self, taxonomy_path: str | Path):
        self.taxonomy_path = Path(taxonomy_path)
        self.tags_metadata: Dict[str, dict] = {}
        self.approved_tags: List[str] = []
        self.alias_to_tag: Dict[str, str] = {}
        self._load()

    def _load(self):
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data.get("tags", []):
            tag_name = item["tag"]
            self.approved_tags.append(tag_name)
            self.tags_metadata[tag_name] = item

            # Map the exact tag name (lowercased) to the canonical tag
            self.alias_to_tag[tag_name.lower().strip()] = tag_name

            # Map all declared aliases
            for alias in item.get("aliases", []):
                cleaned_alias = alias.lower().strip()
                self.alias_to_tag[cleaned_alias] = tag_name

    def is_valid_tag(self, tag: str) -> bool:
        """Check if a tag name is an approved editorial tag."""
        return tag in self.tags_metadata

    def resolve_alias(self, term: str) -> Optional[str]:
        """Resolve a raw term or alias to the canonical approved tag."""
        cleaned = term.lower().strip()
        return self.alias_to_tag.get(cleaned)

    def get_tag_info(self, tag: str) -> Optional[dict]:
        """Get definition and examples for a tag."""
        return self.tags_metadata.get(tag)

    def all_approved_tags(self) -> List[str]:
        """Return list of canonical approved tags."""
        return list(self.approved_tags)

    def normalize_tag_list(self, raw_tags: List[str]) -> List[str]:
        """Given a list of candidate or legacy tags, resolve aliases and deduplicate."""
        resolved: List[str] = []
        seen: Set[str] = set()
        for t in raw_tags:
            canonical = self.resolve_alias(t)
            if canonical and canonical not in seen:
                seen.add(canonical)
                resolved.append(canonical)
        return resolved
