"""Preprocessing and text cleaning module for HTML/Markdown blog articles."""

import hashlib
import html
import re
from typing import Dict, List, Optional, Set, Tuple


def strip_html_and_boilerplate(raw_text: str) -> str:
    """Removes HTML boilerplate, navigation, footers, and noise while preserving headings and meaningful content."""
    if not raw_text or not isinstance(raw_text, str):
        return ""

    text = raw_text

    # Strip script and style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Strip navigation, header, footer, aside, and tag widgets
    text = re.sub(r"<(nav|footer|aside|header)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Strip boilerplate phrases in markdown or text
    boilerplate_patterns = [
        r"(?i)subscribe and share this article\.?",
        r"(?i)share this post on (twitter|linkedin|facebook|x)",
        r"(?i)leave a comment below",
        r"(?i)copyright \d{4}.*?all rights reserved"
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, " ", text)

    # Strip HTML tags but preserve inner text
    text = re.sub(r"<[^>]+>", " ", text)

    # Strip markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Unescape HTML entities (&amp; -> &, etc.)
    text = html.unescape(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_article(title: str, content: str, title_weight: int = 2) -> Dict[str, any]:
    """Cleans title and content, returning normalized representations and length checks."""
    cleaned_title = strip_html_and_boilerplate(title)
    cleaned_content = strip_html_and_boilerplate(content)

    repeated_title = " ".join([cleaned_title] * title_weight) if cleaned_title else ""
    combined_text = f"{repeated_title} {cleaned_content}".strip()
    word_count = len(cleaned_content.split())

    return {
        "cleaned_title": cleaned_title,
        "cleaned_content": cleaned_content,
        "combined_text": combined_text,
        "word_count": word_count,
        "is_short_draft": word_count < 15
    }
