"""Preprocessing, cleaning, deduplication, and data splitting module."""

import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


def strip_html_and_boilerplate(raw_text: str) -> str:
    """Removes HTML tags, boilerplate widgets, navigation, footers, and scripts."""
    if not raw_text or not isinstance(raw_text, str):
        return ""

    text = raw_text

    # Strip script and style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Strip navigation, header, footer, aside, and tag widget elements
    text = re.sub(r"<(nav|header|footer|aside|div class=[\"'].*?(tag|footer|sidebar|nav).*?[\"'])[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Strip markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Strip markdown images ![alt](url) -> ""
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)

    # Strip markdown heading markers and blockquotes
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # Strip common boilerplate phrases found in blogs
    boilerplate_patterns = [
        r"(?i)share this post on (twitter|linkedin|facebook|x)",
        r"(?i)subscribe to our newsletter for more updates",
        r"(?i)leave a comment below",
        r"(?i)read more articles like this",
        r"(?i)copyright \d{4}.*?all rights reserved"
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_article(title: str, body: str, title_weight: int = 2) -> Dict[str, str]:
    """Cleans title and body, producing individual and combined weighted representations."""
    cleaned_title = strip_html_and_boilerplate(title)
    cleaned_body = strip_html_and_boilerplate(body)

    # Weighted representation repeats title to emphasize high-signal terms
    repeated_title = " ".join([cleaned_title] * title_weight) if cleaned_title else ""
    combined_text = f"{repeated_title} {cleaned_body}".strip()

    word_count = len(cleaned_body.split())

    return {
        "cleaned_title": cleaned_title,
        "cleaned_body": cleaned_body,
        "combined_text": combined_text,
        "word_count": word_count,
        "is_short_draft": word_count < 30
    }


def compute_content_fingerprint(text: str, num_tokens: int = 6) -> str:
    """Computes a normalized prefix fingerprint to detect republished near duplicates."""
    normalized = re.sub(r"[^\w\s]", "", text.lower())
    words = [w for w in normalized.split() if len(w) > 2]
    if not words:
        return ""
    sample = words[:num_tokens]
    return hashlib.md5(" ".join(sample).encode("utf-8")).hexdigest()


def split_dataset_temporal(
    df: pd.DataFrame,
    date_col: str = "published_date",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset chronologically into train, validation, and test sets.
    
    Ensures duplicate/near-duplicate articles stay in the same split.
    """
    df = df.copy()

    # Assign duplicate group IDs so near-duplicates are grouped together
    df["fingerprint"] = df["combined_text"].apply(compute_content_fingerprint)

    # Sort chronologically
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Initial split indices
    train_fps = set(df.iloc[:train_end]["fingerprint"])
    val_fps = set(df.iloc[train_end:val_end]["fingerprint"])

    # Prevent leak: if a fingerprint in val/test was in train, pull it into train
    train_mask = df.index < train_end
    val_mask = (df.index >= train_end) & (df.index < val_end)
    test_mask = df.index >= val_end

    # Adjust masks so duplicate groups never cross train/val/test boundary
    train_groups = set(df[train_mask]["fingerprint"].unique())
    val_groups = set(df[val_mask]["fingerprint"].unique()) - train_groups

    final_train = df[df["fingerprint"].isin(train_groups)].copy()
    remaining = df[~df["fingerprint"].isin(train_groups)]
    final_val = remaining[remaining["fingerprint"].isin(val_groups)].copy()
    final_test = remaining[~remaining["fingerprint"].isin(val_groups)].copy()

    return final_train, final_val, final_test
