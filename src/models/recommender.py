"""Related articles recommender based on text similarity."""

import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_article


class RelatedArticleRecommender:
    """Suggests top-3 related articles by content similarity, excluding self and formatting siblings (same group_id)."""

    def __init__(self, data_path: str = "data/articles.csv"):
        self.data_path = Path(data_path)
        self.articles: List[dict] = []
        self.article_ids: List[str] = []
        self.group_ids: List[str] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=1000)
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        if not self.data_path.exists():
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            texts = []
            for row in reader:
                self.articles.append(row)
                self.article_ids.append(row["article_id"])
                self.group_ids.append(row.get("group_id", ""))
                cleaned = clean_article(row["title"], row["content"])
                texts.append(cleaned["combined_text"])

        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def recommend(
        self,
        query_title: str,
        query_content: str,
        exclude_article_id: Optional[str] = None,
        top_k: int = 3
    ) -> List[str]:
        """Returns top_k article IDs most similar to the query, excluding query article and siblings."""
        if self.tfidf_matrix is None or len(self.article_ids) == 0:
            return []

        # Find group_id to exclude if query is an existing article
        exclude_group = None
        if exclude_article_id and exclude_article_id in self.article_ids:
            idx = self.article_ids.index(exclude_article_id)
            exclude_group = self.group_ids[idx]

        cleaned = clean_article(query_title, query_content)
        query_vec = self.vectorizer.transform([cleaned["combined_text"]])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        ranked_indices = np.argsort(sims)[::-1]
        recommended = []

        for idx in ranked_indices:
            art_id = self.article_ids[idx]
            grp_id = self.group_ids[idx]

            # Exclude self and formatting sibling (same group_id)
            if exclude_article_id and art_id == exclude_article_id:
                continue
            if exclude_group and grp_id == exclude_group:
                continue

            recommended.append(art_id)
            if len(recommended) >= top_k:
                break

        return recommended
