"""Comprehensive test suite for the Automated Content Tagging system."""

import unittest
from pathlib import Path
from src.taxonomy import TaxonomyManager
from src.preprocess import strip_html_and_boilerplate, clean_article
from src.models.keyword_model import KeywordMatcherModel
from src.models.recommender import RelatedArticleRecommender
from src.engine import TaggingEngine
from src.app import get_taxonomy, tag_content, TagRequest


class TestOfficialTaxonomyAndDisambiguation(unittest.TestCase):
    def setUp(self):
        self.taxonomy = TaxonomyManager("data")

    def test_taxonomy_categories_and_tags(self):
        cats = self.taxonomy.all_categories()
        self.assertEqual(len(cats), 6)
        self.assertIn("cloud", cats)
        self.assertIn("devices", cats)
        self.assertIn("nutrition", cats)

    def test_alias_mapping(self):
        self.assertEqual(self.taxonomy.alias_to_tag.get("reactjs"), "react")
        self.assertEqual(self.taxonomy.alias_to_tag.get("ml"), "model")
        self.assertEqual(self.taxonomy.alias_to_tag.get("cloud hosting"), "cloud")

    def test_apple_context_disambiguation(self):
        # Fruit context
        fruit_text = "An apple supplies texture and flavour in a fresh salad recipe."
        tag_fruit = self.taxonomy.resolve_alias_with_context("apple", fruit_text)
        self.assertEqual(tag_fruit, "fruit")

        # Device context
        device_text = "Testing the Apple Inc. smartphone reveals improved battery life and screen performance."
        tag_device = self.taxonomy.resolve_alias_with_context("apple", device_text)
        self.assertEqual(tag_device, "apple")


class TestPreprocessingAndRecommender(unittest.TestCase):
    def test_boilerplate_stripping(self):
        raw_html = "<h1>Cloud Case Study</h1><p>Deploying servers.</p><footer>Subscribe and share this article.</footer>"
        cleaned = strip_html_and_boilerplate(raw_html)
        self.assertNotIn("Subscribe and share", cleaned)
        self.assertIn("Cloud Case Study", cleaned)
        self.assertIn("Deploying servers.", cleaned)

    def test_related_articles_recommender(self):
        recommender = RelatedArticleRecommender("data/articles.csv")
        recs = recommender.recommend("Cloud server deployment", "Hosting inventory on virtual servers", exclude_article_id="A001", top_k=3)
        self.assertEqual(len(recs), 3)
        # Excluded self
        self.assertNotIn("A001", recs)
        # Excluded formatting sibling A002 (shares group_id G001)
        self.assertNotIn("A002", recs)


class TestEngineAndAPI(unittest.TestCase):
    def setUp(self):
        self.taxonomy = TaxonomyManager("data")
        self.engine = TaggingEngine(self.taxonomy, default_model="tfidf", default_threshold=0.30)
        if Path("models/tfidf_model.joblib").exists():
            self.engine.tfidf_model.load("models/tfidf_model.joblib")

    def test_short_draft_routing(self):
        res = self.engine.predict("Note", "Quick TODO")
        self.assertTrue(res["needs_review"])
        self.assertEqual(res["status"], "flagged_short_draft")

    def test_api_tag_request(self):
        req = TagRequest(
            title="A ReactJS form review",
            content="The browser interface needs accessible keyboard navigation, responsive layout, and client-side form validation. We implement reusable React components with smooth browser rendering.",
            model="keyword"
        )
        data = tag_content(req)
        self.assertIn("web", data["category_ids"])
        self.assertIn("react", data["tags"])
        self.assertGreaterEqual(len(data["related_article_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
