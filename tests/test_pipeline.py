"""Comprehensive test suite for the Automated Content Tagging system."""

import unittest
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from src.taxonomy import TaxonomyManager
from src.preprocess import (
    strip_html_and_boilerplate,
    clean_article,
    compute_content_fingerprint,
    split_dataset_temporal
)
from src.models.keyword_model import KeywordMatcherModel
from src.models.tfidf_model import TfidfTaggingModel
from src.engine import TaggingEngine


class TestTaxonomyAndPreprocessing(unittest.TestCase):
    def setUp(self):
        self.taxonomy = TaxonomyManager("data/taxonomy.json")

    def test_taxonomy_loading(self):
        tags = self.taxonomy.all_approved_tags()
        self.assertGreaterEqual(len(tags), 10)
        self.assertIn("Machine Learning", tags)
        self.assertIn("Python", tags)

    def test_alias_resolution(self):
        self.assertEqual(self.taxonomy.resolve_alias("ml"), "Machine Learning")
        self.assertEqual(self.taxonomy.resolve_alias("python3"), "Python")
        self.assertEqual(self.taxonomy.resolve_alias("k8s"), "DevOps & Cloud")
        self.assertEqual(self.taxonomy.resolve_alias("appsec"), "Application Security")

    def test_boilerplate_stripping(self):
        raw_html = "<nav><a href='/'>Home</a></nav><h1>Title</h1><p>Clean text.</p><footer>Share on Twitter</footer>"
        cleaned = strip_html_and_boilerplate(raw_html)
        self.assertNotIn("Home", cleaned)
        self.assertNotIn("Twitter", cleaned)
        self.assertIn("Title", cleaned)
        self.assertIn("Clean text.", cleaned)

    def test_short_draft_flagging(self):
        cleaned = clean_article("Short Draft", "TODO: write later")
        self.assertTrue(cleaned["is_short_draft"])
        self.assertLess(cleaned["word_count"], 30)

    def test_duplicate_fingerprinting(self):
        t1 = "Building Fast APIs with Python and FastAPI framework"
        t2 = "Building fast APIs with python and Fastapi Framework updated"
        fp1 = compute_content_fingerprint(t1)
        fp2 = compute_content_fingerprint(t2)
        self.assertEqual(fp1, fp2)


class TestModelsAndEngine(unittest.TestCase):
    def setUp(self):
        self.taxonomy = TaxonomyManager("data/taxonomy.json")
        self.engine = TaggingEngine(self.taxonomy, default_model="tfidf", default_threshold=0.30)
        if Path("models/tfidf_model.joblib").exists():
            self.engine.tfidf_model.load("models/tfidf_model.joblib")

    def test_keyword_matcher(self):
        kw_model = KeywordMatcherModel(self.taxonomy)
        scores = kw_model.predict_scores("Python REST APIs", "Building web endpoints with FastAPI and Python.")
        self.assertGreater(scores.get("Python", 0), 0.3)
        self.assertGreater(scores.get("API Development", 0), 0.3)

    def test_engine_short_draft_routing(self):
        res = self.engine.predict("Note", "Quick snippet")
        self.assertTrue(res["needs_review"])
        self.assertEqual(res["status"], "flagged_short_draft")

    def test_engine_confident_prediction(self):
        body = """In modern API development, securing backend endpoints is paramount. Using FastAPI with Python,
        we can implement robust OAuth2 bearer token authentication with JSON Web Tokens (JWT).
        We demonstrate password hashing using passlib and Bcrypt, token expiration validation, and rate limiting."""
        res = self.engine.predict("Hardening FastAPI Services with Python and JWT", body, model_name="keyword")
        suggested_tags = [t["tag"] for t in res["suggested_tags"]]
        self.assertFalse(res["needs_review"])
        self.assertIn("Python", suggested_tags)
        self.assertIn("API Development", suggested_tags)


class TestFastAPIEndpoints(unittest.TestCase):
    def test_taxonomy_endpoint(self):
        from src.app import get_taxonomy
        data = get_taxonomy()
        self.assertIn("tags", data)
        self.assertIn("Python", data["tags"])

    def test_tag_endpoint(self):
        from src.app import tag_content, TagRequest
        req = TagRequest(
            title="PostgreSQL Query Optimization and Performance Indexing",
            body="Analyzing slow database queries in PostgreSQL begins with EXPLAIN ANALYZE and buffer inspection. We walk through index scans, vacuuming, and partitioning to speed up analytical workloads significantly.",
            model="keyword",
            threshold=0.20
        )
        data = tag_content(req)
        self.assertIn("suggested_tags", data)
        tags = [t["tag"] for t in data["suggested_tags"]]
        self.assertIn("Database Systems", tags)

    def test_feedback_submission(self):
        from src.app import submit_editorial_review, FeedbackSubmission
        sub = FeedbackSubmission(
            post_id="test-post-1",
            title="Test Title",
            suggested_tags=["Python"],
            approved_tags=["Python", "API Development"],
            rejected_tags=[]
        )
        res = submit_editorial_review(sub)
        self.assertEqual(res["status"], "success")

    def test_batch_run_and_exports(self):
        from src.app import run_batch_evaluation
        res = run_batch_evaluation()
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["processed_count"], 0)
        self.assertTrue(Path(res["exported_csv"]).exists())
        self.assertTrue(Path(res["exported_json"]).exists())


if __name__ == "__main__":
    unittest.main()
