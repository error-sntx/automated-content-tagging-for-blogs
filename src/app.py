"""FastAPI Web Application and Interactive Editorial Review Dashboard."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from src.taxonomy import TaxonomyManager
from src.preprocess import clean_article
from src.engine import TaggingEngine

app = FastAPI(
    title="Automated Content Tagging for Blogs",
    description="Multi-label ML tagging service with human-in-the-loop review workflow.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

taxonomy = TaxonomyManager("data")
engine = TaggingEngine(taxonomy, default_model="tfidf", default_threshold=0.35)

if Path("models/tfidf_model.joblib").exists():
    engine.tfidf_model.load("models/tfidf_model.joblib")


class TagRequest(BaseModel):
    title: str
    content: Optional[str] = None
    body: Optional[str] = None
    model: Optional[str] = "tfidf"
    threshold: Optional[float] = 0.35
    article_id: Optional[str] = None


class FeedbackSubmission(BaseModel):
    post_id: Optional[str] = "custom-draft"
    title: str
    suggested_tags: List[str]
    approved_tags: List[str]
    rejected_tags: List[str]
    editor_notes: Optional[str] = ""


def extract_supporting_snippets(text: str, tag: str, window_words: int = 12) -> List[str]:
    """Extracts context quotes where the tag or its aliases appear."""
    words = text.split()
    lower_words = [w.lower().strip(".,;:!?()[]\"'") for w in words]
    snippets = []
    
    target_terms = [tag.lower()]
    for alias, can_tag in taxonomy.alias_to_tag.items():
        if can_tag.lower() == tag.lower():
            target_terms.append(alias.lower())

    for term in target_terms:
        term_tokens = term.split()
        for i in range(len(lower_words) - len(term_tokens) + 1):
            if lower_words[i : i + len(term_tokens)] == term_tokens:
                start = max(0, i - window_words // 2)
                end = min(len(words), i + len(term_tokens) + window_words // 2)
                snippet = " ".join(words[start:end])
                if snippet not in snippets:
                    snippets.append(f"... {snippet} ...")
                if len(snippets) >= 2:
                    return snippets
    return snippets


@app.get("/api/taxonomy")
def get_taxonomy():
    return {
        "categories": taxonomy.categories,
        "approved_tags": taxonomy.all_approved_tags(),
        "aliases": taxonomy.alias_to_tag
    }


@app.post("/api/tag")
def tag_content(req: TagRequest):
    article_content = req.content if req.content is not None else (req.body or "")
    result = engine.predict(
        title=req.title,
        content=article_content,
        model_name=req.model,
        custom_threshold=req.threshold,
        article_id=req.article_id
    )

    full_text = f"{req.title} {article_content}"
    tag_snippets = {}
    for tag in result["tags"]:
        tag_snippets[tag] = extract_supporting_snippets(full_text, tag)
    result["supporting_snippets"] = tag_snippets

    return result


@app.post("/api/review/submit")
def submit_editorial_review(submission: FeedbackSubmission):
    record = {
        "post_id": submission.post_id,
        "model_version": "tfidf_v1.0",
        "suggested_tags": "|".join(submission.suggested_tags),
        "suggestion_scores": json.dumps({t: 1.0 for t in submission.suggested_tags}),
        "review_status": "editor_reviewed",
        "accepted_tags": "|".join(submission.approved_tags),
        "rejected_tags": "|".join(submission.rejected_tags),
        "added_approved_tags": "|".join([t for t in submission.approved_tags if t not in submission.suggested_tags]),
        "final_tags": "|".join(submission.approved_tags),
        "reviewer_id": "editor_web",
        "reviewed_at": pd.Timestamp.now().isoformat()
    }
    
    csv_file = Path("editor_feedback.csv")
    exists = csv_file.exists()
    fieldnames = ["post_id", "model_version", "suggested_tags", "suggestion_scores", "review_status", "accepted_tags", "rejected_tags", "added_approved_tags", "final_tags", "reviewer_id", "reviewed_at"]
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(record)

    return {"status": "success", "message": "Feedback saved to editor_feedback.csv"}


@app.get("/api/batch/run")
def run_batch_evaluation():
    if not Path("data/articles.csv").exists():
        raise HTTPException(status_code=404, detail="articles.csv not found.")

    with open("data/articles.csv", "r", encoding="utf-8") as f:
        articles = list(csv.DictReader(f))
    with open("data/labels.csv", "r", encoding="utf-8") as f:
        labels_map = {r["article_id"]: r for r in csv.DictReader(f)}

    rows = []
    for a in articles:
        res = engine.predict(a["title"], a["content"], model_name="tfidf", article_id=a["article_id"])
        gold = labels_map.get(a["article_id"], {})
        rows.append({
            "article_id": a["article_id"],
            "title": a["title"],
            "format": a.get("format", ""),
            "split": a.get("split", ""),
            "predicted_categories": res["category_ids"],
            "suggested_tags": res["tags"],
            "related_article_ids": res["related_article_ids"],
            "gold_categories": gold.get("category_ids", "").split("|") if gold else [],
            "gold_tags": gold.get("tags", "").split("|") if gold else [],
            "review_status": res["status"],
            "needs_review": res["needs_review"]
        })
    return {"status": "success", "processed_count": len(rows), "items": rows}


@app.get("/api/metrics")
def get_metrics():
    if Path("metrics.json").exists():
        with open("metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Run src.train_and_evaluate first."}


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Automated Content Tagging for Blogs</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
  <style>
    body { background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #1e293b; }
    .header-banner { background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); color: white; padding: 2rem 0; margin-bottom: 2rem; }
    .card { border: none; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 1.5rem; }
    .badge-tag { font-size: 0.95rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-weight: 500; margin: 4px; }
    .snippet-box { font-size: 0.85rem; background: #f1f5f9; padding: 8px; border-radius: 6px; border-left: 3px solid #3b82f6; font-style: italic; margin-top: 6px; }
  </style>
</head>
<body>
<div class="header-banner">
  <div class="container">
    <h2><i class="bi bi-tags-fill me-2 text-primary"></i>Automated Content Tagging for Blogs</h2>
    <p class="mb-0 text-white-50">Multi-label classification, contextual alias disambiguation, and editorial review workflow.</p>
  </div>
</div>

<div class="container pb-5">
  <ul class="nav nav-tabs mb-4">
    <li class="nav-item"><a class="nav-link active" data-bs-toggle="tab" href="#tagger"><i class="bi bi-pencil-square me-1"></i>Interactive Tagger</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#batch" onclick="loadBatch()"><i class="bi bi-collection me-1"></i>Batch Evaluation</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#metrics" onclick="loadMetrics()"><i class="bi bi-graph-up me-1"></i>Metrics</a></li>
  </ul>

  <div class="tab-content">
    <div class="tab-pane fade show active" id="tagger">
      <div class="row">
        <div class="col-lg-7">
          <div class="card p-4">
            <h5>Article Draft Input</h5>
            <div class="mb-3">
              <label class="form-label fw-bold">Title</label>
              <input type="text" id="draftTitle" class="form-control" value="A ReactJS form review">
            </div>
            <div class="mb-3">
              <label class="form-label fw-bold">Content (Plain text, Markdown, or HTML)</label>
              <textarea id="draftContent" class="form-control" rows="6">The browser interface needs accessible keyboard navigation and validation. React components render smoothly.</textarea>
            </div>
            <div class="d-flex gap-2">
              <button class="btn btn-primary" onclick="runTag()"><i class="bi bi-magic me-1"></i>Predict Tags</button>
              <button class="btn btn-outline-secondary" onclick="loadSample('apple_fruit')">Test Apple (Fruit)</button>
              <button class="btn btn-outline-secondary" onclick="loadSample('apple_device')">Test Apple (Device)</button>
            </div>
          </div>
        </div>
        <div class="col-lg-5">
          <div class="card p-4">
            <h5>Predictions & Related Articles</h5>
            <div id="outputContainer" class="text-muted text-center py-4">Click Predict Tags to run.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="tab-pane fade" id="batch">
      <div class="card p-4">
        <h5>Official 96 Articles Evaluation Table</h5>
        <div class="table-responsive">
          <table class="table table-hover table-sm" id="batchTable">
            <thead><tr><th>ID</th><th>Format</th><th>Split</th><th>Predicted Categories</th><th>Tags</th><th>Related IDs</th></tr></thead>
            <tbody id="batchBody"><tr><td colspan="6">Loading...</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="tab-pane fade" id="metrics">
      <div class="card p-4">
        <h5>Evaluation Metrics</h5>
        <div id="metricsBody">Loading...</div>
      </div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
  async function runTag() {
    const title = document.getElementById('draftTitle').value;
    const content = document.getElementById('draftContent').value;
    const res = await fetch('/api/tag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, content })
    });
    const d = await res.json();
    let html = '<div class="alert alert-info py-2"><strong>Categories:</strong> ' + d.category_ids.join(', ') + '</div>';
    html += '<h6 class="fw-bold">Canonical Tags:</h6><div>';
    d.tags.forEach(t => { html += `<span class="badge bg-primary badge-tag">${t}</span>`; });
    html += '</div>';
    if (d.related_article_ids.length) {
      html += '<h6 class="fw-bold mt-3">Related Articles:</h6><p><code>' + d.related_article_ids.join(', ') + '</code></p>';
    }
    document.getElementById('outputContainer').innerHTML = html;
  }

  function loadSample(type) {
    if (type === 'apple_fruit') {
      document.getElementById('draftTitle').value = 'Fresh apple slices for a balanced lunch recipe';
      document.getElementById('draftContent').value = 'An apple supplies texture and flavour in a fresh fruit salad. We prepare meal ingredients with vegetables.';
    } else {
      document.getElementById('draftTitle').value = 'Apple device battery charging test';
      document.getElementById('draftContent').value = 'Testing the new Apple Inc. hardware device reveals extended battery life under heavy screen usage.';
    }
    runTag();
  }

  async function loadBatch() {
    const res = await fetch('/api/batch/run');
    const data = await res.json();
    let rows = '';
    data.items.slice(0, 30).forEach(i => {
      rows += `<tr><td><strong>${i.article_id}</strong></td><td>${i.format}</td><td><span class="badge bg-light text-dark border">${i.split}</span></td><td>${i.predicted_categories.join('|')}</td><td>${i.suggested_tags.join('|')}</td><td><code>${i.related_article_ids.join(', ')}</code></td></tr>`;
    });
    document.getElementById('batchBody').innerHTML = rows;
  }

  async function loadMetrics() {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    document.getElementById('metricsBody').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=DASHBOARD_HTML)
