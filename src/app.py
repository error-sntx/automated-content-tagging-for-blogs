"""FastAPI Web Application and Interactive Editorial Review Dashboard."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from src.taxonomy import TaxonomyManager
from src.preprocess import clean_article
from src.engine import TaggingEngine

# Initialize FastAPI app
app = FastAPI(
    title="Automated Content Tagging for Blogs",
    description="Multi-label ML tagging service with human-in-the-loop editorial review workflow.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load taxonomy and engine
TAXONOMY_PATH = "data/taxonomy.json"
FEEDBACK_FILE = "data/editorial_feedback.json"
BATCH_EXPORT_CSV = "data/batch_export.csv"
BATCH_EXPORT_JSON = "data/batch_export.json"

taxonomy = TaxonomyManager(TAXONOMY_PATH)
engine = TaggingEngine(taxonomy, default_model="tfidf", default_threshold=0.35)

# Load trained models if available
if Path("models/tfidf_model.joblib").exists():
    engine.tfidf_model.load("models/tfidf_model.joblib")


class TagRequest(BaseModel):
    title: str
    body: str
    model: Optional[str] = "tfidf"
    threshold: Optional[float] = 0.35


class FeedbackSubmission(BaseModel):
    post_id: Optional[str] = "custom-draft"
    title: str
    suggested_tags: List[str]
    approved_tags: List[str]
    rejected_tags: List[str]
    editor_notes: Optional[str] = ""


def extract_supporting_snippets(text: str, tag: str, window_words: int = 15) -> List[str]:
    """Extracts supporting text snippets for a suggested tag based on taxonomy aliases."""
    info = taxonomy.get_tag_info(tag)
    aliases = [tag] + (info.get("aliases", []) if info else [])
    words = text.split()
    snippets = []

    lower_words = [w.lower().strip(".,;:!?()[]\"'") for w in words]
    for alias in aliases:
        alias_tokens = alias.lower().split()
        for i in range(len(lower_words) - len(alias_tokens) + 1):
            if lower_words[i : i + len(alias_tokens)] == alias_tokens:
                start = max(0, i - window_words // 2)
                end = min(len(words), i + len(alias_tokens) + window_words // 2)
                snippet = " ".join(words[start:end])
                if snippet not in snippets:
                    snippets.append(f"... {snippet} ...")
                if len(snippets) >= 2:
                    return snippets
    return snippets


@app.get("/api/taxonomy", response_model=Dict[str, Any])
def get_taxonomy():
    """Return approved taxonomy tags and aliases."""
    return {
        "tags": taxonomy.all_approved_tags(),
        "metadata": taxonomy.tags_metadata
    }


@app.post("/api/tag")
def tag_content(req: TagRequest):
    """Predict tags for an article draft."""
    result = engine.predict(
        title=req.title,
        body=req.body,
        model_name=req.model,
        custom_threshold=req.threshold
    )

    cleaned_full_text = f"{result['title']} {req.body}"
    for item in result["suggested_tags"]:
        item["supporting_snippets"] = extract_supporting_snippets(cleaned_full_text, item["tag"])

    return result


@app.post("/api/review/submit")
def submit_editorial_review(submission: FeedbackSubmission):
    """Saves editorial review decisions (accepted, rejected, added tags) as feedback."""
    Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
    feedback_records = []
    if Path(FEEDBACK_FILE).exists():
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            try:
                feedback_records = json.load(f)
            except Exception:
                feedback_records = []

    record = {
        "post_id": submission.post_id,
        "title": submission.title,
        "suggested_tags": submission.suggested_tags,
        "approved_tags": submission.approved_tags,
        "rejected_tags": submission.rejected_tags,
        "editor_notes": submission.editor_notes,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    feedback_records.append(record)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback_records, f, indent=2)

    return {"status": "success", "saved_feedback_count": len(feedback_records)}


@app.get("/api/batch/run")
def run_batch_evaluation():
    """Processes test posts, generates predictions, and saves batch export files."""
    test_path = Path("data/test.json")
    if not test_path.exists():
        raise HTTPException(status_code=404, detail="Test dataset not found. Run training pipeline first.")

    with open(test_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    rows = []
    for p in posts:
        res = engine.predict(p["title"], p["body"], model_name="tfidf")
        suggested = [t["tag"] for t in res["suggested_tags"]]
        scores = {t["tag"]: t["score"] for t in res["suggested_tags"]}
        
        # Ground truth acts as initial editor-approved reference in batch
        ground_truth = p.get("tags", [])

        rows.append({
            "post_id": p.get("id", f"post-{len(rows)+1}"),
            "title": p.get("title", ""),
            "suggested_tags": suggested,
            "scores": scores,
            "review_status": res["status"],
            "needs_review": res["needs_review"],
            "editor_approved_tags": ground_truth
        })

    df = pd.DataFrame(rows)
    df.to_json(BATCH_EXPORT_JSON, orient="records", indent=2)
    
    # Flat CSV export
    csv_rows = []
    for r in rows:
        csv_rows.append({
            "post_id": r["post_id"],
            "title": r["title"],
            "suggested_tags": ", ".join(r["suggested_tags"]),
            "scores_json": json.dumps(r["scores"]),
            "review_status": r["review_status"],
            "needs_review": r["needs_review"],
            "editor_approved_tags": ", ".join(r["editor_approved_tags"])
        })
    pd.DataFrame(csv_rows).to_csv(BATCH_EXPORT_CSV, index=False)

    return {
        "status": "success",
        "processed_count": len(rows),
        "exported_csv": BATCH_EXPORT_CSV,
        "exported_json": BATCH_EXPORT_JSON,
        "items": rows
    }


@app.get("/api/batch/export/csv")
def export_csv():
    """Download batch results as CSV."""
    if not Path(BATCH_EXPORT_CSV).exists():
        run_batch_evaluation()
    content = Path(BATCH_EXPORT_CSV).read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tagging_batch_export.csv"}
    )


@app.get("/api/batch/export/json")
def export_json():
    """Download batch results as JSON."""
    if not Path(BATCH_EXPORT_JSON).exists():
        run_batch_evaluation()
    content = Path(BATCH_EXPORT_JSON).read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=tagging_batch_export.json"}
    )


@app.get("/api/metrics")
def get_metrics():
    """Returns saved evaluation metrics and threshold sweeps."""
    report_file = Path("reports/benchmark_results.json")
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Run evaluation pipeline to generate metrics."}


# Dashboard HTML
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
    .header-banner { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; padding: 2rem 0; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .card { border: none; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 1.5rem; }
    .badge-tag { font-size: 0.95rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-weight: 500; display: inline-flex; align-items: center; gap: 6px; margin: 4px; }
    .tag-confidence { font-size: 0.8rem; opacity: 0.85; background: rgba(0,0,0,0.1); padding: 1px 6px; border-radius: 10px; }
    .score-bar { height: 6px; border-radius: 3px; background: #e2e8f0; overflow: hidden; margin-top: 4px; }
    .score-fill { height: 100%; background: #3b82f6; transition: width 0.3s; }
    .review-flag-warning { background-color: #fffbeb; border-left: 5px solid #f59e0b; padding: 1rem; border-radius: 8px; }
    .review-flag-danger { background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 1rem; border-radius: 8px; }
    .review-flag-success { background-color: #f0fdf4; border-left: 5px solid #22c55e; padding: 1rem; border-radius: 8px; }
    .snippet-box { font-size: 0.85rem; background: #f1f5f9; padding: 8px; border-radius: 6px; border-left: 3px solid #94a3b8; font-style: italic; margin-top: 6px; }
    .nav-tabs .nav-link { font-weight: 600; color: #64748b; }
    .nav-tabs .nav-link.active { color: #1e3a8a; border-bottom: 3px solid #3b82f6; }
    .btn-primary { background-color: #2563eb; border: none; }
    .btn-primary:hover { background-color: #1d4ed8; }
  </style>
</head>
<body>

<div class="header-banner">
  <div class="container">
    <div class="d-flex align-items-center justify-content-between">
      <div>
        <h2 class="mb-1"><i class="bi bi-tags-fill me-2"></i>Automated Content Tagging for Blogs</h2>
        <p class="mb-0 opacity-75">Multi-label ML Suggestion Pipeline & Human-in-the-Loop Review Dashboard</p>
      </div>
      <div>
        <span class="badge bg-light text-dark px-3 py-2"><i class="bi bi-shield-check text-success me-1"></i>Approved Taxonomy</span>
      </div>
    </div>
  </div>
</div>

<div class="container pb-5">

  <ul class="nav nav-tabs mb-4" id="mainTab" role="tablist">
    <li class="nav-item">
      <button class="nav-link active" id="tagger-tab" data-bs-toggle="tab" data-bs-target="#tagger" type="button"><i class="bi bi-lightning-charge me-1"></i>Interactive Tagger</button>
    </li>
    <li class="nav-item">
      <button class="nav-link" id="batch-tab" data-bs-toggle="tab" data-bs-target="#batch" type="button" onclick="loadBatchPosts()"><i class="bi bi-collection me-1"></i>Batch Review & Export</button>
    </li>
    <li class="nav-item">
      <button class="nav-link" id="metrics-tab" data-bs-toggle="tab" data-bs-target="#metrics" type="button" onclick="loadMetrics()"><i class="bi bi-graph-up me-1"></i>Model Evaluation</button>
    </li>
    <li class="nav-item">
      <button class="nav-link" id="taxonomy-tab" data-bs-toggle="tab" data-bs-target="#taxonomy" type="button" onclick="loadTaxonomy()"><i class="bi bi-journal-bookmark me-1"></i>Taxonomy Explorer</button>
    </li>
  </ul>

  <div class="tab-content" id="mainTabContent">

    <!-- 1. Interactive Tagger Tab -->
    <div class="tab-pane fade show active" id="tagger" role="tabpanel">
      <div class="row">
        <div class="col-lg-7">
          <div class="card p-4">
            <h5 class="card-title mb-3"><i class="bi bi-pencil-square me-2 text-primary"></i>Draft Article Input</h5>
            
            <div class="mb-3">
              <label class="form-label fw-bold">Article Title</label>
              <input type="text" id="draftTitle" class="form-control" placeholder="e.g. Securing FastAPI microservices with JWT and OAuth2" value="Securing FastAPI Microservices with JWT and Rate Limiting">
            </div>

            <div class="mb-3">
              <label class="form-label fw-bold">Article Body (Plain Text, Markdown, or HTML)</label>
              <textarea id="draftBody" class="form-control" rows="8" placeholder="Paste article body here...">In modern API development, securing backend endpoints is paramount. Using FastAPI with Python, we can implement robust OAuth2 bearer token authentication with JSON Web Tokens (JWT). We demonstrate password hashing using passlib and Bcrypt, token expiration validation, and middleware for sliding window rate limiting. Protecting against credential stuffing requires careful CORS configuration and secure headers.</textarea>
            </div>

            <div class="row g-3 mb-3">
              <div class="col-md-6">
                <label class="form-label fw-bold">Select Model</label>
                <select id="modelSelect" class="form-select">
                  <option value="tfidf" selected>TF-IDF + Logistic Regression (Fast Baseline)</option>
                  <option value="semantic">SentenceTransformer Embedding (Deep Semantic)</option>
                  <option value="keyword">Taxonomy Keyword/Alias Matcher</option>
                </select>
              </div>
              <div class="col-md-6">
                <label class="form-label fw-bold">Confidence Threshold: <span id="threshVal" class="text-primary">0.35</span></label>
                <input type="range" class="form-range" id="threshRange" min="0.10" max="0.80" step="0.05" value="0.35" oninput="document.getElementById('threshVal').innerText=this.value">
              </div>
            </div>

            <div class="d-flex gap-2">
              <button class="btn btn-primary px-4" onclick="runTagger()"><i class="bi bi-magic me-1"></i>Generate Tag Suggestions</button>
              <button class="btn btn-outline-secondary" onclick="loadSampleDraft('short')">Load Short Draft Edge Case</button>
              <button class="btn btn-outline-secondary" onclick="loadSampleDraft('mlops')">Load ML Article</button>
            </div>
          </div>
        </div>

        <div class="col-lg-5">
          <div class="card p-4">
            <h5 class="card-title mb-3"><i class="bi bi-stars me-2 text-primary"></i>Model Predictions & Review</h5>

            <div id="reviewBanner" class="mb-3 d-none"></div>

            <div id="suggestionsContainer">
              <p class="text-muted text-center py-5"><i class="bi bi-arrow-left-circle fs-3 d-block mb-2"></i>Enter an article and click <strong>Generate Tag Suggestions</strong>.</p>
            </div>

            <div id="editorControls" class="mt-4 pt-3 border-top d-none">
              <h6 class="fw-bold mb-2"><i class="bi bi-person-check me-1"></i>Editorial Approval Actions</h6>
              
              <div class="mb-3">
                <label class="form-label small text-muted">Add Missing Approved Tag:</label>
                <div class="input-group">
                  <select id="missingTagSelect" class="form-select form-select-sm"></select>
                  <button class="btn btn-sm btn-outline-primary" onclick="addCustomTag()">Add Tag</button>
                </div>
              </div>

              <div class="d-grid gap-2">
                <button class="btn btn-success" onclick="saveEditorialApproval()"><i class="bi bi-check2-circle me-1"></i>Approve & Save Decisions</button>
              </div>
              <div id="saveFeedbackMsg" class="mt-2 text-center small text-success"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 2. Batch Review & Export Tab -->
    <div class="tab-pane fade" id="batch" role="tabpanel">
      <div class="card p-4">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <div>
            <h5 class="card-title mb-0"><i class="bi bi-table me-2 text-primary"></i>Batch Evaluation & Review Export</h5>
            <p class="text-muted small mb-0">Evaluate test articles, verify review statuses, and export approved tag assignments.</p>
          </div>
          <div class="d-flex gap-2">
            <a href="/api/batch/export/csv" class="btn btn-outline-success btn-sm"><i class="bi bi-file-earmark-spreadsheet me-1"></i>Export CSV</a>
            <a href="/api/batch/export/json" class="btn btn-outline-primary btn-sm"><i class="bi bi-filetype-json me-1"></i>Export JSON</a>
            <button class="btn btn-primary btn-sm" onclick="loadBatchPosts()"><i class="bi bi-arrow-repeat me-1"></i>Refresh Batch</button>
          </div>
        </div>

        <div class="table-responsive">
          <table class="table table-hover align-middle" id="batchTable">
            <thead class="table-light">
              <tr>
                <th>ID</th>
                <th>Title</th>
                <th>Suggested Tags & Scores</th>
                <th>Review Status</th>
                <th>Editor Approved Tags</th>
              </tr>
            </thead>
            <tbody id="batchTableBody">
              <tr><td colspan="5" class="text-center py-4 text-muted">Loading batch data...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- 3. Metrics Tab -->
    <div class="tab-pane fade" id="metrics" role="tabpanel">
      <div class="row">
        <div class="col-lg-6">
          <div class="card p-4">
            <h5 class="card-title mb-3"><i class="bi bi-bar-chart-fill me-2 text-primary"></i>Model Comparison Table</h5>
            <div id="metricsTableContainer">Loading metrics...</div>
          </div>
        </div>
        <div class="col-lg-6">
          <div class="card p-4">
            <h5 class="card-title mb-3"><i class="bi bi-sliders me-2 text-primary"></i>Threshold Calibration Curve</h5>
            <div id="thresholdCurveContainer">Loading curve...</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 4. Taxonomy Explorer Tab -->
    <div class="tab-pane fade" id="taxonomy" role="tabpanel">
      <div class="card p-4">
        <h5 class="card-title mb-3"><i class="bi bi-diagram-3 me-2 text-primary"></i>Approved Tag Taxonomy & Mapped Aliases</h5>
        <div class="row g-3" id="taxonomyCards">Loading taxonomy...</div>
      </div>
    </div>

  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
  let activeSuggestions = [];
  let rejectedTags = [];
  let allTaxonomyTags = [];

  async function init() {
    const res = await fetch('/api/taxonomy');
    const data = await res.json();
    allTaxonomyTags = data.tags;
    const select = document.getElementById('missingTagSelect');
    select.innerHTML = data.tags.map(t => `<option value="${t}">${t}</option>`).join('');
  }
  init();

  async function runTagger() {
    const title = document.getElementById('draftTitle').value;
    const body = document.getElementById('draftBody').value;
    const model = document.getElementById('modelSelect').value;
    const threshold = parseFloat(document.getElementById('threshRange').value);

    const container = document.getElementById('suggestionsContainer');
    container.innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary" role="status"></div><p class="mt-2 text-muted">Running multi-label tagging pipeline...</p></div>';

    try {
      const res = await fetch('/api/tag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, body, model, threshold })
      });
      const data = await res.json();
      renderResults(data);
    } catch (e) {
      container.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
  }

  function renderResults(data) {
    const banner = document.getElementById('reviewBanner');
    banner.classList.remove('d-none', 'review-flag-warning', 'review-flag-danger', 'review-flag-success');

    if (data.status === 'flagged_short_draft') {
      banner.classList.add('review-flag-warning');
      banner.innerHTML = `<strong><i class="bi bi-exclamation-triangle-fill text-warning me-1"></i>Flagged for Review: Short Draft</strong><br><small>${data.review_reason}</small>`;
    } else if (data.status === 'flagged_low_confidence') {
      banner.classList.add('review-flag-danger');
      banner.innerHTML = `<strong><i class="bi bi-question-circle-fill text-danger me-1"></i>Flagged for Review: Low Confidence</strong><br><small>${data.review_reason}</small>`;
    } else {
      banner.classList.add('review-flag-success');
      banner.innerHTML = `<strong><i class="bi bi-check-circle-fill text-success me-1"></i>High Confidence Prediction</strong><br><small>All suggestions meet editorial thresholds (${data.word_count} words analyzed).</small>`;
    }

    activeSuggestions = [...(data.suggested_tags || [])];
    rejectedTags = [];
    renderTagsList();
    document.getElementById('editorControls').classList.remove('d-none');
    document.getElementById('saveFeedbackMsg').innerText = '';
  }

  function renderTagsList() {
    const container = document.getElementById('suggestionsContainer');
    if (!activeSuggestions.length) {
      container.innerHTML = '<p class="text-muted text-center py-3">No tags currently suggested. Use the dropdown below to assign approved tags.</p>';
      return;
    }

    let html = '<div class="list-group">';
    activeSuggestions.forEach((item, idx) => {
      const pct = Math.round(item.score * 100);
      const snippetsHtml = (item.supporting_snippets || []).map(s => `<div class="snippet-box"><i class="bi bi-quote"></i> ${s}</div>`).join('');
      html += `
        <div class="list-group-item p-3 mb-2 border rounded">
          <div class="d-flex justify-content-between align-items-center">
            <div>
              <span class="badge bg-primary badge-tag">${item.tag}</span>
              <span class="text-muted small ms-2">Score: <strong>${item.score.toFixed(3)}</strong> (${pct}%)</span>
            </div>
            <div>
              <button class="btn btn-sm btn-outline-danger" onclick="removeTag(${idx})"><i class="bi bi-x-lg"></i> Reject</button>
            </div>
          </div>
          <div class="score-bar"><div class="score-fill" style="width: ${pct}%"></div></div>
          ${snippetsHtml}
        </div>
      `;
    });
    html += '</div>';
    container.innerHTML = html;
  }

  function removeTag(idx) {
    const removed = activeSuggestions.splice(idx, 1)[0];
    rejectedTags.push(removed.tag);
    renderTagsList();
  }

  function addCustomTag() {
    const select = document.getElementById('missingTagSelect');
    const tag = select.value;
    if (tag && !activeSuggestions.some(s => s.tag === tag)) {
      activeSuggestions.push({ tag: tag, score: 1.0, supporting_snippets: ["Manually assigned by editor"] });
      renderTagsList();
    }
  }

  async function saveEditorialApproval() {
    const title = document.getElementById('draftTitle').value;
    const approved_tags = activeSuggestions.map(s => s.tag);
    const suggested_tags = activeSuggestions.map(s => s.tag).concat(rejectedTags);

    const res = await fetch('/api/review/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        suggested_tags,
        approved_tags,
        rejected_tags
      })
    });
    const data = await res.json();
    document.getElementById('saveFeedbackMsg').innerText = `✓ Decisions saved to feedback log (${data.saved_feedback_count} total records).`;
  }

  function loadSampleDraft(type) {
    if (type === 'short') {
      document.getElementById('draftTitle').value = 'WIP: Redis Notes';
      document.getElementById('draftBody').value = 'Need to configure LRU eviction on Redis cache cluster.';
    } else {
      document.getElementById('draftTitle').value = 'Distributed Model Training with PyTorch and Spark';
      document.getElementById('draftBody').value = 'Training foundation deep learning models on large multi-node clusters requires efficient data parallel pipelines. In this article, we use Python and Apache Spark for ETL data processing and PyTorch DistributedDataParallel for multi-GPU training.';
    }
    runTagger();
  }

  async function loadBatchPosts() {
    const body = document.getElementById('batchTableBody');
    body.innerHTML = '<tr><td colspan="5" class="text-center py-4"><div class="spinner-border spinner-border-sm text-primary"></div> Processing test dataset...</td></tr>';
    const res = await fetch('/api/batch/run');
    const data = await res.json();

    let html = '';
    data.items.forEach(item => {
      const tagsBadges = item.suggested_tags.map(t => `<span class="badge bg-secondary badge-tag me-1">${t}</span>`).join('');
      const approvedBadges = item.editor_approved_tags.map(t => `<span class="badge bg-success badge-tag me-1">${t}</span>`).join('');
      const statusBadge = item.needs_review ? 
        `<span class="badge bg-warning text-dark"><i class="bi bi-flag-fill me-1"></i>Review Needed (${item.review_status})</span>` :
        `<span class="badge bg-success"><i class="bi bi-check-circle me-1"></i>Approved</span>`;

      html += `
        <tr>
          <td><code>${item.post_id}</code></td>
          <td class="fw-bold">${item.title}</td>
          <td>${tagsBadges || '<span class="text-muted small">None</span>'}</td>
          <td>${statusBadge}</td>
          <td>${approvedBadges}</td>
        </tr>
      `;
    });
    body.innerHTML = html;
  }

  async function loadMetrics() {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    if (!data.test_results) return;

    let table = '<table class="table table-bordered"><thead><tr class="table-light"><th>Model</th><th>Micro-F1</th><th>Macro-F1</th><th>Precision@3</th><th>Recall@3</th><th>Latency</th></tr></thead><tbody>';
    for (const [k, v] of Object.entries(data.test_results)) {
      table += `<tr><td><strong>${k.toUpperCase()}</strong></td><td>${v.micro_f1}</td><td>${v.macro_f1}</td><td>${v['precision@3']}</td><td>${v['recall@3']}</td><td>${v.avg_latency_ms} ms</td></tr>`;
    }
    table += '</tbody></table>';
    document.getElementById('metricsTableContainer').innerHTML = table;

    let curveTable = '<table class="table table-sm"><thead><tr class="table-light"><th>Threshold</th><th>Micro-F1</th><th>Precision@3</th><th>Recall@3</th><th>Review Rate</th></tr></thead><tbody>';
    (data.threshold_sweep_tfidf || []).forEach(row => {
      curveTable += `<tr><td>${row.threshold}</td><td>${row.micro_f1}</td><td>${row['precision@3']}</td><td>${row['recall@3']}</td><td>${Math.round(row.review_coverage_rate*100)}%</td></tr>`;
    });
    curveTable += '</tbody></table>';
    document.getElementById('thresholdCurveContainer').innerHTML = curveTable;
  }

  async function loadTaxonomy() {
    const res = await fetch('/api/taxonomy');
    const data = await res.json();
    let cards = '';
    for (const [tag, info] of Object.entries(data.metadata)) {
      cards += `
        <div class="col-md-6 col-lg-4">
          <div class="card h-100 p-3">
            <h6 class="fw-bold text-primary mb-1">${tag}</h6>
            <p class="small text-muted mb-2">${info.definition}</p>
            <div class="mb-2">
              <span class="small fw-bold">Aliases:</span><br>
              ${info.aliases.map(a => `<span class="badge bg-light text-dark border me-1">${a}</span>`).join('')}
            </div>
          </div>
        </div>
      `;
    }
    document.getElementById('taxonomyCards').innerHTML = cards;
  }
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the interactive editorial review and tagging dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)
