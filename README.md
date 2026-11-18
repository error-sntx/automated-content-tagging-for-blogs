# Automated Content Tagging for Blogs

An end-to-end multi-label machine learning and editorial decision system that ingests blog article drafts (titles and bodies in plain text, Markdown, or HTML), suggests relevant tags from an approved taxonomy, and routes uncertain or thin drafts to human editors for review.

---

## 🎯 Architecture Overview

```
                          +------------------------+
                          |   Raw Article Draft    |
                          | (Markdown / HTML / TXT)|
                          +-----------+------------+
                                      |
                                      v
                          +------------------------+
                          | Preprocessing Pipeline |
                          | - HTML/Boilerplate Strip
                          | - Title-Weighted Combo |
                          | - Word-Count Guardrail |
                          +-----------+------------+
                                      |
                       +--------------+--------------+
                       |                             |
             [Words < 25 / Empty]             [Valid Length]
                       |                             |
                       v                             v
           +-----------------------+     +-----------------------+
           | Flagged: Short Draft  |     |   Multi-Label Models  |
           | Route to Editorial    |     | 1. Taxonomy Keyword   |
           +-----------------------+     | 2. TF-IDF + OvR LogReg|
                                         | 3. Chunked Semantic   |
                                         +-----------+-----------+
                                                     |
                                                     v
                                         +-----------------------+
                                         |  Decision Engine &    |
                                         | Threshold Calibration |
                                         +-----------+-----------+
                                                     |
                                    +----------------+---------------+
                                    |                                |
                        [Any Score >= Threshold]           [All Scores < Thresh]
                                    |                                |
                                    v                                v
                        +-----------------------+        +-----------------------+
                        | Ranked Top-5 Tags     |        | Flagged: Low Conf     |
                        | + Supporting Snippets |        | Route to Editorial    |
                        +-----------------------+        +-----------------------+
                                    |                                |
                                    +----------------+---------------+
                                                     |
                                                     v
                                         +-----------------------+
                                         | Human-in-the-Loop     |
                                         | Editorial Dashboard   |
                                         | - Accept / Reject     |
                                         | - Add Missing Tags    |
                                         | - Export Batch CSV/JSON
                                         +-----------------------+
```

---

## 📁 Repository Structure

```
PROJECT/
├── data/
│   ├── taxonomy.json             # Approved 12-tag taxonomy with definitions & aliases
│   ├── blog_posts.json           # Labeled corpus of technical blog posts
│   ├── train.json                # Temporal training split
│   ├── val.json                  # Validation split for threshold tuning
│   ├── test.json                 # Held-out recent articles test split
│   ├── batch_export.csv          # Exported CSV with IDs, tags, scores, review statuses
│   ├── batch_export.json         # Exported JSON batch file
│   └── editorial_feedback.json   # Log of human editor approvals and adjustments
├── models/
│   ├── all-MiniLM-L6-v2/         # Pretrained SentenceTransformer model
│   ├── tfidf_model.joblib        # Serialized TF-IDF vectorizer + OvR classifier
│   └── semantic_model.joblib     # Serialized semantic multi-label classifier
├── reports/
│   ├── evaluation_report.md      # Detailed model comparison and analysis
│   └── benchmark_results.json    # Machine-readable performance metrics
├── src/
│   ├── __init__.py
│   ├── taxonomy.py               # Taxonomy manager, alias resolution, canonical mapper
│   ├── preprocess.py             # HTML/boilerplate stripper, deduplication, temporal split
│   ├── generate_dataset.py       # Corpus generator with edge cases & timestamps
│   ├── engine.py                 # Tagging inference engine, guardrails & routing
│   ├── evaluate.py               # Micro/Macro F1, Precision@3, Recall@3, coverage
│   ├── train_and_evaluate.py     # Master pipeline: train, calibrate, evaluate
│   ├── app.py                    # FastAPI service + interactive editorial dashboard
│   └── models/
│       ├── __init__.py
│       ├── keyword_model.py      # Baseline 1: Regex & alias matcher
│       ├── tfidf_model.py        # Baseline 2: Weighted TF-IDF + Logistic Regression
│       └── semantic_model.py     # Semantic 3: Long-text chunking + MiniLM embeddings
├── tests/
│   └── test_pipeline.py          # 12 automated unit tests across the entire stack
└── README.md                     # Documentation & usage guide
```

---

## 🏷️ Approved Tag Taxonomy

The taxonomy contains 12 standardized editorial tags across software engineering and machine learning, with canonical names, definitions, and an alias map:

| Canonical Tag | Example Aliases Mapped |
| :--- | :--- |
| **Machine Learning** | `ml`, `machinelearning`, `statistical learning`, `supervised learning` |
| **Deep Learning** | `dl`, `neural networks`, `transformers`, `llm`, `large language models` |
| **Natural Language Processing** | `nlp`, `text mining`, `text processing`, `ner` |
| **Python** | `python3`, `py`, `cpython`, `python programming` |
| **API Development** | `apis`, `rest api`, `graphql`, `grpc`, `fastapi`, `endpoints` |
| **Application Security** | `appsec`, `cybersecurity`, `security`, `jwt`, `oauth`, `vulnerability` |
| **DevOps & Cloud** | `devops`, `cloud`, `aws`, `docker`, `kubernetes`, `k8s`, `ci/cd`, `terraform` |
| **Database Systems** | `databases`, `sql`, `nosql`, `postgres`, `mysql`, `mongodb`, `redis` |
| **Frontend Development** | `frontend`, `react`, `reactjs`, `vue`, `javascript`, `typescript`, `css` |
| **Data Engineering** | `data-engineering`, `etl`, `spark`, `kafka`, `data warehouse`, `airflow` |
| **Software Architecture** | `system design`, `architecture`, `microservices`, `design patterns` |
| **Performance Optimization** | `performance`, `optimization`, `profiling`, `latency`, `caching` |

---

## 🧪 Quickstart & Reproduction

### 1. Run Automated Unit Tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```
Runs 12 unit tests verifying taxonomy resolution, boilerplate stripping, near-duplicate hashing, guardrail routing, models, and FastAPI endpoints.

### 2. Retrain and Evaluate Models
```powershell
.\.venv\Scripts\python.exe -m src.train_and_evaluate
```
This runs the full pipeline:
1. Loads taxonomy and generates/preprocesses the dataset.
2. Chronologically splits data (Train: 70%, Val: 15%, Test: 15%).
3. Trains the TF-IDF and Semantic models.
4. Performs threshold sweep on validation posts ($0.20$ to $0.70$).
5. Evaluates models on unseen test posts and outputs `reports/evaluation_report.md`.

### 3. Launch the Interactive Web Dashboard & API
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access:
- **Interactive Tagger:** Paste drafts, select models, adjust thresholds, view tags and supporting snippet quotes.
- **Human-in-the-Loop Review:** 1-click accept/reject, add missing taxonomy tags, and save feedback.
- **Batch Review & Export:** Process test sets and download results as **CSV** or **JSON**.
- **Model Evaluation:** Live view of metrics, threshold curves, and error analysis.
- **Taxonomy Explorer:** View all approved tags, definitions, and aliases.

---

## 📊 Evaluation Summary

Tested on held-out recent blog articles:

| Model | Micro-F1 | Macro-F1 | Precision@3 | Recall@3 | Latency | Review Route Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Keyword Matcher (Baseline)** | 0.8000 | 0.6111 | 0.7500 | 0.6667 | 1.2 ms | 16.7% |
| **TF-IDF + OvR Logistic Regression** | 0.5000 | 0.5028 | 0.3333 | 0.5000 | 4.6 ms | 16.7% |
| **Semantic SentenceTransformer** | 0.4444 | 0.4444 | 0.3889 | 0.5833 | 30.3 ms | 16.7% |

- **Edge Case Protection:** Drafts under 25 words are automatically intercepted (`flagged_short_draft`) without false model inferences.
- **Duplicate Shielding:** Content fingerprinting keeps republished near-duplicates in the same data split to prevent data contamination.
