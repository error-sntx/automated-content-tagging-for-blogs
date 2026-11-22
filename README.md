# Automated Content Tagging for Blogs

An end-to-end multi-label machine learning and editorial decision system that ingests blog article drafts (titles and content in plain text, Markdown, or HTML), suggests relevant tags from an approved taxonomy, and routes uncertain or thin drafts to human editors for review.

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
                          | - Heading Preservation |
                          | - Word-Count Guardrail |
                          +-----------+------------+
                                      |
                       +--------------+--------------+
                       |                             |
             [Words < 15 / Empty]             [Valid Length]
                       |                             |
                       v                             v
           +-----------------------+     +-----------------------+
           | Flagged: Short Draft  |     |   Multi-Label Models  |
           | Route to Editorial    |     | 1. Keyword/Alias Base |
           +-----------------------+     | 2. TF-IDF + OvR LogReg|
                                         | 3. Chunked MiniLM DL  |
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
                        | + Related Articles (3)|        | Route to Editorial    |
                        +-----------------------+        +-----------------------+
                                    |                                |
                                    +----------------+---------------+
                                                     |
                                                     v
                                         +-----------------------+
                                         | Human-in-the-Loop     |
                                         | Editorial Dashboard   |
                                         | - 1-Click Approval    |
                                         | - Feedback Logging    |
                                         | - Batch Export        |
                                         +-----------------------+
```

---

## 📁 Repository Structure

```
PROJECT/
├── data/
│   ├── articles.csv              # Official 96 articles across Train (48), Val (24), Test (24)
│   ├── labels.csv                # Official gold category and tag sets
│   ├── taxonomy.csv              # Flat 2-level taxonomy (6 categories, 18 canonical tags)
│   └── aliases.csv               # Official alias mapping dictionary
├── models/
│   ├── tfidf_model.joblib        # Fitted TF-IDF + OneVsRest Logistic Regression
│   └── semantic_model.joblib     # Pretrained Chunked MiniLM multi-label classifier
├── reports/
│   └── evaluation_report.md      # Comprehensive technical evaluation & error analysis
├── src/
│   ├── __init__.py
│   ├── taxonomy.py               # Taxonomy manager, alias resolution, Apple disambiguation
│   ├── preprocess.py             # HTML/Markdown cleaner, heading preservation
│   ├── engine.py                 # Multi-label inference engine, guardrails & routing
│   ├── train_and_evaluate.py     # Master training, threshold calibration, and evaluation
│   ├── app.py                    # FastAPI service + interactive editorial dashboard
│   └── models/
│       ├── __init__.py
│       ├── keyword_model.py      # Keyword matcher with contextual disambiguation
│       ├── tfidf_model.py        # Multi-label TF-IDF + One-vs-Rest Logistic Regression
│       ├── semantic_model.py     # Chunked SentenceTransformer model
│       └── recommender.py        # Top-3 related articles recommender (excludes self & sibling)
├── tests/
│   └── test_pipeline.py          # Automated unit test suite (7 tests)
├── baseline.py                   # Provided single-category naive word-count baseline
├── predictions.csv               # Official predictions (article_id, category_ids, tags)
├── metrics.json                  # Official validation & test metrics
├── editor_feedback.csv           # Human-in-the-loop review log matching template
├── DATA_DICTIONARY.md            # Official data dictionary documentation
├── PROJECT_VARIANT_GUIDE.md      # Official task & variant guide
├── EDITOR_FEEDBACK_TEMPLATE.csv  # Official review feedback format specification
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation & usage guide
```

---

## 🏷️ Approved Taxonomy & Alias Mapping

The official taxonomy consists of 6 categories and 18 canonical tags:

| Category ID | Parent Category | Category Name | Canonical Tags |
| :--- | :--- | :--- | :--- |
| `cloud` | Technology | Cloud Computing | `cloud`, `deployment`, `server` |
| `ml` | Technology | Machine Learning | `model`, `training`, `prediction` |
| `finance` | Business | Personal Finance | `budget`, `savings`, `expenses` |
| `nutrition` | Lifestyle | Nutrition | `fruit`, `meal`, `nutrition` |
| `web` | Technology | Web Development | `react`, `browser`, `interface` |
| `devices` | Technology | Consumer Devices | `apple`, `device`, `battery` |

### 🍏 Contextual Disambiguation for "Apple"
- **Problem:** Naive string matching maps any mention of "apple" to the technology company tag under `devices`.
- **Solution:** A context-aware disambiguator inspects co-occurring tokens:
  - If surrounding context mentions *fruit, recipe, lunch, meal, salad, vegetables* $\rightarrow$ maps to `fruit` under `nutrition`.
  - If surrounding context mentions *device, battery, screen, hardware, company, phone* $\rightarrow$ maps to `apple` under `devices`.

---

## 📊 Benchmark Results

Evaluated on official held-out test articles (24 articles):

| Model | Micro-F1 | Macro-F1 | Exact Match | Precision@3 | Recall@3 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Provided Baseline (Single-Label)** | 0.8889 | - | 75.0% | 0.8333 | 0.7500 | 0.5 ms |
| **Multi-Label TF-IDF (Our Model)** | **0.9375** | **0.9524** | **83.3%** | **1.0000** | **0.8750** | **1.1 ms** |
| **Chunked MiniLM Semantic Model** | 0.9167 | 0.9167 | 79.2% | 0.9583 | 0.8333 | 28.5 ms |

- **Multi-Topic Mastery:** The naive baseline fails on multi-topic posts (`cloud|ml`, `web|devices`, `finance|nutrition`) by forcing a single label. Our model captures multi-label combinations accurately.
- **Related Articles:** Automatically recommends 3 related articles per draft using TF-IDF cosine similarity while enforcing the constraint to exclude the query article itself and its formatting sibling (`group_id`).

---

## 🚀 Quickstart & Commands

### 1. Run with Virtual Environment (Recommended)
```powershell
# Run the provided naive baseline
.\.venv\Scripts\python.exe baseline.py

# Train & evaluate multi-label models (generates predictions.csv & metrics.json)
.\.venv\Scripts\python.exe -m src.train_and_evaluate

# Run the unit test suite
.\.venv\Scripts\python.exe -m unittest discover tests

# Launch the interactive editorial review dashboard & API
.\.venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

### 2. Run with Global Python (After installing dependencies)
```powershell
pip install -r requirements.txt
python -m src.train_and_evaluate
python -m unittest discover tests
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access the interactive tagging UI.
