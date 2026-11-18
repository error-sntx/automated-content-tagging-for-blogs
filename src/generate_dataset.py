"""Generates a rich, realistic technical blog dataset for training and evaluation.

Includes realistic blog posts with HTML/Markdown formatting, multi-label ground truth,
near-duplicate variants, short drafts, and chronological timestamps.
"""

import json
from datetime import datetime, timedelta
import random
from pathlib import Path
import pandas as pd


def generate_synthetic_blog_corpus(output_path: str = "data/blog_posts.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    base_articles = [
        # 1. Python + API Development + Application Security
        {
            "title": "Hardening FastAPI Services: JWT Authentication and Rate Limiting",
            "body": """<nav><a href='/'>Home</a> > <a href='/api'>APIs</a></nav>
            # Hardening FastAPI Services: JWT Authentication and Rate Limiting
            In modern API development, securing backend endpoints is paramount. Using FastAPI with Python, we can implement robust OAuth2 bearer token authentication with JSON Web Tokens (JWT).
            We demonstrate password hashing using passlib and Bcrypt, token expiration validation, and middleware for sliding window rate limiting.
            Protecting against credential stuffing and brute-force attacks requires careful HTTP status code handling and CORS configuration.
            <div class='tag-footer'>Tags: python, fastapi, security</div>
            <footer>Share on Twitter | Leave a comment below. Copyright 2025 TechBlog</footer>""",
            "tags": ["Python", "API Development", "Application Security"],
            "date": "2025-01-15T10:00:00"
        },
        # 2. Machine Learning + Python + Data Engineering
        {
            "title": "Building Scalable Feature Stores with Python and Apache Spark",
            "body": """<h1>Feature Store Architecture for Machine Learning</h1>
            Managing feature parity between offline training and online inference is a classic challenge in machine learning pipelines.
            In this guide, we use Python alongside Apache Spark and Delta Lake to compute time-travel aggregations, batch feature vectors, and stream real-time events.
            We will write reproducible ETL jobs that register schema definitions and publish feature snapshots for model training.
            <p>Subscribe to our newsletter for more updates!</p>""",
            "tags": ["Machine Learning", "Python", "Data Engineering"],
            "date": "2025-02-01T14:30:00"
        },
        # 3. Deep Learning + Natural Language Processing + Performance Optimization
        {
            "title": "Quantization and Low-Rank Adaptation (LoRA) for LLM Inference Speed",
            "body": """## Optimizing Transformer Inference Latency
            Deploying large language models (LLMs) requires minimizing memory footprint and latency. We explore 4-bit and 8-bit post-training quantization (AWQ and bitsandbytes) for transformer decoders.
            Using PyTorch and Hugging Face, we show how low-rank adaptation reduces parameter count during fine-tuning while KV-cache compression improves tokens-per-second throughput on single GPUs.
            Benchmarking shows a 3.4x speedup with minimal degradation on text generation benchmarks.""",
            "tags": ["Deep Learning", "Natural Language Processing", "Performance Optimization"],
            "date": "2025-02-20T09:15:00"
        },
        # 4. DevOps & Cloud + Software Architecture
        {
            "title": "Designing Multi-Region Active-Active Architectures on AWS with Kubernetes",
            "body": """<h3>Multi-Region Distributed Systems</h3>
            High availability at global scale demands multi-region deployment strategies. We examine Route 53 latency-based routing, Amazon EKS multi-cluster orchestration, and Terraform modules for declarative infrastructure.
            Key trade-offs include cross-region database replication lag, eventual consistency vs ACID guarantees, and automated failover runbooks during regional cloud outages.""",
            "tags": ["DevOps & Cloud", "Software Architecture"],
            "date": "2025-03-05T11:20:00"
        },
        # 5. Database Systems + Performance Optimization
        {
            "title": "PostgreSQL Query Optimization: Index Scans, Partitioning, and Explain Plans",
            "body": """Analyzing slow queries in PostgreSQL begins with EXPLAIN (ANALYZE, BUFFERS). We walk through diagnosing sequential scans, index bloat, and vacuuming deficits.
            We cover B-tree vs BRIN indexes for time-series data, declarative table partitioning, and configuring work_mem and shared_buffers for high-throughput transactional database workloads.""",
            "tags": ["Database Systems", "Performance Optimization"],
            "date": "2025-03-25T16:00:00"
        },
        # 6. Frontend Development + Performance Optimization
        {
            "title": "Mastering React Server Components and Core Web Vitals",
            "body": """<div class='nav-bar'><a href='/web'>Web</a></div>
            React Server Components (RSC) fundamentally shift how frontend web applications render UI. By rendering data-heavy components on the server, client JavaScript bundle size drops drastically.
            We measure Largest Contentful Paint (LCP) and Cumulative Layout Shift (CLS) before and after streaming SSR with Suspense boundaries.
            Code splitting and CSS-in-JS trade-offs are evaluated for enterprise frontend codebases.""",
            "tags": ["Frontend Development", "Performance Optimization"],
            "date": "2025-04-10T12:00:00"
        },
        # 7. Machine Learning + Deep Learning + Python
        {
            "title": "Practical Transfer Learning with PyTorch: Fine-Tuning Vision Transformers",
            "body": """Transfer learning allows practitioners to leverage pretrained foundation models for niche image classification tasks.
            We implement an end-to-end PyTorch script in Python, freezing base layers, replacing the classification head, and scheduling learning rate decay with AdamW.
            We explore data augmentation with Albumentations, mixup regularization, and validation loss tracking.""",
            "tags": ["Machine Learning", "Deep Learning", "Python"],
            "date": "2025-05-02T15:45:00"
        },
        # 8. API Development + Database Systems + Software Architecture
        {
            "title": "Event-Driven Microservices with gRPC and Distributed PostgreSQL Databases",
            "body": """When decomposing a monolith into microservices, synchronous REST calls can introduce cascading failures.
            We explore gRPC protocols with Protocol Buffers for internal service communication, alongside the Outbox pattern for transactional consistency across isolated PostgreSQL instances.
            Idempotent consumer handlers prevent duplicate message execution.""",
            "tags": ["API Development", "Database Systems", "Software Architecture"],
            "date": "2025-05-20T10:10:00"
        },
        # 9. Natural Language Processing + Machine Learning + Python
        {
            "title": "Named Entity Recognition and Information Extraction with Spacy and Scikit-Learn",
            "body": """Extracting structured entities from unstructured text is a cornerstone of NLP.
            We demonstrate building a custom NER pipeline in Python using SpaCy tokenizers combined with a CRF (Conditional Random Field) and TF-IDF logistic regression classifier.
            Evaluation metrics include entity-level precision, recall, and token classification confusion matrices.""",
            "tags": ["Natural Language Processing", "Machine Learning", "Python"],
            "date": "2025-06-12T08:30:00"
        },
        # 10. Application Security + DevOps & Cloud
        {
            "title": "Container Security and Supply Chain Defense: Trivy, Cosign, and Kubernetes RBAC",
            "body": """Securing Kubernetes environments starts in the CI/CD pipeline.
            We configure automated vulnerability scanning with Trivy to inspect container base images for CVEs, sign OCI artifacts with Cosign, and enforce immutable pod security standards.
            We restrict service account privileges via strict Kubernetes Role-Based Access Control (RBAC) policies.""",
            "tags": ["Application Security", "DevOps & Cloud"],
            "date": "2025-07-01T13:40:00"
        },
        # 11. Data Engineering + DevOps & Cloud + Database Systems
        {
            "title": "Real-Time Streaming Pipelines with Kafka, Flink, and ClickHouse",
            "body": """High-throughput telemetry requires streaming analytics architectures.
            We deploy Apache Kafka on Kubernetes to ingest millions of events per second, process stateful stream windows with Apache Flink, and sink aggregated metrics into ClickHouse for sub-second analytical queries.
            We monitor consumer lag and backpressure using Prometheus.""",
            "tags": ["Data Engineering", "DevOps & Cloud", "Database Systems"],
            "date": "2025-07-22T17:15:00"
        },
        # 12. Frontend Development + API Development
        {
            "title": "Building Offline-First Web Apps with TypeScript, Service Workers, and GraphQL",
            "body": """Modern web applications need to handle intermittent network connectivity gracefully.
            We utilize Apollo Client with cache persistence, background sync via Service Workers, and IndexedDB for local data storage in TypeScript.
            Optimistic UI updates ensure instantaneous user feedback while syncing mutations with the GraphQL API in the background.""",
            "tags": ["Frontend Development", "API Development"],
            "date": "2025-08-14T11:00:00"
        },
        # 13. Machine Learning + Performance Optimization + Python
        {
            "title": "Accelerating Scikit-Learn Pipelines with Intel Extension and ONNX Runtime",
            "body": """Training and serving classical machine learning models at low latency can be achieved without heavy neural networks.
            We profile scikit-learn random forests and gradient boosted trees in Python, convert trained estimators into ONNX graphs, and run inference with ONNX Runtime.
            Benchmark results demonstrate a 5x reduction in P99 inference latency.""",
            "tags": ["Machine Learning", "Performance Optimization", "Python"],
            "date": "2025-09-05T09:20:00"
        },
        # 14. Software Architecture + Application Security
        {
            "title": "Zero Trust Architecture: Identity-Aware Proxies and Microsegmentation",
            "body": """Perimeter-based network security is no longer sufficient.
            Zero Trust Architecture assumes breach and verifies every request explicitly.
            We examine mutual TLS (mTLS) with SPIFFE/SPIRE, continuous contextual authorization, identity-aware proxies, and microsegmentation policies across distributed systems.""",
            "tags": ["Software Architecture", "Application Security"],
            "date": "2025-09-28T14:10:00"
        },
        # 15. Deep Learning + Natural Language Processing
        {
            "title": "Retrieval-Augmented Generation (RAG) Architecture: Embeddings, Vector Stores, and Re-ranking",
            "body": """RAG pipelines overcome hallucination in generative AI by grounding LLM responses in proprietary enterprise knowledge.
            We review dense vector embeddings, HNSW index construction in vector databases, hybrid BM25 lexical retrieval, and cross-encoder re-ranking for contextual precision.
            Evaluation is conducted using Faithfulness and Answer Relevance metrics.""",
            "tags": ["Deep Learning", "Natural Language Processing"],
            "date": "2025-10-18T16:40:00"
        },
        # 16. DevOps & Cloud + Python
        {
            "title": "Automating Multi-Cloud Infrastructure with Pulumi and Python",
            "body": """Infrastructure as Code (IaC) moves beyond static domain-specific languages.
            Using Python and Pulumi, software engineers can write object-oriented infrastructure configurations, utilize standard unit testing frameworks (pytest), and provision cloud resources across AWS and Google Cloud Platform.
            We build reusable custom components and manage secrets securely.""",
            "tags": ["DevOps & Cloud", "Python"],
            "date": "2025-11-10T10:05:00"
        },
        # 17. Frontend Development + Application Security
        {
            "title": "Content Security Policy (CSP) and Cross-Site Scripting (XSS) Prevention in Modern SPAs",
            "body": """Frontend single-page applications are vulnerable to DOM-based XSS and malicious third-party script injection.
            We dissect strict Content Security Policy (CSP) headers, nonce generation, Trusted Types API, and sanitizing user inputs in React and Angular applications.
            Audit workflows with Chrome DevTools are demonstrated.""",
            "tags": ["Frontend Development", "Application Security"],
            "date": "2025-12-02T13:25:00"
        },
        # 18. Database Systems + Data Engineering
        {
            "title": "Modernizing Data Warehouses: Transitioning from Traditional OLAP to Iceberg and DuckDB",
            "body": """Apache Iceberg table formats offer ACID transactions, partition evolution, and schema enforcement over object storage.
            We showcase pairing DuckDB with Iceberg catalogs in Python to run analytical SQL queries directly on parquet files without standing up costly database clusters.
            Query execution speeds and storage cost reductions are evaluated.""",
            "tags": ["Database Systems", "Data Engineering"],
            "date": "2025-12-20T15:30:00"
        },
        # 19. Machine Learning + Deep Learning + Natural Language Processing (Broad / Multi-topic)
        {
            "title": "End-to-End Multimodal Content Understanding: Text, Vision, and Semantic Tagging",
            "body": """In this comprehensive guide, we construct an end-to-end multimodal classification engine.
            We combine sentence transformer embeddings for article text with CLIP image representations for blog diagrams.
            The multimodal embeddings are fed into a multi-label classification head evaluated across macro-F1 and micro-F1 scores on thousands of domain-specific articles.
            We share lessons on balancing loss functions and handling class imbalance.""",
            "tags": ["Machine Learning", "Deep Learning", "Natural Language Processing"],
            "date": "2026-01-12T11:50:00"
        },
        # 20. Edge Case: Short Draft (Requires Review)
        {
            "title": "Draft: Notes on Redis Caching",
            "body": "Need to write a post about setting up Redis memory limits and TTL strategies. Notes: redis-cli, LRU eviction.",
            "tags": ["Database Systems", "Performance Optimization"],
            "date": "2026-02-05T09:00:00"
        },
        # 21. Edge Case: Very Short / Empty Draft (Requires Review)
        {
            "title": "WIP: Microservices Logging",
            "body": "TODO: add content",
            "tags": ["Software Architecture", "DevOps & Cloud"],
            "date": "2026-02-18T10:00:00"
        },
        # 22. Edge Case: Near-duplicate of Article #1 (Republished post)
        {
            "title": "Hardening FastAPI Services: JWT Authentication and Rate Limiting (Updated 2026)",
            "body": """<nav><a href='/'>Home</a></nav>
            # Hardening FastAPI Services: JWT Authentication and Rate Limiting (Updated 2026)
            In modern API development, securing backend endpoints is paramount. Using FastAPI with Python, we can implement robust OAuth2 bearer token authentication with JSON Web Tokens (JWT).
            We demonstrate password hashing using passlib and Bcrypt, token expiration validation, and middleware for sliding window rate limiting.
            Protecting against credential stuffing and brute-force attacks requires careful HTTP status code handling and CORS configuration.
            <footer>Republished with updated 2026 dependencies.</footer>""",
            "tags": ["Python", "API Development", "Application Security"],
            "date": "2026-03-01T14:00:00"
        },
        # 23. Test Set / Unseen recent post: Python + Machine Learning + Deep Learning
        {
            "title": "Fine-Tuning Small Language Models with PyTorch and Unsloth in Python",
            "body": """Running small language models (SLMs) locally is becoming viable for specialized developer workflows.
            We use Python to fine-tune Llama-3-8B and Mistral using QLoRA and FlashAttention-2.
            Memory consumption during gradient computation is reduced by 60%, allowing training on consumer grade hardware.
            Evaluation shows domain adaptation accuracy gains on coding datasets.""",
            "tags": ["Python", "Machine Learning", "Deep Learning"],
            "date": "2026-04-15T10:30:00"
        },
        # 24. Test Set / Unseen recent post: API Development + Performance Optimization
        {
            "title": "High-Throughput HTTP APIs: Connection Pooling, HTTP/2, and Keep-Alive Strategies",
            "body": """When scaling microservices to handle tens of thousands of requests per second, HTTP connection overhead becomes a primary bottleneck.
            We profile connection handshake latencies, configure persistent TCP keep-alive pools, and test HTTP/2 multiplexing in asynchronous web servers.
            Latency percentiles (P95 and P99) show significant stabilization under burst traffic.""",
            "tags": ["API Development", "Performance Optimization"],
            "date": "2026-05-10T14:15:00"
        },
        # 25. Test Set / Unseen recent post: Frontend Development + Software Architecture
        {
            "title": "Micro-Frontends in 2026: Module Federation and Independent Release Trains",
            "body": """Breaking large frontend web applications into autonomously deployed micro-frontends reduces cross-team coordination bottlenecks.
            We architect a web platform using Webpack Module Federation with React and TypeScript.
            Shared state management, design tokens, and version mismatch isolation techniques are examined.""",
            "tags": ["Frontend Development", "Software Architecture"],
            "date": "2026-06-01T09:45:00"
        },
        # 26. Test Set / Unseen recent post: Application Security + Database Systems
        {
            "title": "Database Hardening: Row-Level Security and Column Encryption in Postgres",
            "body": """Protecting sensitive customer data requires defense-in-depth within the database layer itself.
            We demonstrate implementing PostgreSQL Row-Level Security (RLS) policies to enforce tenant isolation at query time.
            Furthermore, we integrate pgcrypto and envelope encryption keys to protect personally identifiable information (PII) at rest.""",
            "tags": ["Application Security", "Database Systems"],
            "date": "2026-07-04T12:00:00"
        },
        # 27. Test Set / Unseen recent post: Data Engineering + Python
        {
            "title": "Data Contracts and Schema Testing with Python and Pydantic in ETL Pipelines",
            "body": """Data quality issues cascade silently into downstream dashboards and ML features.
            We implement automated data contracts in Python using Pydantic V2 and Great Expectations.
            Invalid records are quarantined into dead-letter queues while valid streaming records proceed to analytical warehouse sinks.""",
            "tags": ["Data Engineering", "Python"],
            "date": "2026-08-11T16:20:00"
        },
        # 28. Test Set / Unseen recent post: DevOps & Cloud + Performance Optimization
        {
            "title": "Kubernetes Autoscaling at Scale: KEDA and Custom Metric Optimization",
            "body": """Standard CPU-based Horizontal Pod Autoscalers (HPA) often react too slowly to traffic spikes.
            We install KEDA (Kubernetes Event-driven Autoscaling) to trigger pod scaling directly from RabbitMQ queue depths and Redis stream backlog lengths.
            Cold-start latencies and resource over-provisioning costs are analyzed.""",
            "tags": ["DevOps & Cloud", "Performance Optimization"],
            "date": "2026-09-02T11:10:00"
        },
        # 29. Test Set / Unseen recent post: Natural Language Processing + API Development
        {
            "title": "Building a Production Semantic Search API with Hybrid Dense-Sparse Ranking",
            "body": """Designing an enterprise search API requires combining BM25 keyword matching with dense neural embeddings.
            We build a FastAPI web endpoint in Python that accepts search queries, computes vector embeddings, and executes reciprocal rank fusion (RRF) across Elasticsearch and Qdrant.
            Result relevance and response latencies are measured.""",
            "tags": ["Natural Language Processing", "API Development"],
            "date": "2026-10-01T15:30:00"
        },
        # 30. Test Set / Unseen recent post: Short draft edge case
        {
            "title": "Quick reminder on Docker compose flags",
            "body": "docker compose up -d --build --remove-orphans",
            "tags": ["DevOps & Cloud"],
            "date": "2026-11-05T08:00:00"
        }
    ]

    # Expand corpus programmatically with variations to ensure statistical richness
    records = []
    post_id = 1001

    for base in base_articles:
        records.append({
            "id": f"post-{post_id}",
            "title": base["title"],
            "body": base["body"],
            "tags": base["tags"],
            "published_date": base["date"],
            "author": f"author_{random.randint(1, 8)}"
        })
        post_id += 1

    # Generate additional synthetic variations to reach a solid evaluation size (~60 articles)
    tag_combinations = [
        (["Machine Learning", "Python"], "Supervised Classification with Scikit-Learn: Pipelines and Hyperparameter Tuning", "Building robust machine learning pipelines in Python requires encapsulating feature scaling, categorical encoding, and estimator training into scikit-learn Pipelines. We demonstrate randomized search cross-validation, custom transformers, and metric tracking."),
        (["Deep Learning", "Performance Optimization"], "Distributed Training Strategies: Data Parallel vs Pipeline Parallel in PyTorch", "Training large neural networks exceeds single GPU VRAM. We analyze DistributedDataParallel (DDP), Fully Sharded Data Parallel (FSDP), and tensor parallelism trade-offs across multi-node GPU clusters."),
        (["API Development", "Frontend Development"], "Contract-Driven Development with OpenAPI and TypeScript Generator", "Preventing frontend and backend drift through schema-driven development. We show how backend OpenAPI 3.1 specs automatically generate typed TypeScript API client SDKs with zero manual boilerplate."),
        (["DevOps & Cloud", "Application Security"], "Securing CI/CD Pipelines: Ephemeral Runners and OIDC Federation", "Traditional hardcoded cloud credentials in CI/CD secrets present major security hazards. We implement OpenID Connect (OIDC) federation between GitHub Actions and AWS IAM for credential-less pipeline deployments."),
        (["Database Systems", "Software Architecture"], "Database Sharding and Distributed Transactions with CockroachDB", "Scaling relational databases horizontally requires understanding distributed consensus and Raft consensus engines. We examine CockroachDB architecture, multi-raft ranges, and cross-node transactional isolation."),
        (["Natural Language Processing", "Python"], "Tokenization Strategies: BPE, WordPiece, and SentencePiece in Python", "Understanding subword tokenization is essential for modern NLP. We build Byte-Pair Encoding and WordPiece tokenizers from scratch in Python, analyzing vocabulary size trade-offs and out-of-vocabulary handling."),
        (["Data Engineering", "Performance Optimization"], "Optimizing Apache Spark Shuffle Performance and Memory Management", "Spark shuffle stages frequently cause out-of-memory errors and disk spills. We tune spark.sql.shuffle.partitions, explore adaptive query execution (AQE), and benchmark broadcast hash joins against sorted merge joins."),
        (["Software Architecture", "API Development"], "Event Sourcing and CQRS with Kafka and EventStore", "Separating read and write models with Command Query Responsibility Segregation (CQRS). We trace event immutability, projection building, and eventual consistency challenges in high-volume transaction systems."),
        (["Frontend Development", "Performance Optimization"], "Optimizing Web Fonts and Image Assets for Sub-Second First Contentful Paint", "Modern websites suffer performance degradation from unoptimized media. We examine responsive picture elements, AVIF/WebP image formats, font-display swap, and subsetting custom web fonts."),
        (["Machine Learning", "Application Security"], "Adversarial Machine Learning: Defending Against Data Poisoning and Model Inversion", "Machine learning models in production are susceptible to adversarial evasion attacks and data poisoning. We explore defense mechanisms including input sanitization, differential privacy, and robust adversarial training.")
    ]

    start_date = datetime(2025, 1, 10)
    for i, (tags, title, body_text) in enumerate(tag_combinations):
        pub_date = (start_date + timedelta(days=i * 30 + 12)).isoformat()
        formatted_body = f"""<article>
        <h1>{title}</h1>
        <p>{body_text}</p>
        <p>In this article we step through practical implementation details, code samples, and real-world deployment challenges.</p>
        <footer><p>Tags: {', '.join(tags)}</p><a href='/rss'>RSS Feed</a></footer>
        </article>"""
        records.append({
            "id": f"post-{post_id}",
            "title": title,
            "body": formatted_body,
            "tags": tags,
            "published_date": pub_date,
            "author": f"author_{random.randint(1, 8)}"
        })
        post_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Generated {len(records)} realistic blog articles at {output_path}")
    return records


if __name__ == "__main__":
    generate_synthetic_blog_corpus()
