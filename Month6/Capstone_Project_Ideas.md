# Month 6: Capstone Project Ideas & Specification Blueprints (2026 Edition)

```
================================================================================
COURSE:    AI & LLM SYSTEMS ENGINEERING (2026 EDITION)
MODULE:    MONTH 6: CAPSTONE ENGINEERING STUDIO
FOCUS:     PRODUCTION SPECIFICATION BLUEPRINTS & ARCHITECTURAL IDEAS
TARGET:    SENIOR AI SYSTEMS ARCHITECTS & RESEARCH ENGINEERS
================================================================================
```

---

## Blueprint 1: Autonomous SRE Incident Remediation Multi-Agent System (Track 1)

### Problem Statement
Cloud infrastructure outages cost enterprise organizations an average of $\$9,000$ per minute. Human on-call engineers spend the majority of their response time manually querying dashboards, grepping logs, and tracing stack traces before identifying root causes.

### System Architecture
An autonomous, self-correcting multi-agent team coordinated by **LangGraph** and integrated with external infrastructure via the **Model Context Protocol (MCP)**:
1. **Supervisor Agent**: Ingests PagerDuty alert webhooks, extracts affected microservices, and dispatches sub-tasks.
2. **Telemetry Agent (MCP)**: Queries Prometheus / Datadog endpoints using MCP tools to detect metric anomalies (e.g., CPU, memory, error rates).
3. **Log Diagnostic Agent (MCP)**: Queries OpenSearch / Elasticsearch clusters to isolate exact stack traces and exception patterns.
4. **GitOps Remediation Agent**: Inspects recent Git commits and drafts a pull request or Kubernetes rollback manifest to resolve the incident.
5. **Human-in-the-Loop Gateway**: Pauses execution via a LangGraph checkpointer, sending an interactive Slack notification for human on-call sign-off before applying destructive changes.

### Key Technical Challenges
- Preventing infinite agent loops when diagnostic logs are ambiguous.
- Enforcing strict timeout budgets and deterministic rollback safety.
- Exposing clean MCP server interfaces for Kubernetes (`kubectl`) and Prometheus.

---

## Blueprint 2: Enterprise SEC Financial Hybrid RAG Intelligence (Track 2)

### Problem Statement
Financial analysts and legal auditors spend weeks manually auditing 10-K and 10-Q corporate filings. Dense semantic search fails to retrieve exact numerical figures, footnote statutes, and fiscal balance sheet tables, leading to dangerous financial hallucinations.

### System Architecture
A production-grade, multi-stage hybrid RAG platform built on **Qdrant** and **Cross-Encoder Re-Ranking**:
1. **Semantic Dynamic Chunking**: Parses tabular and textual SEC filings, segmenting documents along financial section cliffs rather than arbitrary token boundaries.
2. **Hybrid Inverted & Dense Storage**: Combines **BM25** (for exact company tickers, fiscal quarters, and regulation numbers) with **Qdrant HNSW** dense embeddings.
3. **Reciprocal Rank Fusion (RRF)**: Merges lexical and dense candidate pools ($k=60$) into a unified top-25 candidate list.
4. **Cross-Encoder Re-Ranker**: Evaluates candidate passages with `bge-reranker-large`, selecting the Top-5 most relevant chunks.
5. **Automated Ragas Triad Evaluation**: Continuously calculates Faithfulness and Context Precision, rejecting any generated response where Faithfulness $<0.95$.

### Key Technical Challenges
- Preserving tabular row-column relationships in Markdown/JSON embeddings.
- Ensuring $<100\text{ ms}$ retrieval latency across a 10-million-document Qdrant index using 8-bit Scalar Quantization (SQ8).
- Hard multi-tenant isolation preventing cross-corporate data leakage.

---

## Blueprint 3: High-Throughput Domain-Adapted SQL Compiler (Track 3)

### Problem Statement
General-purpose foundation models generate invalid SQL queries when confronted with massive enterprise databases featuring hundreds of tables, non-standard dialect syntax (Oracle, Snowflake), and proprietary business logic.

### System Architecture
A fine-tuned, quantized foundation model serving real-time SQL translations at high concurrency:
1. **Domain Adaptation with QLoRA**: Fine-tunes **Llama-3-8B-Instruct** using 4-bit NormalFloat (NF4) and LoRA ($r=16, \alpha=32$) on a curated dataset of complex enterprise schema queries.
2. **Offline Weight Merging**: Merges adapter weights into the base foundation model, producing a frozen, zero-latency artifact.
3. **High-Throughput vLLM Serving**: Deploys the model with **PagedAttention** and Continuous Dynamic Batching, achieving $>80\text{ tokens/sec}$ per stream.
4. **Asynchronous FastAPI Gateway**: Exposes an SSE streaming endpoint with token-bucket rate limiting and Pydantic v2 schema validation.
5. **Deterministic SQL Guardrails**: Validates generated SQL against AST parsers (`sqlglot`), rejecting any destructive statements (`DROP`, `DELETE`) before execution.

### Key Technical Challenges
- Fine-tuning without catastrophic forgetting of general natural language understanding.
- Eliminating schema hallucinations by grounding prompts with dynamic metadata catalogs.
- Managing GPU VRAM and preventing Out-of-Memory crashes under high-load stress testing.

---

## Blueprint 4: Ultra-Low-Latency Edge Vision Inspection Engine (Track 4)

### Problem Statement
High-speed automated manufacturing lines inspect circuit boards and silicon wafers moving at multiple meters per second. Cloud-based computer vision APIs cannot satisfy the required sub-10ms response time.

### System Architecture
An optimized real-time computer vision perception pipeline deployed on embedded edge accelerators:
1. **Modern Backbone**: Custom anchor-free detector head based on **ConvNeXt** or **YOLOv8** decoupled architectures.
2. **Complete IoU (CIoU) Optimization**: Trains regression heads with CIoU loss, eliminating scale and aspect ratio distortion.
3. **Hardware Acceleration**: Exports the trained PyTorch graph to **ONNX**, compiling into a fully fused **NVIDIA TensorRT INT8** engine using entropy calibration.
4. **Vectorized Non-Maximum Suppression**: Runs NMS directly on edge GPU tensors, eliminating CPU-GPU memory roundtrips.
5. **Zero-Copy DMA Memory Pipeline**: Ingests camera streams directly into unified device memory, achieving P99 latency $<4\text{ ms}$ at $120\text{ FPS}$.

### Key Technical Challenges
- Maintaining detection accuracy after INT8 post-training quantization.
- Aspect-ratio preserving letterboxing with zero memory reallocation.
- Operating within tight edge thermal envelopes ($<25\text{ Watts}$).

