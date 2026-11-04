# Week 4 Production Engineering Assignment: Enterprise Hybrid RAG, Cross-Encoder Re-Ranking & Ragas Evaluation Suite

## Executive Overview & Architectural Motivation
Naive RAG systems suffer from two fatal production weaknesses: keyword blind spots (where dense vectors miss exact serial codes, error codes, and technical identifiers) and silent LLM hallucinations (where the generation model invents facts unsupported by the retrieved documents).

In this capstone assignment for Month 4, you will architect an **Enterprise-Grade Advanced RAG System** combining:
1. **Hybrid Retrieval**: BM25 sparse keyword search + Dense semantic embeddings fused via **Reciprocal Rank Fusion (RRF)**.
2. **Cross-Encoder Neural Re-Ranking**: Fine-grained contextual re-scoring to eliminate irrelevant chunks.
3. **Deterministic Guardrails Layer**: Sanitizes prompts against prompt-injection attacks and filters toxic tokens.
4. **Automated Ragas Evaluation Harness**: Continuous calculation of Faithfulness, Answer Relevance, and Context Recall metrics.

---

## Technical Specifications & Architecture Diagram

```
                 Incoming User Query ("How to configure TLS certs on Port 8443?")
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          Deterministic Guardrail Interceptor          │
                 │   - Prompt Injection Regex & Perplexity Classifier    │
                 │   - PII Redaction & System Prompt Shield              │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │           Parallel Hybrid Retrieval Engine            │
                 │   ├─ Sparse Lexical Path: BM25 Okapi (Top-25)         │
                 │   └─ Dense Vector Path: Cosine Similarity (Top-25)    │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │         Reciprocal Rank Fusion (RRF, k=60)            │
                 │   - Fused Candidate Set (Top-20 Chunks)               │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │        Cross-Encoder Neural Re-Ranker (Top-3)         │
                 │   - Token-level cross-attention relevance scoring     │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │           LLM Generation & Ragas Evaluation           │
                 │   - Prompt Injection into Frozen Context Window       │
                 │   - Automated Claim Faithfulness Scoring (> 0.85)     │
                 └───────────────────────────────────────────────────────┘
```

---

## Detailed Deliverable Requirements

### 1. Hybrid Search & Reciprocal Rank Fusion (`hybrid_search.py`)
Implement `HybridSearchEngine`:
- Sparse index: BM25 Okapi tokenizer and inverted index.
- Dense index: Cosine similarity over normalized sentence embeddings.
- Reciprocal Rank Fusion: combines the two ranked candidate sets using $k=60$.
- Returns fused candidates with normalized relevance ranks.

### 2. Cross-Encoder Re-Ranking Pipeline (`reranker.py`)
- Implement `CrossEncoderReRanker`:
  - Accepts query $q$ and list of top candidate texts $c_1, \dots, c_K$.
  - Computes joint cross-attention relevance scores: $s_i = \text{Model}(q, c_i)$.
  - Truncates to the Top-$N$ (e.g. $N=3$) highest-scoring chunks, pruning out low-signal context.

### 3. Automated Ragas Faithfulness Evaluation Suite (`evaluator.py`)
- Implement `RagasEvaluator`:
  - Extracts atomic factual propositions from the generated answer.
  - Verifies factual entailment against the retrieved context passages.
  - Computes:
    $$\text{Faithfulness} = \frac{|\text{Entailed Claims}|}{|\text{Total Claims}|}$$
  - Flags and blocks responses falling below the safety threshold ($< 0.85$).

---

## Starter Code Template & Submission Skeleton

```python
"""
advanced_hybrid_rag_pipeline.py - Production Advanced RAG Suite
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class RetrievedChunk:
    doc_id: str
    text: str
    score: float

class ProductionHybridRAG:
    def __init__(self, corpus: Dict[str, str], dim: int = 16):
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.dim = dim
        
        # Simulated dense embeddings
        np.random.seed(42)
        self.dense_embeddings = np.random.randn(len(self.doc_ids), dim)
        self.dense_embeddings /= np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)

    def bm25_lexical_search(self, query: str, top_k: int = 5) -> List[str]:
        # Simple BM25 keyword matching frequency simulation
        query_terms = set(query.lower().split())
        scored = []
        for doc_id, text in self.corpus.items():
            words = text.lower().split()
            overlap = sum(1 for w in words if w in query_terms)
            scored.append((overlap, doc_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in scored[:top_k]]

    def dense_semantic_search(self, query_vec: np.ndarray, top_k: int = 5) -> List[str]:
        scores = self.dense_embeddings @ query_vec
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [self.doc_ids[idx] for idx in best_indices]

    def reciprocal_rank_fusion(self, list1: List[str], list2: List[str], k: int = 60) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for ranked_list in [list1, list2]:
            for rank, doc_id in enumerate(ranked_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        sorted_ranks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_ranks

    def cross_encoder_rerank(self, query: str, candidate_ids: List[str], top_n: int = 2) -> List[RetrievedChunk]:
        # Cross-encoder joint scoring simulation
        scored = []
        for doc_id in candidate_ids:
            text = self.corpus[doc_id]
            # Higher score if exact keywords match + length bonus
            score = 0.5 * len(set(query.lower().split()).intersection(text.lower().split())) + 0.1
            scored.append(RetrievedChunk(doc_id=doc_id, text=text, score=score))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]

# Verification Run
if __name__ == "__main__":
    docs = {
        "doc_tls": "To configure TLS certificates, mount the secret into /etc/ssl/certs and listen on port 8443.",
        "doc_auth": "Authentication uses OAuth2 Bearer tokens issued by the internal identity provider.",
        "doc_rate": "Rate limits are strictly enforced at 100 requests per minute using Redis token buckets."
    }
    
    rag = ProductionHybridRAG(corpus=docs, dim=16)
    q_text = "configure TLS certificates 8443"
    q_vec = np.random.randn(16)
    q_vec /= np.linalg.norm(q_vec)
    
    # 1. Sparse + Dense Search
    bm25_res = rag.bm25_lexical_search(q_text, top_k=3)
    dense_res = rag.dense_semantic_search(q_vec, top_k=3)
    
    # 2. RRF Fusion
    fused = rag.reciprocal_rank_fusion(bm25_res, dense_res, k=60)
    top_candidates = [doc_id for doc_id, _ in fused]
    
    # 3. Cross-Encoder Re-rank
    final_chunks = rag.cross_encoder_rerank(q_text, top_candidates, top_n=1)
    
    print("Top Re-Ranked Document ID:", final_chunks[0].doc_id)
    print("Content:", final_chunks[0].text)
    assert final_chunks[0].doc_id == "doc_tls"
    print("Advanced RAG Pipeline successfully identified and re-ranked the exact technical document!")
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **BM25 & Dense Hybrid Search** | 30 | Correct sparse inverted index, dense dot-product evaluation, and reciprocal rank fusion. |
| **Cross-Encoder Re-Ranking** | 25 | Joint query-document token scoring; accurate pruning of irrelevant candidates. |
| **Ragas Faithfulness Evaluation** | 25 | Atomic claim decomposition, entailment verification, and strict hallucination alerts. |
| **Guardrails & Prompt Defense** | 10 | Robust regex and semantic sanitization against prompt injection attacks. |
| **Verification & Latency Benchmarks** | 10 | Fully automated end-to-end execution script with deterministic assertions. |
