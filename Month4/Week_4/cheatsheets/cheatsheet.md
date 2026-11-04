# Month 4, Week 4 Cheatsheet: Advanced RAG, Hybrid Search & Guardrail Engineering

---

## 1. Reciprocal Rank Fusion (RRF) Architecture

Merges disparate ranked lists without requiring raw score normalization:
$$\text{RRF\_Score}(d) = \sum_{m \in \mathcal{M}} \frac{1}{k + r_m(d)}$$
- $\mathcal{M} = \{\text{Sparse BM25}, \text{Dense Vector}\}$.
- $r_m(d)$: Rank position of document $d$ in retrieval system $m$ ($1, 2, \dots, N$).
- $k$: Smoothing constant (industry standard: $k=60$).

```python
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 2. Bi-Encoder vs Cross-Encoder Re-Ranking

```
                     Two-Stage Hybrid RAG Architecture
  Query ──► [Stage 1: Fast Bi-Encoder + BM25] ──► Top 50 Candidates (Latency < 10ms)
                                                        │
                                                        ▼
            [Stage 2: Cross-Encoder Transformer] ──► Top 3 Reranked Context (Latency ~ 30ms)
```

- **Bi-Encoder**: Encodes query and documents independently ($e_q, e_d$). Computes cosine similarity via dot product. Fast, indexable, but lacks cross-attention token interaction.
- **Cross-Encoder**: Jointly feeds $[CLS] \circ \text{query} \circ [SEP] \circ \text{document}$ through all Transformer layers. Highly accurate; eliminates irrelevant chunks.

---

## 3. Ragas RAG Evaluation Triad

The standard metric framework for assessing production RAG pipelines:

### 1. Faithfulness (Hallucination Detector)
$$\text{Faithfulness} = \frac{|\text{Factual Claims in Answer Entailed by Context}|}{|\text{Total Claims Extracted from Answer}|}$$
- Target: $> 0.90$. Measures if the LLM grounded its answer strictly in the retrieved text.

### 2. Answer Relevance
$$\text{Answer Relevance} = \frac{1}{N} \sum_{i=1}^N \cos(e(q), e(q_i^{\text{generated}}))$$
- Measures whether the answer directly addresses the user's intent.

### 3. Context Recall
$$\text{Context Recall} = \frac{|\text{Ground-Truth Sentences Attributable to Retrieved Context}|}{|\text{Total Ground-Truth Sentences}|}$$
- Measures whether the retrieval stage fetched all necessary information.

---

## 4. Deterministic Guardrails & Prompt Shielding

Protects against prompt injection and malicious jailbreaks:
- **Input Sanitization**: Scans for system prompt leaks (`"Ignore previous instructions"`).
- **Output Guardrails**: Regex and classifier filters blocking toxic outputs and PII (SSNs, credit card numbers).
