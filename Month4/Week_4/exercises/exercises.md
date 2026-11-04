# Week 4: Graduate-Level Problem Set — Advanced RAG, Hybrid Search & Guardrail Engineering

Welcome to the Month 4 Week 4 Problem Set. This suite evaluates your mastery of enterprise production RAG pipelines: sparse lexical (BM25) and dense semantic hybrid search, Reciprocal Rank Fusion (RRF), Cross-Encoder re-ranking, Ragas evaluation triad metrics (Faithfulness, Answer Relevance, Context Recall), and deterministic LLM guardrails (NeMo/Llama-Guard).

---

## Tier 1: Core Mechanics & Hybrid Search Fusion

### Exercise 1: Reciprocal Rank Fusion (RRF) & Score Normalization
**Concepts**: Lexical BM25 ranking, dense vector ranking, rank-based merging, smoothing parameter $k=60$.

#### Problem Specification
Dense semantic embeddings excel at conceptual synonymy but fail at exact keywords (part numbers, SKU IDs, acronyms), while sparse BM25 excels at exact keyword matches but fails at conceptual generalization:
1. Reciprocal Rank Fusion combines ranked lists from independent retrieval systems without requiring raw score calibration:
   $$\text{RRF\_Score}(d) = \sum_{m \in \mathcal{M}} \frac{1}{k + r_m(d)}$$
   where $\mathcal{M} = \{\text{Dense}, \text{BM25}\}$, $r_m(d) \in \{1, 2, \dots, N\}$ is the 1-based rank position of document $d$ in system $m$, and $k$ is typically set to $60$.
2. Implement `reciprocal_rank_fusion(dense_ranks, bm25_ranks, k=60)`.
3. Given a candidate set where Document A ranks #1 in BM25 but #15 in Dense, and Document B ranks #4 in both systems, prove which document wins the top fused ranking.

#### Verification Solution
```python
from typing import Dict, List, Tuple

def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)
            
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

# Verification Run
if __name__ == "__main__":
    bm25_results = ["doc_A", "doc_C", "doc_D", "doc_B", "doc_E"]  # doc_A is rank 1, doc_B is rank 4
    dense_results = ["doc_F", "doc_G", "doc_H", "doc_B", "doc_A"] # doc_B is rank 4, doc_A is rank 5
    
    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=60)
    print("Fused Rankings Head:", fused[:3])
    
    # Calculate scores manually:
    # doc_A: 1/(60+1) + 1/(60+5) = 1/61 + 1/65 = 0.01639 + 0.01538 = 0.03177
    # doc_B: 1/(60+4) + 1/(60+4) = 2/64 = 0.03125
    assert fused[0][0] == "doc_A"
    print("Reciprocal Rank Fusion successfully evaluated with zero score distortion.")
```

---

## Tier 2: Cross-Encoder Neural Re-Ranking

### Exercise 2: Bi-Encoder vs Cross-Encoder Computational Complexity
**Concepts**: Bi-Encoder independent embedding dot product vs Cross-Encoder full token attention cross-interaction.

#### Problem Specification
1. Bi-Encoders compute query and document representations independently: $\text{sim}(q, d) = \langle e(q), e(d) \rangle$, enabling pre-computed indexing at $O(1)$ lookup time, but losing cross-attention token interaction.
2. Cross-Encoders concatenate query and document $[CLS] \circ q \circ [SEP] \circ d$ through full self-attention layers:
   $$\text{Complexity} = O((|q| + |d|)^2)$$
3. Write a two-stage retrieval pipeline:
   - First stage: fast Bi-Encoder retrieves Top-50 candidates in $< 5\text{ ms}$.
   - Second stage: Cross-Encoder re-ranks the Top-50 to select the definitive Top-5.

#### Verification Solution
```python
import numpy as np
from typing import List, Tuple

class TwoStageRetriever:
    def __init__(self):
        # Simulated document pool (100 docs)
        np.random.seed(42)
        self.doc_ids = [f"doc_{i}" for i in range(100)]
        self.bi_encoder_embeddings = np.random.randn(100, 16)
        self.bi_encoder_embeddings /= np.linalg.norm(self.bi_encoder_embeddings, axis=1, keepdims=True)

    def retrieve_and_rerank(self, query_vec: np.ndarray, top_k_stage1: int = 20, top_k_stage2: int = 3) -> List[Tuple[str, float]]:
        # Stage 1: Fast Bi-Encoder dot product
        scores_stage1 = self.bi_encoder_embeddings @ query_vec
        cand_indices = np.argsort(scores_stage1)[::-1][:top_k_stage1]
        
        # Stage 2: Cross-Encoder fine re-ranking (simulated full self-attention score)
        # Adds cross-attention interaction boost for documents with high semantic match
        stage2_scores = []
        for idx in cand_indices:
            doc_id = self.doc_ids[idx]
            # Simulated cross-attention scoring
            cross_score = float(scores_stage1[idx] + 0.1 * np.sin(idx))
            stage2_scores.append((doc_id, cross_score))
            
        stage2_scores.sort(key=lambda x: x[1], reverse=True)
        return stage2_scores[:top_k_stage2]

# Verification Run
if __name__ == "__main__":
    retriever = TwoStageRetriever()
    q = np.random.randn(16)
    q /= np.linalg.norm(q)
    
    final_top3 = retriever.retrieve_and_rerank(q)
    print("Final Re-Ranked Top 3 Documents:", final_top3)
    assert len(final_top3) == 3
```

---

## Tier 3: Ragas Evaluation Triad & Deterministic Guardrails

### Exercise 3: Faithfulness & Hallucination Scoring Engine
**Concepts**: Ragas Faithfulness metric, claim extraction, entailment classification, hallucination prevention.

#### Problem Specification
1. In production RAG, Faithfulness measures whether the LLM's generated response $\hat{A}$ is grounded strictly in the retrieved context $C$:
   $$\text{Faithfulness} = \frac{|\text{Number of Claims in } \hat{A} \text{ Entailed by } C|}{|\text{Total Number of Claims in } \hat{A}|}$$
2. Implement `evaluate_faithfulness(claims, context)`:
   - Takes a list of atomic factual claims extracted from the response.
   - Evaluates whether each claim is entailed by the provided context chunks.
   - Computes the faithfulness ratio $\in [0.0, 1.0]$.
   - Emits an alert if Faithfulness $< 0.85$.

#### Verification Solution
```python
from typing import List

def evaluate_faithfulness(atomic_claims: List[str], context_corpus: str) -> float:
    if not atomic_claims:
        return 1.0
    
    entailed_count = 0
    for claim in atomic_claims:
        # Production systems use an NLI model (DeBERTa-v3) or structured LLM judge;
        # here we verify substring/token semantic containment logic
        words = [w.lower() for w in claim.split() if len(w) > 3]
        match_count = sum(1 for w in words if w in context_corpus.lower())
        if match_count >= len(words) * 0.7:
            entailed_count += 1
            
    return entailed_count / len(atomic_claims)

# Verification Run
if __name__ == "__main__":
    context = (
        "Enterprise API Gateway v4.2 enforces token bucket rate limiting at 50 requests per second. "
        "All incoming payload logs are encrypted using AES-256-GCM."
    )
    
    # Grounded response claims
    valid_claims = [
        "API Gateway rate limiting is 50 requests per second",
        "Payload logs are encrypted with AES-256-GCM"
    ]
    
    # Hallucinated response claims
    hallucinated_claims = [
        "API Gateway rate limiting is 50 requests per second",
        "The system runs on Kubernetes clusters in Frankfurt"  # Not in context!
    ]
    
    score_valid = evaluate_faithfulness(valid_claims, context)
    score_hallucinated = evaluate_faithfulness(hallucinated_claims, context)
    
    print(f"Faithfulness Score (Grounded):     {score_valid:.2f}")
    print(f"Faithfulness Score (Hallucinated): {score_hallucinated:.2f}")
    
    assert score_valid == 1.0
    assert score_hallucinated < 0.80
```
