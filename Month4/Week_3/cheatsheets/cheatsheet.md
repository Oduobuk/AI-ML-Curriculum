# Month 4, Week 3 Cheatsheet: Vector Databases, HNSW Indexing & Core RAG

---

## 1. Vector Distance & Similarity Metrics

For two embedding vectors $u, v \in \mathbb{R}^d$:

| Metric | Mathematical Formula | Range | Normalization Invariant |
|:---|:---|:---|:---|
| **Cosine Similarity** | $\frac{u \cdot v}{\|u\|_2 \|v\|_2}$ | $[-1.0, 1.0]$ | Invariant to vector magnitude |
| **Dot Product** | $\sum_{i=1}^d u_i v_i$ | $(-\infty, \infty)$ | Equals Cosine Similarity if $\|u\|=\|v\|=1$ |
| **Euclidean Distance ($L_2$)** | $\sqrt{\sum_{i=1}^d (u_i - v_i)^2}$ | $[0, \infty)$ | Monotonically related to Dot Product for unit vectors |

---

## 2. Hierarchical Navigable Small World (HNSW) Indexing

Constructs a multi-layer geometric graph enabling sub-linear $O(\log N)$ approximate nearest neighbor search:

```
                    HNSW Multi-Layer Skip-Graph Hierarchy
 Layer 2:     (node_1) ─────────────────────────────► (node_8)   [Long-distance skips]
                 │                                       │
 Layer 1:     (node_1) ──────────► (node_4) ────────► (node_8)   [Medium clusters]
                 │                    │                  │
 Layer 0:     (node_1) ──► (node_2)──►(node_4)──►(node_6)►(node_8)[Dense ground layer]
```

### Key HNSW Hyperparameters
- **$M$**: Number of bidirectional edges per node (typically $16 - 64$). Higher $M$ increases recall and index size.
- **`efConstruction`**: Size of dynamic candidate list during graph construction (typically $100 - 400$).
- **`efSearch`**: Size of candidate queue during runtime query execution (balances latency vs recall).

---

## 3. Product Quantization (PQ) & Asymmetric Distance (ADC)

Decomposes vectors into $M$ orthogonal subspaces, quantizing each to $K=256$ centroids:
$$d_{\text{ADC}}(q, x) = \sqrt{\sum_{m=1}^M \|q_m - c_{m, k_m}\|_2^2}$$
- Compresses 1536-dimensional FP32 vectors ($6,144$ bytes) into 64 bytes ($96\times$ compression).
- Precomputes an $M \times 256$ distance lookup table from query $q$ to centroids; distance lookups require zero floating-point operations.

---

## 4. Multi-Tenant Payload Filtering Architecture (Qdrant)

Always execute vector search with strict boolean pre-filtering to prevent cross-tenant data leaks:
```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333")

results = client.search(
    collection_name="enterprise_knowledge",
    query_vector=query_embedding,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value="client_alpha")),
            models.FieldCondition(key="department", match=models.MatchValue(value="FINANCE")),
            models.FieldCondition(key="access_level", range=models.Range(lte=3))
        ]
    ),
    limit=5
)
```
