# Week 3: Graduate-Level Problem Set — Vector Databases, Hierarchical Navigable Small World (HNSW) & Core RAG

Welcome to the Month 4 Week 3 Problem Set. This suite evaluates your mastery of enterprise retrieval-augmented generation: dense vector similarity geometries, Hierarchical Navigable Small World (HNSW) graph indexing, Inverted File Product Quantization (IVF-PQ), multi-tenant Qdrant payloads, and contextual query routing.

---

## Tier 1: Core Mechanics & Approximate Nearest Neighbor Graphs

### Exercise 1: Hierarchical Navigable Small World (HNSW) Multi-Layer Search
**Concepts**: Small world graphs, skip-list hierarchy, greedy routing, probability decay factor $m_L = \frac{1}{\ln(M)}$.

#### Problem Specification
In large vector databases (Qdrant, Milvus), exact exhaustive search is $O(N \cdot d)$. HNSW constructs a multi-layer graph with exponentially decaying density:
1. The maximum level $l$ for an inserted vector is sampled from:
   $$l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor$$
2. On layer $l > 0$, search routes greedily to the single nearest neighbor to transition rapidly across global distances.
3. On layer 0, beam search explores $efSearch$ candidates to maximize recall.
4. Implement a 2-layer HNSW graph in Python for 500 vectors in $\mathbb{R}^{16}$.
5. Benchmark recall@10 against exhaustive flat $L_2$ search.

#### Verification Solution
```python
import numpy as np

class ToyHNSWLayer:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.nodes: dict[int, np.ndarray] = {}
        self.edges: dict[int, list[int]] = {}

    def add_node(self, node_id: int, vec: np.ndarray, neighbors: list[int]):
        self.nodes[node_id] = vec
        self.edges[node_id] = neighbors
        for n in neighbors:
            if n in self.edges and node_id not in self.edges[n]:
                self.edges[n].append(node_id)

    def greedy_search(self, query: np.ndarray, entry_point: int) -> int:
        curr = entry_point
        curr_dist = float(np.linalg.norm(self.nodes[curr] - query))
        
        while True:
            best_neighbor = curr
            best_dist = curr_dist
            for neighbor in self.edges.get(curr, []):
                dist = float(np.linalg.norm(self.nodes[neighbor] - query))
                if dist < best_dist:
                    best_dist = dist
                    best_neighbor = neighbor
            if best_neighbor == curr:
                break
            curr = best_neighbor
            curr_dist = best_dist
            
        return curr

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    layer = ToyHNSWLayer(layer_id=0)
    
    # Insert 5 points in a simple line graph
    pts = [np.array([float(i)]) for i in range(5)]
    for i in range(5):
        neighbors = [i - 1] if i > 0 else []
        if i < 4:
            neighbors.append(i + 1)
        layer.add_node(i, pts[i], neighbors)
        
    query = np.array([3.2])
    res = layer.greedy_search(query, entry_point=0)
    print("Nearest Node found from entry 0:", res)
    assert res == 3
```

---

## Tier 2: Product Quantization (PQ) Compression

### Exercise 2: Product Quantization (PQ) Codebook & Distance Table Lookups
**Concepts**: Subspace decomposition, $K$-means codebook centroids, asymmetric distance computation (ADC).

#### Problem Specification
To store 100 million 1536-dimensional embeddings in RAM, Product Quantization decomposes vectors into $M$ orthogonal subspaces:
1. Split vector $x \in \mathbb{R}^D$ into $M$ subvectors of dimension $D/M$.
2. Quantize each subspace into $K=256$ centroids via $K$-means, encoding each vector as an $M$-byte array.
3. Compute distances via Asymmetric Distance Computation (ADC): query vector $q$ remains unquantized; precompute an $M \times 256$ lookup table of distances to centroids:
   $$d_{\text{ADC}}(q, x) = \sqrt{\sum_{m=1}^M \|q_m - c_{m, k_m}\|_2^2}$$
4. Implement `compute_adc_table(query, codebooks)` and verify that distance lookups require zero floating-point multiplications per stored vector.

#### Verification Solution
```python
import numpy as np

def compute_adc_table(query: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """
    query: [D]
    codebooks: [M, K, D/M] where K=256
    Returns: [M, K] precomputed squared distance table
    """
    M, K, sub_dim = codebooks.shape
    q_sub = query.reshape(M, sub_dim)  # [M, sub_dim]
    
    # Table: [M, K]
    table = np.sum((codebooks - q_sub[:, None, :]) ** 2, axis=-1)
    return table

def adc_lookup(table: np.ndarray, pq_codes: np.ndarray) -> np.ndarray:
    """
    table: [M, K]
    pq_codes: [N, M] where each entry is an integer code in [0, K-1]
    """
    N, M = pq_codes.shape
    distances = np.zeros(N)
    for m in range(M):
        distances += table[m, pq_codes[:, m]]
    return np.sqrt(distances)

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    D, M, K = 32, 4, 16
    query = np.random.randn(D)
    codebooks = np.random.randn(M, K, D // M)
    
    # 5 sample vectors encoded as PQ byte codes
    pq_codes = np.random.randint(0, K, size=(5, M))
    
    table = compute_adc_table(query, codebooks)
    distances = adc_lookup(table, pq_codes)
    print("ADC Computed Distances for 5 vectors:", np.round(distances, 4))
    assert len(distances) == 5
    assert not np.isnan(distances).any()
```

---

## Tier 3: Asynchronous Multi-Tenant Qdrant Retrieval Engine

### Exercise 3: Filtered Vector Search with Strict Multi-Tenant Isolation
**Concepts**: Pre-filtering vs post-filtering, payload index matching, tenant key partitioning.

#### Problem Specification
1. In enterprise RAG, tenant data cross-contamination is a critical security vulnerability.
2. In Qdrant, configure a collection with a `tenant_id` payload index.
3. Write an asynchronous search query executing vector similarity ranking strictly constrained by:
   ```json
   {
     "must": [
       {"key": "tenant_id", "match": {"value": "client_alpha"}},
       {"key": "department", "match": {"value": "FINANCE"}},
       {"key": "access_level", "range": {"lte": 3}}
     ]
   }
   ```
4. Verify that vectors from `client_beta` are never retrieved, even if their cosine similarity is $1.0$.

#### Verification Solution
```python
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    payload: Dict[str, Any]

class MockQdrantCollection:
    def __init__(self):
        self.records: List[VectorRecord] = []

    def insert(self, record: VectorRecord):
        self.records.append(record)

    def search(self, query_vector: np.ndarray, tenant_id: str, dept: str, max_access: int, top_k: int = 3) -> List[VectorRecord]:
        # Pre-filtering phase
        candidates = [
            r for r in self.records 
            if r.payload.get("tenant_id") == tenant_id
            and r.payload.get("department") == dept
            and r.payload.get("access_level", 99) <= max_access
        ]
        
        if not candidates:
            return []
            
        # Cosine similarity
        scores = []
        for c in candidates:
            cos = float(np.dot(c.vector, query_vector) / (np.linalg.norm(c.vector) * np.linalg.norm(query_vector)))
            scores.append((cos, c))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    q_coll = MockQdrantCollection()
    
    # Insert target vector under client_beta (identical to query)
    vec = np.ones(8)
    q_coll.insert(VectorRecord("doc_leak", vec, {"tenant_id": "client_beta", "department": "FINANCE", "access_level": 1}))
    
    # Insert valid vector under client_alpha
    q_coll.insert(VectorRecord("doc_safe", np.array([1, 0, 1, 0, 1, 0, 1, 0]), {"tenant_id": "client_alpha", "department": "FINANCE", "access_level": 2}))
    
    results = q_coll.search(vec, tenant_id="client_alpha", dept="FINANCE", max_access=3, top_k=5)
    print("Retrieved Records for client_alpha:", [r.id for r in results])
    assert "doc_leak" not in [r.id for r in results]
    assert [r.id for r in results] == ["doc_safe"]
    print("Multi-tenant pre-filter completely isolated data!")
```
