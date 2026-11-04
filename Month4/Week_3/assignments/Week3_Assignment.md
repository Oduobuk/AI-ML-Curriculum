# Week 3 Production Engineering Assignment: Enterprise Multi-Tenant Vector Database & Core RAG Pipeline

## Executive Overview & Architectural Motivation
Enterprise Retrieval-Augmented Generation (RAG) applications must query millions of contextual knowledge documents while strictly enforcing multi-tenant isolation, department-level role-based access control (RBAC), and low-latency chunk retrieval ($< 15\text{ ms}$). Simple in-memory list traversals cause security data leaks, catastrophic latency spikes, and severe context pollution in downstream LLMs.

In this assignment, you will architect a production-ready **Multi-Tenant Vector Database & Core RAG Pipeline** utilizing **Qdrant** (or in-process Qdrant client), dense sentence embeddings, and recursive character text chunking.

Your pipeline will ingest unstructured enterprise policy documents, apply token-aware recursive chunking with overlap, generate normalized vector embeddings, insert them into a multi-tenant Qdrant collection with payload indexes, and execute hybrid filtered semantic search.

---

## Technical Specifications & Architecture Diagram

```
                 Enterprise Policy PDFs & Technical Documentation
                                        │
                                        ▼
                 ┌───────────────────────────────────────────┐
                 │       Recursive Token-Aware Chunking      │
                 │   - Chunk Size: 512 tokens                │
                 │   - Overlap: 64 tokens                    │
                 │   - Preserves section header metadata     │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │         Dense Embedding Generator         │
                 │   - fastembed / sentence-transformers     │
                 │   - L2 Unit Normalization                 │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │      Multi-Tenant Qdrant Vector Engine    │
                 │   - HNSW Index (M=16, efConstruct=128)    │
                 │   - Strict Tenant Isolation Payload Filter│
                 │   - RBAC Security Clearance Hierarchy     │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 Clean Context Chunks Injected into LLM Prompt
```

---

## Detailed Deliverable Requirements

### 1. Document Chunking & Ingestion Engine (`chunker.py`)
Implement `SemanticDocumentChunker`:
- Recursively splits documents on structural separators (`\n\n`, `\n`, `. `, ` `).
- Enforces maximum chunk length (512 tokens) with 64-token sliding window overlap.
- Attaches structured payload metadata to every chunk: `tenant_id`, `department`, `doc_id`, `access_level`, `created_at`.

### 2. Multi-Tenant Vector Store Pipeline (`qdrant_store.py`)
- Initialize a local Qdrant collection using `qdrant-client` (`:memory:` or local Docker).
- Configure vector distance metric: `Distance.COSINE`.
- Create payload indexes on `tenant_id` (Keyword) and `access_level` (Integer).
- Implement batch asynchronous upserting with retry handling.

### 3. Contextual Query Retrieval & Evaluation (`retriever.py`)
- Implement `retrieve_context(query, tenant_id, user_role, top_k=3)`:
  - Embeds incoming user query into vector space.
  - Builds Qdrant Boolean filter:
    $$\text{Filter} = \text{TenantMatch}(\text{tenant\_id}) \land \text{AccessLevel} \le \text{user\_clearance}$$
  - Returns formatted context strings ready for system prompt injection.
- Assert that search requests across tenants return strictly zero overlapping document IDs.

---

## Starter Code Template & Submission Skeleton

```python
"""
enterprise_rag_qdrant.py - Production Multi-Tenant RAG Pipeline
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

class SimpleRecursiveChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        words = text.split()
        chunks = []
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_str = " ".join(chunk_words)
            chunks.append(DocumentChunk(
                chunk_id=f"{metadata.get('doc_id', 'doc')}_{chunk_idx}",
                text=chunk_str,
                metadata=metadata.copy()
            ))
            chunk_idx += 1
            i += (self.chunk_size - self.overlap)
        return chunks

class MockProductionVectorStore:
    def __init__(self, dim: int = 16):
        self.dim = dim
        self.storage: List[Dict[str, Any]] = []

    def mock_embed(self, text: str) -> np.ndarray:
        # Deterministic mock embedding for testing
        np.random.seed(abs(hash(text)) % (2**31))
        vec = np.random.randn(self.dim)
        return vec / np.linalg.norm(vec)

    def upsert_chunks(self, chunks: List[DocumentChunk]):
        for c in chunks:
            self.storage.append({
                "id": c.chunk_id,
                "text": c.text,
                "vector": self.mock_embed(c.text),
                "payload": c.metadata
            })

    def search(self, query: str, tenant_id: str, max_clearance: int, top_k: int = 3) -> List[Dict[str, Any]]:
        q_vec = self.mock_embed(query)
        
        # 1. Multi-tenant and RBAC pre-filter
        candidates = [
            item for item in self.storage
            if item["payload"].get("tenant_id") == tenant_id
            and item["payload"].get("access_level", 99) <= max_clearance
        ]
        
        # 2. Score similarity
        scored = []
        for c in candidates:
            score = float(np.dot(q_vec, c["vector"]))
            scored.append((score, c))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

# Verification Run
if __name__ == "__main__":
    chunker = SimpleRecursiveChunker(chunk_size=50, overlap=10)
    store = MockProductionVectorStore(dim=8)
    
    # Ingest confidential doc for Tenant A (Level 3 clearance)
    doc_a = "Enterprise financial report. Revenue grew by 45 percent. Internal audit completed."
    chunks_a = chunker.chunk_text(doc_a, {"doc_id": "rep_2026", "tenant_id": "tenant_A", "access_level": 3})
    store.upsert_chunks(chunks_a)
    
    # Ingest public doc for Tenant B
    doc_b = "Public marketing handbook. Brand guidelines and logo placement instructions."
    chunks_b = chunker.chunk_text(doc_b, {"doc_id": "mkt_2026", "tenant_id": "tenant_B", "access_level": 1})
    store.upsert_chunks(chunks_b)
    
    # Query from Tenant A with low clearance (Level 1) -> Should return 0 docs
    low_clearance_res = store.search("Revenue growth", tenant_id="tenant_A", max_clearance=1)
    assert len(low_clearance_res) == 0
    print("Low clearance query correctly returned 0 confidential results.")
    
    # Query from Tenant A with executive clearance (Level 3) -> Returns doc
    high_clearance_res = store.search("Revenue growth", tenant_id="tenant_A", max_clearance=3)
    assert len(high_clearance_res) > 0
    assert high_clearance_res[0]["payload"]["tenant_id"] == "tenant_A"
    print("High clearance query successfully retrieved confidential chunk:", high_clearance_res[0]["text"][:40])
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Recursive Chunking & Token Preservation** | 25 | Proper separator priority, sliding window overlap, metadata propagation. |
| **Multi-Tenant Security Isolation** | 35 | $100\%$ airtight data isolation; cross-tenant searches return zero leaking matches. |
| **RBAC Security Clearance Filtering** | 20 | Robust pre-filtering on integer clearance levels before vector distance sorting. |
| **HNSW / Vector Engine Architecture** | 10 | Proper index configuration, metric selection (Cosine), and payload schemas. |
| **Verification & Test Coverage** | 10 | Fully automated end-to-end verification script with assertions. |
