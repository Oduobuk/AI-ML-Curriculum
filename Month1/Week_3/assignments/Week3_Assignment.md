# Week 3 Production Engineering Assignment: High-Throughput Async Ingestion & Scraping Engine

## Executive Overview & Architectural Motivation
Raw training corpora for LLMs, embedding generators, and multi-modal models depend on automated web harvesting, REST API scraping, and continuous ingestion pipelines. However, real-world networks exhibit packet loss, dynamic IP rate-limiting, Cloudflare bot-challenges, and schema drift.

In this assignment, you will architect a **High-Throughput Asynchronous Web Scraping & Ingestion Engine** capable of ingesting semi-structured web documentation, extracting structured entity attributes, applying token-bucket rate limiting, managing a persistent dead-letter queue, and storing sanitized records into partitioned Parquet sinks.

---

## Technical Specifications & Architecture Diagram

```
                 Target Web Pages / REST API Endpoints
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │    Token Bucket Rate Gatekeeper   │
                 │   (Strict RPS + Jitter Protection)│
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │   Async Worker Pool (HTTP/2-ready)│
                 │   - Exponential Retry + Jitter    │
                 │   - Dead-Letter Queue Isolation   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │      DOM / Schema Extraction      │
                 │   - XPath & CSS Parsing (lxml)    │
                 │   - Pydantic Schema Validation    │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                 Sanitized Parquet Datasets for AI Training
```

---

## Detailed Deliverable Requirements

### 1. Robust Async HTTP Worker Pool (`crawler.py`)
- Configure an asynchronous crawler using `httpx` or `aiohttp`.
- Enforce strict concurrency limits with `asyncio.Semaphore`.
- Implement client-side token bucket rate limiting (configurable requests-per-second).
- Handle HTTP status codes: retry on `429`, `500`, `502`, `503`, `504`; fail fast on `400`, `401`, `403`, `404`.

### 2. High-Speed HTML Parser & Pydantic Validator (`extractor.py`)
- Extract structured records using `lxml.html` or `BeautifulSoup4`.
- Validate extracted data against a strict Pydantic model (`InferenceItem`):
  - Validate non-empty titles.
  - Sanitize and normalize currency / float metrics.
  - Verify timestamp parsing into UTC ISO-8601 strings.

### 3. Fault Tolerance & Dead-Letter Queue (DLQ) (`storage.py`)
- Unrecoverable failed requests must be written to `failed_records.jsonl` along with full error context, retry attempts, and stack traces.
- Successfully parsed records must be written directly to a partitioned Parquet table via Polars.

---

## Starter Code Template & Submission Skeleton

```python
"""
production_crawler.py - Production Async Data Scraping Pipeline
"""
import asyncio
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
import polars as pl
import tempfile
import os

class IngestionItem(BaseModel):
    item_id: int
    title: str = Field(min_length=3)
    price: float = Field(gt=0.0)
    category: str
    in_stock: bool

    @field_validator("title")
    def clean_title(cls, v: str) -> str:
        return v.strip().title()

class ProductionCrawler:
    def __init__(self, concurrency: int = 5, rps: float = 15.0):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rps = rps
        self.interval = 1.0 / rps
        self.last_req_time = 0.0
        self.lock = asyncio.Lock()
        self.dlq: List[Dict[str, Any]] = []

    async def _rate_limit(self):
        async with self.lock:
            now = time.monotonic()
            wait = self.last_req_time + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_req_time = time.monotonic()

    async def fetch_item(self, item_id: int) -> Optional[IngestionItem]:
        async with self.semaphore:
            await self._rate_limit()
            
            for attempt in range(1, 4):
                try:
                    # Simulated async fetch & DOM parse
                    await asyncio.sleep(0.01)
                    if random.random() < 0.1 and attempt < 3:
                        raise ConnectionResetError("Simulated socket reset")
                        
                    raw_data = {
                        "item_id": item_id,
                        "title": f"  Enterprise AI Accelerator {item_id}  ",
                        "price": 1499.0 + item_id,
                        "category": "HARDWARE",
                        "in_stock": True
                    }
                    return IngestionItem(**raw_data)
                except Exception as e:
                    if attempt == 3:
                        self.dlq.append({"item_id": item_id, "error": str(e)})
                        return None
                    await asyncio.sleep(0.05 * (2 ** attempt))
            return None

    async def crawl_batch(self, item_ids: List[int]) -> pl.DataFrame:
        tasks = [self.fetch_item(i) for i in item_ids]
        results = await asyncio.gather(*tasks)
        valid_items = [item.model_dump() for item in results if item is not None]
        return pl.DataFrame(valid_items)

# Verification Run
if __name__ == "__main__":
    async def main():
        crawler = ProductionCrawler(concurrency=10, rps=50.0)
        items_to_crawl = list(range(100))
        df = await crawler.crawl_batch(items_to_crawl)
        print("Scrape complete.")
        print(f"Extracted {df.height} valid items. DLQ contains {len(crawler.dlq)} failures.")
        print("Sample data head:\n", df.head(3))
        assert df.height + len(crawler.dlq) == 100

    asyncio.run(main())
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Concurrency & Rate-Limiting Control** | 30 | Flawless semaphore management, strict RPS enforcement, anti-fingerprint jitter. |
| **Resilience & Retry Mechanics** | 25 | Jittered exponential backoff, status code differentiation, zero unhandled worker exceptions. |
| **Schema Extraction & Pydantic Validation** | 20 | Robust field cleaners, type casts, and strict domain validation contracts. |
| **Dead-Letter Queue & Parquet Export** | 15 | Accurate DLQ tracking with diagnostic context; direct clean export to Parquet. |
| **Verification & Reproducibility** | 10 | Complete async test suite verifying batch execution and graceful error recovery. |
