# Week 3: Graduate-Level Problem Set — Asynchronous Data Ingestion, Web Scraping & Resilient Streaming

Welcome to the Month 1 Week 3 Problem Set. This suite covers the engineering practices required for high-throughput, fault-tolerant web data ingestion and API extraction: asynchronous HTTP networking via `aiohttp` / `httpx`, token bucket rate limiting, HTTP/2 connection pooling, HTML DOM parsing with `BeautifulSoup` / `lxml`, and retry backoff mechanics.

---

## Tier 1: Core Mechanics & Async HTTP Concurrency

### Exercise 1: Asynchronous HTTP Pipeline with Bounded Semaphore
**Concepts**: `asyncio.Semaphore`, `httpx.AsyncClient`, connection pooling, connection limits.

#### Problem Specification
Naively spawning 10,000 asynchronous HTTP coroutines causes socket exhaustion (`Too many open files` / `ECONNREFUSED`):
1. Build an asynchronous fetcher function `fetch_url(client, url, semaphore)` that bounds concurrency using an `asyncio.Semaphore(max_concurrent=10)`.
2. Configure an `httpx.AsyncClient` with custom transport limits:
   - `max_connections = 50`
   - `max_keepalive_connections = 20`
   - `timeout = 5.0`
3. Execute simulated parallel requests across 100 endpoints and record the latency distribution ($P_{50}, P_{95}, P_{99}$).

#### Verification Solution
```python
import asyncio
import httpx
import time
import numpy as np
from typing import List

async def bounded_fetch(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore) -> tuple[str, int, float]:
    async with sem:
        t0 = time.perf_counter()
        # In a real environment, query live endpoints; here we demonstrate client semantics
        # using httpx mock transport or httpbin
        try:
            resp = await client.get(url)
            status = resp.status_code
        except Exception:
            status = 500
        latency = time.perf_counter() - t0
        return url, status, latency

async def run_ingestion_suite():
    sem = asyncio.Semaphore(10)
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
    timeout = httpx.Timeout(5.0)
    
    # Target endpoint pool (e.g. JSONPlaceholder / mock targets)
    urls = [f"https://jsonplaceholder.typicode.com/posts/{i}" for i in range(1, 21)]
    
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = [bounded_fetch(client, u, sem) for u in urls]
        results = await asyncio.gather(*tasks)
        
    latencies = [r[2] for r in results]
    print(f"Processed {len(results)} URLs successfully.")
    print(f"Latency P50: {np.percentile(latencies, 50)*1000:.2f} ms")
    print(f"Latency P95: {np.percentile(latencies, 95)*1000:.2f} ms")
    assert len(results) == 20

if __name__ == "__main__":
    # Standard test execution
    asyncio.run(run_ingestion_suite())
```

---

### Exercise 2: DOM Parsing & CSS/XPath Extraction with lxml
**Concepts**: `lxml.html`, XPath queries, malformed HTML recovery, schema extraction.

#### Problem Specification
1. Machine learning datasets scraped from web documents frequently contain malformed HTML, nested broken tags, and dynamically encoded unicode characters.
2. Given a raw HTML payload of an e-commerce catalog page:
   - Parse the document tree using `lxml.html.fromstring()`.
   - Write XPath expressions to extract product title, current price, rating, and availability status.
   - Clean and cast extracted text into structured types (`float` for price, `int` for review count).

#### Verification Solution
```python
from lxml import html
from typing import List, Dict, Any

SAMPLE_HTML = """
<html>
  <body>
    <div class="product-grid">
      <div class="item" data-id="101">
        <h3 class="title">NVIDIA RTX 5090 Blackwell</h3>
        <span class="price">$1,999.99</span>
        <span class="rating">4.9 stars (128 reviews)</span>
        <span class="badge in-stock">In Stock</span>
      </div>
      <div class="item" data-id="102">
        <h3 class="title">AMD Radeon 9900 XTX</h3>
        <span class="price">$1,199.50</span>
        <span class="rating">4.6 stars (84 reviews)</span>
        <span class="badge out-of-stock">Backordered</span>
      </div>
    </div>
  </body>
</html>
"""

def parse_product_catalog(html_content: str) -> List[Dict[str, Any]]:
    tree = html.fromstring(html_content)
    items = tree.xpath('//div[contains(@class, "item")]')
    catalog = []
    
    for item in items:
        item_id = int(item.xpath('./@data-id')[0])
        title = item.xpath('.//h3[contains(@class, "title")]/text()')[0].strip()
        price_raw = item.xpath('.//span[contains(@class, "price")]/text()')[0].strip()
        price = float(price_raw.replace("$", "").replace(",", ""))
        
        status_text = item.xpath('.//span[contains(@class, "badge")]/text()')[0].strip()
        in_stock = (status_text == "In Stock")
        
        catalog.append({
            "id": item_id,
            "title": title,
            "price": price,
            "in_stock": in_stock
        })
    return catalog

# Verification Run
if __name__ == "__main__":
    products = parse_product_catalog(SAMPLE_HTML)
    print("Extracted Products:", products)
    assert len(products) == 2
    assert products[0]["price"] == 1999.99
    assert products[0]["in_stock"] is True
```

---

## Tier 2: Token Bucket Rate Limiting & Anti-Ban Jitter

### Exercise 3: Adaptive Rate Limiter with 429 Retry-After Compliance
**Concepts**: Rate-limiting protocols, HTTP 429 status, `Retry-After` header parsing, token replenishment.

#### Problem Specification
1. External scraping APIs enforce rate limits (e.g. 10 requests per second) and emit `429 Too Many Requests` with a `Retry-After: <seconds>` header when exceeded.
2. Build an async rate-limited requester that:
   - Tracks client-side token bucket limits.
   - Intercepts HTTP 429 status codes, inspects `Retry-After`, and pauses execution automatically before retrying.
   - Injects randomized jitter $\sim \text{Uniform}(0.1, 0.5)$ into every request interval to avoid deterministic bot detection fingerprints.

#### Verification Solution
```python
import asyncio
import time
import random

class ResilientApiConsumer:
    def __init__(self, requests_per_sec: float = 10.0):
        self.rate = requests_per_sec
        self.capacity = requests_per_sec
        self.tokens = requests_per_sec
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def _consume_token(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0

    async def fetch_with_backoff(self, req_id: int) -> dict:
        await self._consume_token()
        # Add anti-fingerprint jitter
        await asyncio.sleep(random.uniform(0.01, 0.03))
        
        # Simulated server response with occasional rate limit
        if random.random() < 0.15:
            # Simulated 429
            retry_after = 0.05
            await asyncio.sleep(retry_after)
            return {"req_id": req_id, "status": 200, "retried": True}
        return {"req_id": req_id, "status": 200, "retried": False}

# Verification Run
if __name__ == "__main__":
    async def main():
        client = ResilientApiConsumer(requests_per_sec=20.0)
        tasks = [client.fetch_with_backoff(i) for i in range(25)]
        results = await asyncio.gather(*tasks)
        print("Completed 25 adaptive requests.")
        assert len(results) == 25
        assert all(r["status"] == 200 for r in results)

    asyncio.run(main())
```

---

## Tier 3: Distributed Web Scraping Pipeline

### Exercise 4: Producer-Consumer Web Scraping Queue with Error Dead-Letter Queue
**Concepts**: `asyncio.Queue`, worker pool architecture, graceful cancellation, dead-letter queue (DLQ).

#### Problem Specification
Build a complete asynchronous web scraping pipeline:
1. `crawl_queue`: holds pending URLs to scrape.
2. `worker_pool`: 5 concurrent worker tasks consuming from `crawl_queue`.
3. `dead_letter_queue`: catches unrecoverable failed URLs after 3 retry attempts with error diagnostics.
4. Process 50 URLs, gracefully shutting down workers upon queue completion (`crawl_queue.join()`).

#### Verification Solution
```python
import asyncio
import random
from typing import Dict, Any

async def scraper_worker(
    worker_id: int, 
    in_queue: asyncio.Queue, 
    dlq: asyncio.Queue, 
    results: list
):
    while True:
        url, attempts = await in_queue.get()
        try:
            # Simulated scrape
            await asyncio.sleep(0.02)
            if random.random() < 0.1:
                raise ConnectionError(f"HTTP 502 Bad Gateway at {url}")
            results.append({"url": url, "status": "SUCCESS", "worker": worker_id})
        except Exception as e:
            if attempts < 3:
                await in_queue.put((url, attempts + 1))
            else:
                await dlq.put({"url": url, "error": str(e), "attempts": attempts})
        finally:
            in_queue.task_done()

async def run_pipeline():
    in_queue: asyncio.Queue = asyncio.Queue()
    dlq: asyncio.Queue = asyncio.Queue()
    results: list = []
    
    for i in range(50):
        await in_queue.put((f"https://data.source.internal/item/{i}", 1))
        
    workers = [
        asyncio.create_task(scraper_worker(w, in_queue, dlq, results))
        for w in range(5)
    ]
    
    await in_queue.join()
    for w in workers:
        w.cancel()
        
    print(f"Scraped Successfully: {len(results)}")
    print(f"Dead Letter Queue (Failed): {dlq.qsize()}")
    assert len(results) + dlq.qsize() == 50

if __name__ == "__main__":
    asyncio.run(run_pipeline())
```
