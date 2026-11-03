# Month 1, Week 3 Cheatsheet: High-Throughput Asynchronous Ingestion & Web Scraping

---

## 1. Asynchronous Networking & Connection Pooling Architecture

```
                  Async Ingestion Worker Pool Architecture
                 ┌───────────────────────────────────────┐
                 │          Target Web / API Pool        │
                 └───────────────────▲───────────────────┘
                                     │
                        HTTP/2 Keep-Alive Connections
                                     │
                 ┌───────────────────┴───────────────────┐
                 │       httpx.AsyncClient / aiohttp     │
                 │   - Connection Pool: max_connections  │
                 │   - Semaphore Gate: bounds in-flight  │
                 └───────────────────▲───────────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
                [Worker 1]      [Worker 2]      [Worker 3]
```

### High-Performance `httpx` Client Setup
```python
import httpx
import asyncio

limits = httpx.Limits(
    max_connections=100,            # Global TCP connection cap
    max_keepalive_connections=20,   # Reusable warm sockets
    keepalive_expiry=30.0           # Seconds before closing idle connections
)
timeout = httpx.Timeout(10.0, connect=3.0)

async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=True) as client:
    response = await client.get("https://api.internal/v1/stream")
```

---

## 2. Token Bucket Rate Limiting with Full Jitter

To prevent HTTP 429 throttling and distributed anti-bot fingerprinting:
$$\Delta t_k = \min(\text{max\_backoff}, \text{base\_delay} \times 2^{k-1}) + \text{Uniform}(0, \text{jitter})$$

```python
import time
import asyncio
import random

class AsyncTokenBucket:
    def __init__(self, rps: float, burst_capacity: float):
        self.rate = rps
        self.capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                # Non-blocking wait for deficit
                deficit = 1.0 - self.tokens
                await asyncio.sleep(deficit / self.rate + random.uniform(0.01, 0.05))
```

---

## 3. High-Speed DOM Extraction with `lxml.html`

`lxml` is written in C and parses malformed HTML $10\times$ faster than pure-Python parsers:
```python
from lxml import html

tree = html.fromstring(raw_html_payload)

# Fast XPath Expressions
titles = tree.xpath('//div[contains(@class, "product-card")]//h3/text()')
prices = tree.xpath('//span[@data-price]/@data-price')
links  = tree.xpath('//a[contains(@href, "/item/")]/@href')

# Clean text stripping
clean_titles = [t.strip() for t in titles if t.strip()]
```

---

## 4. Producer-Consumer Worker Queue with Dead-Letter Isolation

```python
import asyncio

async def scraper_worker(in_queue: asyncio.Queue, dlq: asyncio.Queue, client: httpx.AsyncClient):
    while True:
        url, attempts = await in_queue.get()
        try:
            resp = await client.get(url)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 1.0))
                await asyncio.sleep(retry_after)
                raise ConnectionError("Rate limited")
            # Process success payload...
        except Exception as exc:
            if attempts < 3:
                await in_queue.put((url, attempts + 1))
            else:
                await dlq.put({"url": url, "error": str(exc), "retries": attempts})
        finally:
            in_queue.task_done()
```

---

## 5. Pydantic v2 Ingestion Contract

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class IngestedDocument(BaseModel):
    doc_id: str
    url: str
    title: str = Field(min_length=3)
    content: str = Field(min_length=10)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("title")
    def sanitize_title(cls, v: str) -> str:
        return v.strip().replace("\n", " ")
```
