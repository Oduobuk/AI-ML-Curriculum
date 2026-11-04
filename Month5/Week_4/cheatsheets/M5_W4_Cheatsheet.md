# Month 5, Week 4 Cheatsheet: Production FastAPI Gateways, SSE Token Streaming & Hardened Containers

---

## 1. Asynchronous FastAPI Gateway: Non-Blocking Architecture

```
                  FastAPI Asynchronous Request Flow
                 ┌───────────────────────────────────────┐
                 │       Incoming HTTP/2 Client Pool     │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                 ┌───────────────────────────────────────┐
                 │    FastAPI Event Loop (Single Thread) │
                 └───────────┬───────────────────────┬───┘
                             │                       │
           IO-Bound Tasks (async def)    CPU-Bound Inference
                             │                       │
                             ▼                       ▼
                     [Async DB / Redis]     asyncio.to_thread()
                                                     │
                                                     ▼
                                            [Worker Thread Pool]
```

### Event Loop Starvation Rule
Never execute synchronous CPU-bound model inference directly inside an `async def` route. Use:
```python
result = await asyncio.to_thread(model_inference_sync, payload)
```

---

## 2. Server-Sent Events (SSE) Streaming Protocol

Implements the standard W3C SSE event-stream protocol with immediate chunk flushing:
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

async def event_generator(prompt: str):
    tokens = ["Production", " AI", " gateway", " operational."]
    for tok in tokens:
        await asyncio.sleep(0.01)
        payload = {"choices": [{"delta": {"content": tok}}]}
        yield f"data: {json.dumps(payload)}\n\n"
        
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def stream_chat(req: ChatRequest):
    return StreamingResponse(
        event_generator(req.prompt), 
        media_type="text/event-stream"
    )
```

---

## 3. Pydantic v2 Production Validation

```python
from pydantic import BaseModel, Field

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Target model identifier")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, gt=0, le=4096)
    stream: bool = False
```

---

## 4. Hardened Multi-Stage Dockerfile Blueprint

```dockerfile
# 1. Builder Stage
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 2. Hardened Runtime Stage
FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.11/site-packages:$PYTHONPATH" \
    PYTHONUNBUFFERED=1

# Non-root user for security compliance
RUN groupadd -g 10001 appgroup && useradd -u 10001 -g appgroup -s /bin/false appuser
COPY --from=builder /install /install
COPY --chown=appuser:appgroup ./app /app/app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```
