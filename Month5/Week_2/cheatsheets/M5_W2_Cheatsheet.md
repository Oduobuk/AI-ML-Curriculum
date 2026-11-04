# Month 5, Week 2 Cheatsheet: Multi-Agent Systems, LangGraph Cycles & Model Context Protocol (MCP)

---

## 1. LangGraph Cyclic State Graph Architecture

Autonomous multi-agent collaboration requires cyclic topologies with state immutability:

```
                      LangGraph Cyclic Multi-Agent Workflow
                 ┌───────────────────────────────────────┐
                 │          User Task Ingestion          │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                 ┌───────────────────────────────────────┐
                 │             Planner Node              │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                 ┌───────────────────────────────────────┐
                 │       Developer / Tool Node           │◄────────┐
                 └───────────────────┬───────────────────┘         │
                                     │                             │
                                     ▼                             │ (Reject)
                 ┌───────────────────────────────────────┐         │
                 │          Security Auditor Node        ├─────────┘
                 └───────────────────┬───────────────────┘
                                     │ (Pass)
                                     ▼
                 ┌───────────────────────────────────────┐
                 │            END / Final Output         │
                 └───────────────────────────────────────┘
```

### State Reducers with `Annotated`
```python
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    # Appends new messages rather than overwriting
    messages: Annotated[List[dict], operator.add]
    current_agent: str
    cycle_counter: int
    is_approved: bool
```

---

## 2. Model Context Protocol (MCP) JSON-RPC 2.0 Standard

Standardizes how LLMs invoke external tools over stdio/HTTP:

### 1. `tools/list` Response Schema
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "query_database",
        "description": "Execute read-only SQL query on analytics warehouse",
        "inputSchema": {
          "type": "object",
          "properties": {"sql": {"type": "string"}},
          "required": ["sql"]
        }
      }
    ]
  }
}
```

### 2. `tools/call` Invocation Schema
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "query_database",
    "arguments": {"sql": "SELECT COUNT(*) FROM users;"}
  }
}
```

---

## 3. Human-in-the-Loop Checkpoint State Interruption

For high-risk operations (financial transactions, data deletion), pause execution:
- **Breakpoint Configuration**: Graph pauses before node execution (`interrupt_before=["wire_transfer_node"]`).
- **Thread Checkpoint Persistence**: Saves state snapshot to SQLite/Postgres.
- **Resumption**: Human reviewer inspects state, approves/edits, and calls `app.update_state()` to resume.
