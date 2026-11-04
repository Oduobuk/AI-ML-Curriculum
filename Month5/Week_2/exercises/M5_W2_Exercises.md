# Week 2: Graduate-Level Problem Set — Multi-Agent Systems, LangGraph Cycles & Model Context Protocol (MCP)

Welcome to the Month 5 Week 2 Problem Set. This suite evaluates your mastery of autonomous multi-agent orchestration: cyclic state graph topologies (LangGraph), conditional edge routing, human-in-the-loop checkpoint persistence, tool node isolation, and Model Context Protocol (MCP) JSON-RPC tool specification.

---

## Tier 1: Core Mechanics & LangGraph State Reducer

### Exercise 1: Stateful Graph Reducer & Append-Only Message History
**Concepts**: `TypedDict` state schema, `Annotated` reducers, operator merging, state immutability.

#### Problem Specification
In multi-agent loops, state is passed between nodes. Re-writing state destroys agent memory, while naive concatenation causes duplicate messages:
1. Define a LangGraph agent state `AgentWorkflowState`:
   - `messages: Annotated[List[BaseMessage], operator.add]`
   - `active_agent: str`
   - `iteration_count: int`
   - `final_decision: Optional[str]`
2. Implement two worker nodes:
   - `researcher_node(state)`: appends research findings to message history.
   - `auditor_node(state)`: inspects findings, verifies factual grounding, and increments `iteration_count`.
3. Implement a conditional router `should_continue(state)`:
   - If findings are approved: routes to `END`.
   - If findings are rejected and `iteration_count < 3`: routes back to `researcher_node`.
   - If `iteration_count >= 3`: routes to `fallback_node`.

#### Verification Solution
```python
from typing import TypedDict, List, Dict, Any, Optional
import operator
from dataclasses import dataclass

@dataclass
class Message:
    sender: str
    content: str

class AgentWorkflowState(TypedDict):
    messages: List[Message]
    active_agent: str
    iteration_count: int
    is_approved: bool

def researcher_node(state: AgentWorkflowState) -> Dict[str, Any]:
    current_iter = state["iteration_count"]
    new_msg = Message(sender="Researcher", content=f"Report Draft {current_iter + 1}")
    return {
        "messages": state["messages"] + [new_msg],
        "active_agent": "Auditor",
        "iteration_count": current_iter + 1
    }

def auditor_node(state: AgentWorkflowState) -> Dict[str, Any]:
    # Approves on iteration 2
    approved = (state["iteration_count"] >= 2)
    verdict = "APPROVED" if approved else "NEEDS_REVISION"
    new_msg = Message(sender="Auditor", content=f"Review verdict: {verdict}")
    return {
        "messages": state["messages"] + [new_msg],
        "active_agent": "Router",
        "is_approved": approved
    }

def router_condition(state: AgentWorkflowState) -> str:
    if state["is_approved"]:
        return "END"
    if state["iteration_count"] >= 3:
        return "FALLBACK"
    return "RESEARCHER"

# Verification Run
if __name__ == "__main__":
    state: AgentWorkflowState = {
        "messages": [],
        "active_agent": "Researcher",
        "iteration_count": 0,
        "is_approved": False
    }
    
    # Run loop
    while True:
        if state["active_agent"] == "Researcher":
            updates = researcher_node(state)
            state.update(updates)
            state["active_agent"] = "Auditor"
        elif state["active_agent"] == "Auditor":
            updates = auditor_node(state)
            state.update(updates)
            next_step = router_condition(state)
            print(f"Cycle {state['iteration_count']} -> Router decision: {next_step}")
            if next_step == "END":
                print("Workflow reached successful termination.")
                break
            elif next_step == "FALLBACK":
                print("Workflow reached fallback limit.")
                break
            state["active_agent"] = "Researcher"
            
    assert state["is_approved"] is True
    assert len(state["messages"]) == 4
```

---

## Tier 2: Model Context Protocol (MCP) JSON-RPC Specification

### Exercise 2: Pure Python Model Context Protocol (MCP) Tool Server
**Concepts**: Anthropic Model Context Protocol (MCP), JSON-RPC 2.0 protocol, `tools/list`, `tools/call`.

#### Problem Specification
The Model Context Protocol standardizes how AI agents communicate with external databases, execution sandboxes, and file systems over standard I/O (stdio) via JSON-RPC 2.0:
1. Implement a minimal MCP Server handling JSON-RPC requests:
   - Method `tools/list`: returns JSON Schema for available tools (`query_sql_db`, `read_file`).
   - Method `tools/call`: validates input arguments against JSON schema, executes tool logic, and returns formatted `content: [{"type": "text", "text": "..."}]`.
   - Handles standard JSON-RPC error codes (`-32601 Method not found`, `-32602 Invalid params`).
2. Verify execution across 3 standard JSON-RPC payloads.

#### Verification Solution
```python
import json
from typing import Dict, Any

class MockMCPServer:
    def __init__(self):
        self.tools = {
            "query_sql_db": {
                "description": "Execute read-only SQL query on the analytics database",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }

    def handle_json_rpc(self, request_str: str) -> str:
        try:
            req = json.loads(request_str)
        except json.JSONDecodeError:
            return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "tools/list":
            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": [{"name": k, **v} for k, v in self.tools.items()]}
            })

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            if tool_name not in self.tools:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Tool '{tool_name}' not found"}
                })
            if "query" not in args:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "Missing required argument 'query'"}
                })
            # Execute mock query
            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": "Returned 14 rows from database."}]}
            })

        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not implemented"}
        })

# Verification Run
if __name__ == "__main__":
    server = MockMCPServer()
    
    # 1. List tools
    req_list = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    res_list = json.loads(server.handle_json_rpc(req_list))
    print("MCP tools/list Response:", res_list["result"])
    assert len(res_list["result"]["tools"]) == 1
    
    # 2. Call tool
    req_call = json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "query_sql_db", "arguments": {"query": "SELECT * FROM orders"}}
    })
    res_call = json.loads(server.handle_json_rpc(req_call))
    print("MCP tools/call Response:", res_call["result"])
    assert "Returned 14 rows" in res_call["result"]["content"][0]["text"]
```

---

## Tier 3: Human-in-the-Loop Checkpoint State Interruption

### Exercise 3: Dynamic Checkpoint Serialization & Breakpoint Resumption
**Concepts**: State persistence, interrupted execution graph, dynamic state patching.

#### Problem Specification
For high-risk actions (initiating a financial refund, executing a destructive database drop), autonomous agents must pause execution and await human authorization:
1. Implement a checkpoint state manager `CheckpointMemoryManager`:
   - Saves serialized state checkpoints indexed by `thread_id` and `checkpoint_id`.
   - Pauses graph execution when a node flags `requires_human_approval = True`.
   - Supports external human input (`approve=True` or parameter edit) to resume execution from the exact interrupted state.

#### Verification Solution
```python
from typing import Dict, Any

class CheckpointedGraph:
    def __init__(self):
        self.checkpoints: Dict[str, Dict[str, Any]] = {}

    def save_checkpoint(self, thread_id: str, state: Dict[str, Any]):
        self.checkpoints[thread_id] = state.copy()

    def load_checkpoint(self, thread_id: str) -> Dict[str, Any]:
        return self.checkpoints.get(thread_id, {}).copy()

    def run_step(self, thread_id: str, action: str, amount: float) -> str:
        state = self.load_checkpoint(thread_id) or {"status": "INITIAL", "amount": amount}
        
        # High value risk gate
        if amount > 1000.0 and state.get("approved") is not True:
            state["status"] = "PENDING_HUMAN_APPROVAL"
            state["amount"] = amount
            self.save_checkpoint(thread_id, state)
            return "PAUSED: Action requires human approval."
            
        state["status"] = "EXECUTED"
        self.save_checkpoint(thread_id, state)
        return f"SUCCESS: Action {action} with ${amount} executed."

# Verification Run
if __name__ == "__main__":
    graph = CheckpointedGraph()
    thread = "session_tx_998"
    
    # 1. Attempt large transaction -> should pause
    res1 = graph.run_step(thread, "WIRE_TRANSFER", 50_000.0)
    print("Step 1 Result:", res1)
    assert "PAUSED" in res1
    
    # 2. Human inspects checkpoint and approves
    saved_state = graph.load_checkpoint(thread)
    assert saved_state["status"] == "PENDING_HUMAN_APPROVAL"
    saved_state["approved"] = True
    graph.save_checkpoint(thread, saved_state)
    
    # 3. Resume step
    res2 = graph.run_step(thread, "WIRE_TRANSFER", 50_000.0)
    print("Step 2 Result after approval:", res2)
    assert "SUCCESS" in res2
```
