# Week 2 Production Engineering Assignment: Multi-Agent Autonomous Research & Execution System with LangGraph and MCP

## Executive Overview & Architectural Motivation
Single-prompt LLMs fail when tasked with multi-step software engineering, competitive market intelligence, and automated bug resolution. Complex objectives require **Multi-Agent Systems (MAS)** structured as cyclic state graphs where specialized autonomous agents (Planner, Researcher, Code Auditor, Executor) collaborate, critique each other's outputs, execute external tools via standard interfaces, and pause for human approval.

In this assignment, you will architect a **Multi-Agent Research & Code Generation System** combining **LangGraph** state graph principles and the **Model Context Protocol (MCP)**.

Your system will feature an append-only state reducer, specialized agent worker nodes, a sandboxed Model Context Protocol tool execution server, conditional edge routing with cycle bounds, and human-in-the-loop checkpoint approval gates.

---

## Technical Specifications & Architecture Diagram

```
                 User High-Level Task ("Benchmark PostgreSQL vs DuckDB indexing")
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          LangGraph Orchestration StateGraph          │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │                 Planner / Lead Agent                  │
                 │   - Decomposes task into sub-tasks                    │
                 │   - Emits structured execution plan                   │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │                 Research & Code Agent                 │
                 │   - Interacts with Model Context Protocol (MCP) Server│
                 │   - Executes tool calls: query_db, run_python_sandbox │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │                 Security & Audit Node                 │
                 │   - Verifies code execution safety                    │
                 │   - If code modifies DB -> Pauses for Human Approval  │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼ (Pass)                                      ▼ (Reject)
              [Final Solution]                              [Re-route to Researcher]
```

---

## Detailed Deliverable Requirements

### 1. LangGraph Cyclic State Engine (`graph.py`)
Implement `MultiAgentStateGraph`:
- State schema `AgentTeamState`:
  - `messages`: List of message dictionaries with sender tags.
  - `current_step`: Integer counter tracking cycles.
  - `plan`: List of discrete task actions.
  - `artifacts`: Dictionary storing generated code and analytical data.
- Nodes:
  - `planner_node`: Generates structured breakdown.
  - `developer_node`: Writes Python implementation and formats MCP tool calls.
  - `reviewer_node`: Evaluates code against requirements; routes back if issues found.
- Enforce strict cycle bounds ($\le 4$ iterations) to prevent infinite loops.

### 2. Model Context Protocol (MCP) Tool Server (`mcp_server.py`)
Implement a compliant JSON-RPC 2.0 tool execution server:
- Supports `tools/list` exposing:
  - `run_code_sandbox`: Evaluates Python expressions safely.
  - `fetch_documentation`: Returns local API markdown documentation.
- Supports `tools/call`: executes tools with parameter validation and returns structured text payloads.

### 3. Human-in-the-Loop Checkpoint Gate (`checkpoint.py`)
- If the reviewer identifies an irreversible or high-risk operation (e.g. database migration or external email delivery), pause graph execution.
- Serialize the thread state to disk (`checkpoint_{thread_id}.json`).
- Provide an API method `resume_graph(thread_id, user_approval=True)` to continue execution.

---

## Starter Code Template & Submission Skeleton

```python
"""
multi_agent_langgraph_mcp.py - Production Multi-Agent System
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class AgentMessage:
    agent: str
    content: str
    tool_call: Optional[Dict[str, Any]] = None

@dataclass
class WorkflowState:
    task: str
    messages: List[AgentMessage] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    code_artifact: Optional[str] = None
    audit_passed: bool = False
    cycle_count: int = 0
    paused_for_human: bool = False

class MultiAgentWorkflowEngine:
    def __init__(self, max_cycles: int = 3):
        self.max_cycles = max_cycles

    def planner_step(self, state: WorkflowState) -> WorkflowState:
        state.plan = ["1. Analyze schemas", "2. Generate index benchmark script", "3. Verify execution"]
        state.messages.append(AgentMessage("Planner", "Generated execution plan with 3 tasks."))
        return state

    def developer_step(self, state: WorkflowState) -> WorkflowState:
        state.code_artifact = "import duckdb\ncon = duckdb.connect(':memory:')\n# Benchmarking index"
        state.messages.append(AgentMessage("Developer", "Engineered code artifact for benchmark."))
        return state

    def auditor_step(self, state: WorkflowState) -> WorkflowState:
        state.cycle_count += 1
        
        # Simulate verification check
        if "duckdb" in (state.code_artifact or ""):
            state.audit_passed = True
            state.messages.append(AgentMessage("Auditor", "Code passed syntax, safety, and performance audits."))
        else:
            state.audit_passed = False
            state.messages.append(AgentMessage("Auditor", "Audit failed: Missing required database engine."))
            
        return state

    def run(self, initial_task: str) -> WorkflowState:
        state = WorkflowState(task=initial_task)
        state = self.planner_step(state)
        
        while state.cycle_count < self.max_cycles:
            state = self.developer_step(state)
            state = self.auditor_step(state)
            
            if state.audit_passed:
                print(f"Workflow completed successfully in cycle {state.cycle_count}!")
                break
                
        return state

# Verification Run
if __name__ == "__main__":
    engine = MultiAgentWorkflowEngine(max_cycles=3)
    final_state = engine.run("Benchmark DuckDB in-memory query execution")
    
    print("\nWorkflow Execution Log:")
    for msg in final_state.messages:
        print(f"[{msg.agent}] {msg.content}")
        
    assert final_state.audit_passed is True
    assert final_state.code_artifact is not None
    print("\nMulti-Agent orchestration successfully executed with full audit closure!")
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **StateGraph Reducers & Cyclic Topologies** | 30 | Flawless state propagation, immutable message appending, and strict cycle limits. |
| **Model Context Protocol (MCP) Compliance** | 30 | Valid JSON-RPC 2.0 schema, `tools/list` and `tools/call` handling, error handling. |
| **Conditional Edge Routing** | 20 | Dynamic decision branching based on auditor evaluations and task criteria. |
| **Human-in-the-Loop Checkpoint Persistence** | 10 | Serialized interrupt breakpoints for high-risk operations and clean resumption. |
| **Verification & Reproducibility** | 10 | Automated test suite verifying full multi-agent cycle completion. |
