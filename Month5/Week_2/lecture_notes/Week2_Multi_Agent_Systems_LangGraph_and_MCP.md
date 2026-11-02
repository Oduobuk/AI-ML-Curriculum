# Month 5, Week 2: Autonomous Multi-Agent Systems & Tool Use (LangGraph & MCP)

![Stateful Multi-Agent Architecture](../../../docs/images/agentic_workflow.png)

## Overview
Moving beyond simple single-turn prompt-response patterns, modern AI systems operate as **autonomous, tool-augmented agents**. An AI Agent is an LLM endowed with:
1. **Planning & Reasoning:** Decomposing user goals into iterative steps.
2. **Memory:** Persistent conversation state and retrieved semantic memory.
3. **Tool Execution:** The ability to call external functions (APIs, databases, Python sandboxes, web browsers).
4. **Self-Correction & Reflection:** Evaluating output against specifications before returning results to the user.

---

## 1. Structured Tool Calling & Function Execution

Rather than generating unstructured natural language, modern LLMs are trained to output valid JSON conforming to strict schemas (via **Pydantic** models or JSON Schema).

```python
from pydantic import BaseModel, Field

class DatabaseQueryTool(BaseModel):
    """Executes a read-only SQL query against the analytical data warehouse."""
    sql_query: str = Field(description="Syntactically valid SQL query.")
    timeout_seconds: int = Field(default=30, description="Query execution timeout.")
```

When an agent needs external data, it pauses generation, emits the tool call payload, the execution runtime runs the function, and the observation is injected back into the LLM conversation buffer.

---

## 2. Stateful Agent Orchestration with LangGraph

While linear pipelines (like basic chains) execute sequentially, real-world agentic workflows require **cycles, conditional branching, human-in-the-loop validation, and persistent state**. 

**LangGraph** models agentic applications as Directed Graphs where:
* **Nodes:** Represent Python functions or LLM agents (e.g., Researcher, Coder, Critic).
* **Edges:** Represent control flow transitions based on state decisions.
* **State:** A shared typed schema updated as nodes execute.

### Minimal LangGraph Multi-Agent Implementation:

```python
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END

# 1. Define Shared Agent State
class AgentState(TypedDict):
    task: str
    plan: List[str]
    draft: str
    critique: str
    revision_count: Annotated[int, operator.add]

# 2. Define Node Functions
def planner_node(state: AgentState):
    print("-> Planning step")
    return {"plan": [f"Step 1: Research {state['task']}", "Step 2: Generate code"]}

def generator_node(state: AgentState):
    print("-> Generating initial solution")
    return {"draft": f"Implemented solution for {state['task']}", "revision_count": 1}

def critic_node(state: AgentState):
    print("-> Reviewing solution")
    is_valid = state["revision_count"] >= 2
    critique = "Approved" if is_valid else "Needs more detail on error handling"
    return {"critique": critique}

def should_continue(state: AgentState):
    if state["critique"] == "Approved":
        return END
    return "generator"

# 3. Assemble StateGraph
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("generator", generator_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "generator")
workflow.add_edge("generator", "critic")
workflow.add_conditional_edges(
    "critic",
    should_continue,
    {END: END, "generator": "generator"}
)

app = workflow.compile()

# Run the Graph
initial_state = {"task": "Build a secure FastAPI OAuth2 auth service", "revision_count": 0}
final_state = app.invoke(initial_state)
print("Execution Completed. Final Critique:", final_state["critique"])
```

---

## 3. The Model Context Protocol (MCP) Standard

The **Model Context Protocol (MCP)** is an open industry protocol standardizing how applications provide context and tools to LLMs. Instead of writing custom API integration code for every provider:
* **MCP Clients** (IDE agents, orchestrators) connect via a standard protocol (JSON-RPC over stdio or SSE).
* **MCP Servers** expose standard resources, prompts, and executable tools (e.g., PostgreSQL MCP server, GitHub MCP server, Slack MCP server).

---

## 4. LLM Observability & Tracing with Langfuse

Because agentic workflows execute multi-step tool calls and dynamic loops, production engineering demands distributed tracing:
* **Latency tracking** per tool execution.
* **Token cost analysis** broken down by input/output/cached tokens.
* **Failure isolation** identifying exactly which subagent or tool call failed.

Integrating **Langfuse** or **OpenTelemetry** hooks allows teams to inspect full execution spans and enforce evaluation guardrails.

