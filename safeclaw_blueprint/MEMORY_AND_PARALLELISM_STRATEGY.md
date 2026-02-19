# SafeClaw Strategy: Cognitive Scaling (Memory & Parallelism)

This document outlines the roadmap for integrating **structured memory** (Knowledge Graphs) and **subagent parallelism** into the SafeClaw ecosystem, following the establishment of "Sovereign Decapod" governance.

---

## 1. The Core Conflict: Vectors vs. Graphs
*As highlighted in the @s2Blog.md newsletter*

| Memory Archetype | How it Works | Failure Mode |
| :--- | :--- | :--- |
| **Vector RAG** | Semantic similarity (matching keywords). | "Relationship Blindness." It knows Alice exists and Auth exists, but not that Alice *manages* Auth. |
| **Knowledge Graph** | Entities and relationships (Nodes & Edges). | "Inference Strength." It understands structural hierarchies and provenance. |

### The Cognee Integration Plan:
We will incorporate a **Knowledge Graph layer** (inspired by Cognee) to augment the current `SessionMemory`.
- **Phase 41**: Implement Entity Extraction. After every tool call, the Interceptor feeds the result to an "Observer" that extracts entities and relationships.
- **Phase 42**: Graph-Backed Recall. Before a new task, SafeClaw traverses the graph to provide structured context (e.g., "Note: Alice is the manager of the service you are editing").

---

## 2. Parallelism via Subagents
*Inspired by Ollama/Claude Code's `spawn` logic*

The "4 Primitives" (`read`, `write`, `edit`, `bash`) become the DNA for subagents.

### Scaling Pattern:
1.  **Orchestrator Agent**: Receives the high-level goal.
2.  **Spawn(Primitive)**: The Orchestrator uses a `subagent` tool. This spawns a new Dockerized "Runner" with:
    - **Isolated Context**: No history leak between workers.
    - **Specific Instruction**: e.g., "Research the auth flow."
    - **Filtered Tools**: The Governor only grants `read` and `bash` to the "Research Worker."
3.  **Aggregation**: The Orchestrator waits for the JSON output of all subagents and synthesizes the final action.

---

## 3. The Governance Invariant (The Decapod Priority)

**Critically**, we must implement **Sovereign Decapod (Governance)** *before* Parallelism.

> [!WARNING]
> **The Spawning Hazard**: If we enable subagents without a Governor, one compromised agent can spawn 10 malicious workers, exponentially increasing the attack surface.

### Safety Order:
1.  **Govenance (PDP/PEP)**: Establish the "High-Side" Hub that monitors all calls.
2.  **Bounded Parallelism**: Let the Governor decide if the Orchestrator is *allowed* to spawn a subagent based on current tier (e.g., `Admin` only can spawn `Sub-Admin`).
3.  **Entity-Aware Memory**: Use the Knowledge Graph to prevent "Context Leak" where a subagent accidentally inherits secrets it shouldn't see.

---

## 4. Ranked Strategic Roadmap

### 🥇 Priority 1: Sovereign Decapod (Phase 40)
- **Objective**: Establish the Hub as the mandatory gatekeeper. 
- **Status**: Current Work-in-Progress.

### 🥈 Priority 2: Structured Memory (Phase 41)
- **Objective**: Replace simple Markdown memory with a KG (Knowledge Graph).
- **Reasoning**: This provides the "Contextual Grounding" needed for agents to make fewer mistakes during high-stakes actions.

### 🥉 Priority 3: Subagent Parallelism (Phase 42)
- **Objective**: Enable complex, multi-component research tasks.
- **Reasoning**: Only safe to do once the Governor can "batch-approve" or "batch-deny" a swarm of agents.

---

## Future Invariant: "Memory Isolation"
Each subagent should have its own subgraph. When a subagent completes, the Orchestrator reviews the "Key Relations" found and merges only the safe/relevant ones back into the main Knowledge Graph.
