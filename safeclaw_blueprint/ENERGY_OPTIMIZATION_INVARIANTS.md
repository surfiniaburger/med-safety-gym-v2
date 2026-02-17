# Energy Optimization: Scale-to-Zero & Motion-Detected Agency

SafeClaw aims to be a sustainable platform. Like motion-detected lighting, agentic resources (Compute, MCP Servers, LLM Contexts) should only be active during intentional interaction.

---

## 1. The "Motion Detection" Pattern
In SafeClaw, "Motion" is defined as a **User Input Event**.

### The Lifecycle:
1.  **Sleep State (Scale-to-Zero)**: 0 CPU/Mem. The Agent exists only as a row in the `ConversationSession` table.
2.  **Motion Detected**: A Telegram message or Webhook hit triggers the `Executor`.
3.  **Cold Start**: 
    - The `Executor` spawns the `SafeClawAgent`.
    - The Agent's `client_factory` triggers `uv run` to spin up local MCP servers.
    - Context is pulled from SQLite to restore "Mental Presence."
4.  **Active Engagement**: Agent processes the request.
5.  **Idle Countdown**: An `IDLE_REAPER_TTL` (Default: 5 Minutes) begins after the last outgoing message.
6.  **Scale Down**: If no further motion is detected, the `Executor` calls `agent.shutdown()`, kills the MCP subprocesses, and clears the instance from memory.

---

## 2. Invariants for Energy Efficiency

### A. Subprocess Pruning
> *"An MCP server must never outlive its parent Agent."*
- **Mechanism**: `SafeClawAgent.shutdown()` must guarantee the termination of all `uv run` sub-processes.
- **Why**: Prevents "Zombie MCPs" from draining battery/compute on a host (Mac/Linux).

### B. Adaptive Hibernation
> *"Agents with lower tier permissions should hibernate faster than Admin-escalated agents."*
- **Rationale**: Admin tools are expensive and high-risk. We want a tighter "Motion Window" for high-stakes sessions.
- **TTL Scaling**:
    - `User` Session: 10 min idle.
    - `Admin` Session: 2 min idle (Security + Energy).

### C. Context Compression (Energy Savings on LLM)
> *"Do not send the full 32k context if only the last 5 turns are relevant for 'current motion'."*
- **Mechanism**: Implement a "Context Eviction" policy. Store the full history in SQLite, but only load the "Active Working Set" into the LLM prompt during a cold start.

---

## 3. Implementation Plan: The "Idle Reaper"

- [ ] **`executor.py` Update**: Implement a background `asyncio` loop that monitors `self.agents` last-activity timestamps.
- [ ] **Graceful Persistence**: Ensure that `agent.shutdown()` is called *only* after session state (metadata, repo name) is safely flushed to the database.
- [ ] **Wake-on-Magic-Packet (Telegram)**: Use the Telegram `CallbackQuery` as a "Wake" signal for paused/hibernating tasks.

---

> [!TIP]
> This pattern makes SafeClaw perfectly suited for "Function-as-a-Service" (FaaS) environments like Google Cloud Run, where you only pay for the exact milliseconds the agent is thinking.
