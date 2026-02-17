# Scale Invariants: Safety Across 10,000 Agents

As SafeClaw scales from a single agent to a multi-tenant platform, we must ensure that security doesn't dilute with density. This document outlines the invariants required for high-scale, governed agency.

---

## 1. Identity Lineage Invariant
> *"Every bit of data processed by a Runner must be traceable back to a single human-paired identity."*

- **The Problem**: In high-scale environments, context leakage or "session jumping" can occur if the Hub misroutes a message.
- **The Scale Invariant**: 
    - No two Runners can share a single `Sub-process` ID. 
    - Every tool invocation payload MUST include a **Lineage Hash** signed by the Governor.
    - If a Runner attempts to use a tool without a hash matching its current session ID, the Hub triggers an immediate **Global Revocation**.

## 2. The "Blast Radius" Invariant
> *"A compromised Runner must not be able to discovery or interact with other Runners."*

- **The Problem**: Side-channel attacks in shared Docker networks.
- **The Scale Invariant**:
    - **Network Isolation**: Every Runner container must run in its own VPC or bridge network with strictly denied intra-container traffic.
    - **DNS Pinning**: Runners only resolve white-listed MCP domains and the Governor's internal IP.
    - **No Peer Discovery**: Tools like `nmap` or `ping` are explicitly blocked at the tier level.

## 3. Resource Exhaustion (DoS) Invariant
> *"An agentic loop must not consume unbudgeted resources (API credits or Compute)."*

- **The Problem**: A prompt-injected "infinite loop" that creates thousands of GitHub issues or consumes $500 of Gemini tokens in minutes.
- **The Scale Invariant**:
    - **Tiered Rate Limiting**: 
        - `User` tools: 100/hour.
        - `Write` tools: 10/hour.
        - `Admin` tools: 2/hour (requires HITL).
    - **Semantic Density Check**: The Hub monitors "Entropy Growth" in the context window. If the agent repeats the same tool pattern 3x without progress, it is paused for intervention.

## 4. Manifest Consistency Invariant
> *"The Governor must not grant access to a tool if its tier has been downgraded while a request was in flight."*

- **The Problem**: Race conditions during manifest updates.
- **The Scale Invariant**:
    - Manifests are **Atomic**. Use a logic clock (version number).
    - Every escalation token includes the `Manifest-Version-ID`.
    - If `Current-Manifest-Version` > `Token-Version-ID`, the PEP (Interceptor) must re-verify the tier before execution.

---

## 5. Persistence & Recovery at Scale

### Distributed Session Memory
Use a globally distributed Redis/KeyDB for `escalated_tools` to ensure that if a Hub instance fails, the state persists for the next instance.

### The "Nuke" Command
The Hub maintains a heartbeat to all Runners. If the Hub loses sight of a Runner's security state for > 30 seconds, it sends a high-priority "SIGKILL" to the container orchestrator for that specific CID.

---

> [!CAUTION]
> Scaling agents without these invariants leads to "Agentic Sprawl," where identity confusion becomes an exploitable attack vector.
