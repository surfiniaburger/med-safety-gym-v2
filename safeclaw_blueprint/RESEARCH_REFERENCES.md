# SafeClaw Project: Master Research References

This document serves as the authoritative list of external research, repositories, and newsletters referenced during the development of SafeClaw (Phases 30-45).

---

## 🏗️ Core Architectures & Proofs

### 1. Decapod Control Plane
- **Source**: [https://github.com/decapod-systems/decapod](https://github.com/decapod-systems/decapod)
- **Key Concept**: Proof-gated control planes and session proofs for agentic workloads.
- **Artifacts**: [AGENTS.md](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/tmp_decapod/AGENTS.md)

### 2. PI Agent (Coding Agent Specialist)
- **Source**: [https://github.com/pi-mono/pi-mono](https://github.com/pi-mono/pi-mono)
- **Key Concept**: Subagent spawning logic and the "Skills System" (self-building tools).
- **Artifacts**: [skills.md](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/tmp_pimono/packages/coding-agent/docs/skills.md), [runSubagent](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/tmp_pimono/packages/coding-agent/examples/extensions/subagent/index.ts)

### 3. OpenEnv
- **Source**: RFC 005 (Harness Wrapping)
- **Key Concept**: Wrapping the runner in a harness to audit conversational turns.
- **Comparison**: [SAFECLAW_VS_OPENENV_SYNTHESIS.md](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/SAFECLAW_VS_OPENENV_SYNTHESIS.md)

---

## 🆔 Identity & Governance

### 4. Teleport Agentic Identity
- **Source**: [https://github.com/gravitational/teleport](https://github.com/gravitational/teleport)
- **Key RFD**: [RFD 0238 - Delegating Access to AI Workloads](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/tmp_teleport/rfd/0238-delegating-access-to-ai-workloads.md)
- **Key RFD**: [RFD 0209 - MCP Access](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/tmp_teleport/rfd/0209-mcp-access.md)
- **SafeClaw Analysis**: [TELEPORT_TECH_AUDIT.md](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/TELEPORT_TECH_AUDIT.md)

---

## 🧠 Cognitive Memory & Parallelism

### 5. Cognee (Knowledge Graph Memory)
- **Source**: Cognee Research / s2Blog.md
- **Key Concept**: Replacing Vector RAG with structured Knowledge Graphs to solve "Relationship Blindness."
- **SafeClaw Strategy**: [MEMORY_AND_PARALLELISM_STRATEGY.md](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/MEMORY_AND_PARALLELISM_STRATEGY.md)

---

## 📰 Intelligence & Industry Trends

### 6. Newsletter References
- **sBlog.md**: Analysis of the "Identity Problem" vs "Security Problem" (Teleport x Daily Dose of DS).
- **s2Blog.md**: Analysis of parallel subagents and graph-backed memory.
- **Daily Dose of DS**: IBM 2025 Breach Report findings (97% of orgs lack AI access controls).

---

## 🛡️ Internal Blueprint Documents

- [ARCHITECTURAL_SOVEREIGNTY.md](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/safeclaw_blueprint/ARCHITECTURAL_SOVEREIGNTY.md): Governor vs Runner split.
- [SAFER_OPENCLAW_ROADMAP.md](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/safeclaw_blueprint/SAFER_OPENCLAW_ROADMAP.md): Evolutionary path.
- [AGENTIC_IDENTITY_FINDINGS.md](file:///Users/surfiniaburger/Desktop/med-safety-gym-v2/AGENTIC_IDENTITY_FINDINGS.md): Solving the identity crisis.
