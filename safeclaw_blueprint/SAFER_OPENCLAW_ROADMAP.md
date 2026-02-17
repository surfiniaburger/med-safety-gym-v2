# The Safer OpenClaw Roadmap: Building the Governed Agent

Based on the analysis of `@openclaw` architecture and distilled safety knowledge from the Med-Safety-Gym, this is the plan to rebuild/evolve the platform into a "SafeClaw" architecture.

---

## 1. The Strategy: "Safety-as-Code"
OpenClaw prioritizes "Availability" and "Extensibility." SafeClaw prioritizes **"Governed Agency."**

### Core Pivot points:
| From | To | Why |
| :--- | :--- | :--- |
| Prompt-based Safety | Invariant-based Safety | Prompts are fragile; Code is law. |
| Global Trust | JIT Escalation | Reduces the impact of "Session Takeover." |
| Local Execution | Distributed Governance | Allows central management of high-risk tools. |

---

## 2. Phase 1: The "Governor" (Foundation)
*Target: Move Manifest logic to the Hub.*

- [ ] **Unified Registry**: Centralize `claw_manifest.json` in a database (Supabase/PostgreSQL).
- [ ] **Tiered Classification API**: Hub exposes endpoints to verify if a tool is `user`, `write`, or `admin`.
- [ ] **JWT Escalation Tokens**: When HITL (TouchID/Telegram) is approved, the Hub issues a short-lived token specifically for that tool.

## 3. Phase 2: The "Runner" (Ephemerality)
*Target: Decouple Execution from Authority.*

- [ ] **Manifest-Stripped Runners**: The Agent only knows about "User" tools at boot. It must "fetch" permissions for "Write" tools.
- [ ] **Interceptor Refactor**: Implement the JIT logic from `med_safety_gym` into the core OpenClaw node runner.
- [ ] **Docker Sandboxing**: Default to `Dockerfile.sandbox` for all community-contributed skills.

## 4. Phase 3: The "Guardian" (Defense-in-Depth)
*Target: Automated Auditing.*

- [ ] **Action Parity Audit**: Before executing an `admin` tool, the hub runs a Gemini Vision check to ensure the intent matches the prompt.
- [ ] **Toxic Context Rollback**: If the conversation history exceeds a certain "Conflict Score" (detected by the Hub), the session is rolled back to a safe state.
- [ ] **Formal Verification Integration**: Continuously run TLA+ model checks against the permission flow in the CI/CD pipeline.

## 5. Phase 4: Lifecycle & Energy Optimization
*Target: Scale-to-Zero & Sustainability.*

- [ ] **The "Motion Detection" Pattern**: Agents hibernate (TTL) when idle, killing MCP sub-processes.
- [ ] **Cold Start Persistence**: Restore state from SQLite only when "Motion" (new user input) is detected.
- [ ] **Context Eviction**: Only load the active context set during cold starts to save LLM compute.

---

## 5. Immediate Technical Considerations

### A. The "Skill" Problem
OpenClaw's `apps.acp.md` (Command Palette) and system-level skills (`src/infra/exec-approvals.ts`) are highly privileged.
- **SafeClaw Step**: All system commands must be moved to the `Admin` tier, requiring biometric/HITL confirmation by default. No exceptions for "Developer Mode" unless explicitly toggled (and logged).

### B. The "Context Leak" Problem
Agents often "remember" too much, including previous admin instructions.
- **SafeClaw Step**: Implement **Context Scoping**. When a session is "Escalated," it's a fresh sub-session. When de-escalated (TTL), that context is purged to prevent subsequent low-privilege prompts from referencing high-privilege results.

---

## 6. Success Metrics
- **Zero RCE**: No prompt injection should ever bypass the manifest tier.
- **HITL Latency**: TouchID approval to tool execution < 500ms.
- **Governor Uptime**: Central policy must be reachable; if unreachable, agents fail-safe (withdraw to `User` tier).

---

> [!NOTE]
> We are not building a "New OpenClaw"; we are building the **Safety Layer** that allows OpenClaw to fulfill its "Agency" mission without compromising the host system.
