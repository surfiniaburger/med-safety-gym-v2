# SafeClaw: Holistic Sovereignty & Safety Blueprint

This document provides a unified, consolidated view of the SafeClaw architecture, integrating the **Intent Mediator**, **Safety Guard (Guardian)**, **Sovereign Governor (Hub)**, and **Agentic Evaluation** systems.

## 1. High-Level Architecture: The Sovereignty Loop

SafeClaw operates as a sovereign agent where policy and identity are enforced by a central Governor, and execution is hardened by an internal safety layer.

```mermaid
graph TB
    subgraph "Trust Boundary: SafeClaw Agency"
        Agent["🤖 SafeClaw Agent<br/>(Execution)"]
        Guardian["🛡️ Safety Guard<br/>(Entity Parity)"]
        Mediator["🧠 Intent Mediator<br/>(Classification)"]
    end

    subgraph "Trust Boundary: Evaluation Layer"
        BS["🚀 Benchmark Server<br/>(Verification)"]
        HBG["📝 HealthBench Grader<br/>(Grading)"]
    end

    subgraph "Central Authority: Governor"
        Hub["🏰 SafeClaw Hub<br/>(Policy Enforcement)"]
        SM["📂 Scoped Manifests<br/>(RBAC)"]
    end

    %% Interaction Flows
    Hub --> |"Scoped Manifest"| Agent
    Agent --> |"Classify Intent"| Mediator
    Agent --> |"Tool Validation"| Guardian
    
    BS --> |"Prober Scenarios"| Agent
    Agent --> |"Transcripts"| BS
    BS --> |"Score"| HBG
```

---

## 2. The Message Lifecycle: Intent-First Guarding

Every incoming message undergoes a multi-layer verification process before any tool execution or LLM response is generated.

### Execution Flow:
1.  **Classification**: The `IntentClassifier` categorizes the input (e.g., `Follow-up`, `Correction`).
2.  **Mediator Injection**: The intent is used to enrich the LLM's context, mitigating "Intent Mismatch" (Lost in Conversation).
3.  **Scoped Interception**: If the LLM requests a tool call, the `ManifestInterceptor` ensures the tool is within the Governor-approved scope.
4.  **Clinical Safety Gate**: For medical tools, the **Guardian** verifies **Entity Parity** (ensuring no unauthorized medical entities are introduced).

```mermaid
sequenceDiagram
    participant U as User/Prober
    participant A as SafeClaw Agent
    participant M as Mediator
    participant G as Guardian/Interceptor
    participant T as Tools (MCP/GitHub)

    U->>A: "What about treatment for DIPG?"
    A->>M: Classify Intent
    M-->>A: Intent: Expansion / Medical
    
    A->>A: Generate Tool Call (e.g., check_entity_parity)
    A->>G: Intercept Tool Call
    G->>G: Check Manifest Scope
    G->>T: Execute (if safe)
    T-->>G: Result: {is_safe: True}
    G-->>A: Tool Outcome

    A->>A: Generate Final LLM Response
    A-->>U: "ONC201 is in trial..."
```

---

---

## 3. Data & Evaluation Lifecycle: The "Sovereignty Pulse"

SafeClaw isn't just a static agent; it's a dynamic system that learns from its own execution history and undergoes rigorous multi-turn stress testing.

### The Observability & RAG Flow
This diagram shows how live execution data is streamed to the Governor for real-time monitoring and later distilled into RAG context for future safety alignment.

```mermaid
graph LR
    Agent["🤖 SafeClaw Agent"] --> |"Stream Snapshots (WS)"| Hub["🏰 Hub/Governor"]
    Hub --> |"Persist"| DB[(Evolution DB)]
    DB --> |"Scan & Pair"| Refiner["🧪 Experience Refiner"]
    Refiner --> |"Distill"| Guidelines["📝 Pragmatic Guidelines"]
    Guidelines --> |"Inject"| Mediator["🧠 Intent Mediator"]
    DB --> |"Semantic Search"| RAG["🔍 RAG Service"]
    RAG --> |"Safety Context"| Agent
```

### The Agentic Evaluation Loop
SafeClaw is tested using the **A2A (Agent-to-Agent)** protocol, where specialized probers attempt to trigger safety failures.

```mermaid
sequenceDiagram
    participant B as 🚀 Benchmark Server
    participant P as 🧐 Prober (MT-Eval/MediQ)
    participant A as 🤖 SafeClaw Agent
    participant G as ⚖️ HealthBench Grader

    B->>P: Load Scenario
    P->>A: Turn 1: Specific Query (A2A)
    A-->>P: Response 1
    P->>A: Turn N: Adversarial Follow-up (Context Pressure)
    A-->>P: Response N
    P->>B: Final Transcript (Full session)
    B->>G: Grade Protocol
    G->>G: Verify Entity Parity & Abstention
    G-->>B: Final Logic Score (R/P)
```

---

## 4. Sovereign Policy & Profile Enforcement

The Governor (Hub) dictates the capabilities of the Agent based on a **Profile-to-Tool** mapping. This ensures that a "Read-Only" agent cannot perform destructive "Admin" actions, even if the LLM is compromised.

### Asymmetric Verification Invariant
SafeClaw implements an **Asymmetric Trust Model** to prevent manifest tampering:
1.  **Hub (Governor)**: Generates an Ed25519 key pair. It signs every scoped manifest using its **Private Key**.
2.  **Agent (Subject)**: Fetches the **Public Key** and uses it to verify the digital signature of the manifest.
3.  **Result**: An unsigned or tampered manifest is immediately rejected by the Agent, which falls back to a "Zero-Trust" restricted policy.

```mermaid
graph LR
    H[🏰 Hub] --> |"Sign Manifest (PrivKey)"| M[📜 Scoped Manifest]
    M --> |"Verify (PubKey)"| A[🤖 SafeClaw Agent]
    A --> |"Success"| E[✅ Active Policy]
    A --> |"Fail"| D[🚫 Zero-Trust Fallback]
```

### Profile Scopes (RBAC)
| Profile | Allowed Scopes | Key Capabilities |
| :--- | :--- | :--- |
| **Read-Only** | Information Retrieval | `list_issues`, `check_entity_parity` |
| **Developer** | Repository Management | `configure_repo`, `create_issue` |
| **Admin** | Critical Operations | `delete_repo`, `unlock_admin_tools` |

---

## 5. The Future Role of Agentic Evaluation

Agentic Evaluation is not just a "unit test" for models; it is a **Continuous Security Auditor** and a **Pragmatic Alignment Validator**.

### Evolutionary Roadmap:
- **Pragmatic Drift Detection**: Using the `BenchmarkServer` to detect if the `IntentClassifier` loses accuracy during extremely long multi-turn sessions (over 50+ turns).
- **Adversarial Red-Teaming**: The probers will evolve to intentionally use **"Pragmatic Ellipsis"** (vague directives) to try and trick the Agent into violating Entity Parity.
- **Automated Refinement**: In the future, the `HBG` (Grader) scores will be used to automatically tune the `Mediator` prompts, creating a self-healing safety loop.

---

## 6. Subtle Logic Flows

SafeClaw performs non-trivial logic for data retrieval and multi-turn state management to ensure accuracy and safety.

### Hybrid Search & Comparison
The `DataAgent` uses a layered approach to find interesting failures and compare model versions.

```mermaid
flowchart TD
    Q[Query] --> Typo{Typo Correction}
    Typo --> SQL[SQL Keyword Search]
    Typo --> SEM[Semantic Search /pgvector]
    SQL --> Rank[Unified Ranking]
    SEM --> Rank
    Rank --> Delta[Calculate Reward Delta]
    Delta --> UI[Gauntlet Dashboard]
```

### MT-Eval Recollection State Machine
Specialized probers use a state-driven approach to pressure the Agent's memory.

```mermaid
stateDiagram-v2
    [*] --> Turn1: Initial Query (DIPG)
    Turn1 --> Turn2: Correction (Formulation)
    Turn2 --> Turn3: Expansion (Side Effects)
    Turn3 --> Turn4: Recollection (Base Case)
    Turn4 --> [*]: Grade Transcript
```

---

## 7. Testing & Reliability Invariants

The SafeClaw test suite is designed for **"Zero-Friction" CI/CD**, ensuring that security never prevents rapid iteration.

### Mocking & Privacy Boundaries
Mocks are strictly enforced to prevent keychain popups and external API leaks.

```mermaid
graph TD
    subgraph "SafeClaw Test Suite"
        Tests[Pytest]
    end

    subgraph "Mocked Boundaries"
        Bio["🛡️ Biometric Auth<br/>(Always Return True)"]
        KR["🔑 Keyring/Keychain<br/>(Always Return None)"]
        SS["📂 Secret Store<br/>(InMemory Only)"]
        LLM["🗣️ LiteLLM<br/>(AsyncMock Completion)"]
    end

    Tests --> Bio
    Tests --> KR
    Tests --> SS
    Tests --> LLM
```

---
> [!IMPORTANT]
> This holistic approach ensures that **Sovereignty** is not just about isolated safety checks, but a systemic loop of classification, interception, and verification.
