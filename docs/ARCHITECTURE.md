# DIPG Safety Gym Architecture

## 🏗️ High-Level Architecture

This diagram shows how the components interact in the hybrid A2A + ADK + MCP system.

```mermaid
graph TD
    subgraph "Client Layer"
        User[User / Researcher]
        NB[Jupyter Notebook / Kaggle]
        A2A_Client[A2A Client SDK]
    end

    subgraph "Agent Layer (Port 10000)"
        Agent[ADK Agent]
        LiteLLM[LiteLLM Provider]
        Ollama[Ollama / Local LLM]
    end

    subgraph "MCP Layer (Port 8081)"
        FastMCP[FastMCP Server]
        Tools[MCP Tools]
    end

    subgraph "Evaluation Layer"
        Env[DIPG Environment]
        Dataset[HuggingFace Dataset]
        Metrics[Safety Metrics]
    end

    User -->|Interacts| NB
    NB -->|Uses| A2A_Client
    
    A2A_Client -->|A2A Protocol (HTTP)| Agent
    
    Agent -->|Inference| LiteLLM
    LiteLLM -.->|Optional| Ollama
    
    Agent -->|MCP Protocol (SSE)| FastMCP
    FastMCP -->|Exposes| Tools
    
    Tools -->|get_eval_tasks| Env
    Tools -->|evaluate_responses| Env
    
    Env -->|Loads| Dataset
    Env -->|Calculates| Metrics
```

## 🔄 Evaluation Data Flow

This sequence diagram illustrates the flow of data during an evaluation session.

```mermaid
sequenceDiagram
    participant NB as Notebook/Model
    participant Client as A2A Client
    participant Agent as ADK Agent
    participant MCP as FastMCP Server
    participant Env as DIPG Environment

    Note over NB, Env: 1. Initialization
    NB->>Client: Connect(Agent URL)
    Client->>Agent: Get Agent Card
    Agent-->>Client: Agent Card (Capabilities)

    Note over NB, Env: 2. Task Retrieval
    NB->>Client: "Get me an evaluation task"
    Client->>Agent: SendMessage("Get task")
    Agent->>MCP: CallTool("get_eval_tasks")
    MCP->>Env: Fetch Task from Dataset
    Env-->>MCP: Task Data (Context, Question)
    MCP-->>Agent: Task Data
    Agent-->>Client: Task (JSON)
    Client-->>NB: Task

    Note over NB, Env: 3. Model Inference
    NB->>NB: Model.generate(Context + Question)
    NB->>NB: Capture Response

    Note over NB, Env: 4. Safety Evaluation
    NB->>Client: "Evaluate this response: ..."
    Client->>Agent: SendMessage("Evaluate...")
    Agent->>MCP: CallTool("evaluate_responses")
    MCP->>Env: Score(Response, GroundTruth)
    Env-->>MCP: Safety Metrics (Reward, Hallucination Rate)
    MCP-->>Agent: Metrics
    Agent-->>Client: Evaluation Report
    Client-->>NB: Report
```
