# Data Agent Observability Layer: Implementation Plan

This plan outlines a modular, TDD-driven approach to building the **Data Agent Observability Layer**. Following the "Work in Small Steps" principle, each phase is a stable, testable increment suitable for PyPI releases.

## Tech Stack Strategy
*   **Database**: PostgreSQL (SQLAlchemy for provider-agnostic scaling).
*   **Real-time Streaming**: FastAPI WebSockets.
*   **Observability**: WandB (Training) + Custom Rubric Hooks (Gauntlet UI).
*   **Abstraction**: Decoupled "Data Sinks" for plug-and-play provider support.

---

## Phase 1: Rubric Core Completion (The Foundation)
**Goal**: Complete the RFC 004 specification to allow the Data Agent to "see" into the neural trajectory.

*   **1.1: Pre-Forward Hooks**: Implement `register_forward_pre_hook` in the `Rubric` base class to capture raw inputs.
*   **1.2: Nested Path Access**: Implement `get_rubric(path: str)` for granular introspection (e.g., `rubric.get_rubric("grounding.fuzzy")`).
*   **1.3: Standardized LLMJudge**: Move `LLMJudge` into the core library as a reusable block.
*   **TDD Action**: Write tests verifying pre-hooks capture `action` before `forward()` execution.

## Phase 2: The "Data Agent" Abstraction (The Observer)
**Goal**: Create a decoupled system that "listens" to rubrics and prepares data for the Gauntlet.

*   **2.1: RubricObserver Interface**: Define a stateless observer that can be attached to any Rubric.
*   **2.2: Component Score Aggregator**: A utility that flattens the `named_rubrics()` tree into a "Neural Snapshot" (JSON).
*   **2.3: Data Sinks**: Implement `ConsoleSink`, `WandBSink`, and `DatabaseSink` stubs.
*   **TDD Action**: Verify that attaching an observer to `DIPGRubric` produces a valid JSON snapshot of all sub-scores.

## Phase 3: Real-time Pipeline (The Stream)
**Goal**: Enable "Gauntlet Film" capability by streaming snapshots from the TPU/Server to the UI.

*   **3.1: Trajectory Schema**: Design a Postgres schema for `episode_id`, `step_index`, `rubric_path`, and `score`.
*   **3.2: FastAPI WebSocket Broadcast**: Create `/ws/gauntlet/{session_id}` for real-time snapshot broadcasting.
*   **3.3: Async Buffer**: Implement background batching to handle high-frequency TPU training steps.
*   **TDD Action**: Verify WebSocket broadcast triggers upon `rubric(action, obs)` calls.

## Phase 4: Gauntlet UI Integration (The X-Ray View)
**Goal**: Update the React 3D UI to consume the live stream and provide granular diagnostics.

*   **4.1: Live Stream Hook**: Add `useGauntletStream` to the frontend.
*   **4.2: Neural Diagnostics Panel**: Create a side-panel displaying the "Neural Snapshot" for selected nodes.
*   **4.3: Visual State Mapping**: Map specific rubric failures to 3D effects (e.g., `FormatRubric` failure = jagged path).
*   **TDD Action**: Mock WebSocket stream and verify `GauntletView` updates visual state.

## Phase 5: Training "Film" Recording (The Replay)
**Goal**: Record GRPO/SFT training runs for "playback" and post-mortem analysis.

*   **5.1: DIPGRaxReward Integration**: Attach `DatabaseSink` to the training reward function.
*   **5.2: Session Management**: Enable named training runs and trajectory retrieval.
*   **5.3: Export to Gauntlet**: Utility to generate `results.json` directly from the database.
*   **TDD Action**: Verify 5-step dummy training loop results in 5 complete snapshots in the DB.

## Phase 6: Security Hardening (The Vault)
**Goal**: “Air-tight” security for data at rest and in transit, following stricter validation protocols.

*   **6.1: Secure Sinks**: Implement `EncryptedDatabaseSink` and enforce `wss://` for WebSockets.
*   **6.2: Validation Layer**: strict Pydantic models for all `StepResult` and `NeuralSnapshot` data.
*   **6.3: Audit Logging**: Record all access to the observability stream.
*   **TDD Action**: Verify that invalid JSON payloads are rejected with 422 Unprocessable Entity.
