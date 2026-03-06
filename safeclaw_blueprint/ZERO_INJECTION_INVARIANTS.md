# SafeClaw Zero-Injection Invariants

This specification defines hard constraints for safety-critical paths.

Related implementation overview:
- `safeclaw_blueprint/SAFETY_HARDENING_OVERVIEW_2026-03.md`

## Scope

These invariants apply to:
- `med_safety_gym.claw_agent.SafeClawAgent`
- `med_safety_gym.experience_server.log_contrastive_pair`
- contrastive logging and guideline distillation flows

## Invariant ZI-1: Structured Trace Only

Safety-learning paths must receive structured `semantic_trace` metadata, not raw prompt strings.

Allowed keys:
- `turn_id`
- `intent`
- `is_success`
- `failure_reason`
- `detected_entities`
- `context_entities`
- `error`

Forbidden keys (non-exhaustive):
- `raw_text`
- `raw_user_text`
- `user_prompt`
- `prompt`
- `input_text`
- `message`
- `content`

## Invariant ZI-2: Scoped Session Identity

All contrastive logs must use `session.session_id` (user + scope), not bare `user_id`.

Rationale:
- preserves scope isolation across escalation/base sessions
- prevents trace collision between scopes

## Invariant ZI-3: Training Safety Gate Contract

Guardian training must use the same entity-parity contract as runtime:
- source: `med_safety_gym.mcp_server.check_entity_parity`
- async invocation
- tuple return handling: `(is_safe, reason)`

## Executable Checks

These invariants are enforced by:
- `tests/test_zero_injection_invariants.py`
- `tests/test_experience_server.py`
