# Test List: Lifecycle Telemetry

1. [x] `test_executor_logs_reaping`: `Executor` must log at INFO level when an agent is reaped due to inactivity.
2. [x] `test_agent_logs_shutdown`: `SafeClawAgent` must log at INFO level when `shutdown()` is called.
3. [x] `test_agent_logs_token_expiry`: `SafeClawAgent` must log at WARNING level when `jwt.ExpiredSignatureError` is caught during `_init_scoped_session`.
