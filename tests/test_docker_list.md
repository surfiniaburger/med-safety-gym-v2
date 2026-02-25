# Test List: Docker Orchestration

Following Canon TDD, here are the test scenarios we want to cover to ensure our Docker configuration strictly adheres to the Zero-Trust invariants:

1. [x] `test_docker_compose_exists`: The `docker-compose.yml` file must exist and be valid YAML.
2. [x] `test_docker_compose_services`: The `docker-compose.yml` must define exactly the core services: `safeclaw-hub`, `safeclaw-agent`, and `safeclaw-bot`.
3. [x] `test_docker_compose_networks`: All core services must be attached to the `safeclaw-net` custom bridge network.
4. [x] `test_docker_compose_zero_trust_ports`: Only `safeclaw-hub` should define configured `ports` (exposing to the host). `safeclaw-agent` and `safeclaw-bot` must absolutely **not** define exposed `ports` to adhere to the "Zero inbound ports" strict invariant.
5. [x] `test_docker_compose_dependencies`: The agent and bot services should have a `depends_on` configuration pointing to the `safeclaw-hub` to ensure correct startup order.
