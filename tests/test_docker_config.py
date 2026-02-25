import os
import yaml
import pytest

DOCKER_COMPOSE_PATH = "docker-compose.yml"

def parse_docker_compose():
    if not os.path.exists(DOCKER_COMPOSE_PATH):
        return None
    with open(DOCKER_COMPOSE_PATH, "r") as f:
        return yaml.safe_load(f)

def test_docker_compose_exists():
    """docker-compose.yml must exist and be valid YAML."""
    assert os.path.exists(DOCKER_COMPOSE_PATH), "docker-compose.yml file not found"
    data = parse_docker_compose()
    assert data is not None, "Failed to parse docker-compose.yml"
    assert "services" in data, "docker-compose.yml must define 'services'"

def test_docker_compose_services():
    """docker-compose.yml must define exactly the core services: safeclaw-hub, safeclaw-agent, and safeclaw-bot."""
    data = parse_docker_compose()
    services = data.get("services", {})
    
    expected_services = ["safeclaw-hub", "safeclaw-agent", "safeclaw-bot"]
    for svc in expected_services:
        assert svc in services, f"Missing required service: {svc}"

def test_docker_compose_networks():
    """All core services must be attached to the safeclaw-net custom bridge network."""
    data = parse_docker_compose()
    services = data.get("services", {})
    
    for svc_name in ["safeclaw-hub", "safeclaw-agent", "safeclaw-bot"]:
        svc_config = services.get(svc_name, {})
        networks = svc_config.get("networks", [])
        assert "safeclaw-net" in networks, f"Service {svc_name} missing 'safeclaw-net' network"

def test_docker_compose_zero_trust_ports():
    """Only safeclaw-hub should define configured ports. Agent and bot must not have inbound ports."""
    data = parse_docker_compose()
    services = data.get("services", {})
    
    # Hub must expose 8080
    hub_ports = services.get("safeclaw-hub", {}).get("ports", [])
    assert any("8080" in str(p) for p in hub_ports), "safeclaw-hub must expose port 8080"
    
    # Agent and Bot must NOT expose ports
    assert "ports" not in services.get("safeclaw-agent", {}), "safeclaw-agent must not expose ports (Zero-Trust)"
    assert "ports" not in services.get("safeclaw-bot", {}), "safeclaw-bot must not expose ports (Zero-Trust)"

def test_docker_compose_dependencies():
    """Agent and bot services should depend on safeclaw-hub."""
    data = parse_docker_compose()
    services = data.get("services", {})
    
    for svc_name in ["safeclaw-agent", "safeclaw-bot"]:
        depends_on = services.get(svc_name, {}).get("depends_on", {})
        # depends_on can be a list or a dict in compose files
        if isinstance(depends_on, dict):
            assert "safeclaw-hub" in depends_on, f"{svc_name} missing dependency on safeclaw-hub"
        elif isinstance(depends_on, list):
            assert "safeclaw-hub" in depends_on, f"{svc_name} missing dependency on safeclaw-hub"
        else:
            pytest.fail(f"Invalid depends_on format for {svc_name}")
