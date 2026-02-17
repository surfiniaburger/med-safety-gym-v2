# Deployment Playbook: Scaling SafeClaw with Ephemeral Runners

This playbook details how to deploy SafeClaw in a production environment using Docker and GitHub Actions, ensuring that security invariants are maintained at scale.

---

## 1. The Container Strategy

We utilize two distinct Docker images to enforce the Governor/Runner split.

### A. `Dockerfile.hub` (The Governor)
- **Base**: `python:3.11-slim` (or standard `node` for a JS hub)
- **Role**: Long-lived, persistent service handling Telegram/Web traffic.
- **Exposure**: Binds to public internet (port 80/443).
- **Security**: Hardened API with rate limiting and biometric token validation.

### B. `Dockerfile.a2a` (The Runner)
- **Base**: `mcp-sdk-python` / `uv`
- **Role**: **Ephemeral**. Spawned per session or per high-risk task.
- **Exposure**: **No inbound ports**. Only outbound to the Hub for coordination and to MCP servers.
- **Persistence**: ❌ None. Filesystem is wiped on container exit.

---

## 2. GitHub Actions Workflow: "Verified Build"

Every commit must pass the "Gauntlet" before deployment.

```yaml
name: SafeClaw CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run Safety Hardening Tests
        run: uv run pytest tests/test_manifest_interceptor.py tests/test_jit_escalation.py
      - name: Verify Invariant Parity
        run: uv run python scripts/verify_invariants.py

  build:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Build & Push Ephemeral Runner
        run: |
          docker build -t gcr.io/safeclaw/runner:latest -f Dockerfile.a2a .
          docker push gcr.io/safeclaw/runner:latest
```

---

## 3. Production Hardening Checklist

### Environment Variables
- `SAFECLAW_GOVERNOR_URL`: The Hub address.
- `ADMIN_ESCALATION_SECRET`: Used for cross-service signing of HITL tokens.
- `MANIFEST_DATABASE_URL`: Where tiered tool permissions are stored.

### Networking (Zero Trust)
- [ ] Runners deployed in a **private subnet**.
- [ ] Governor uses **mTLS** or signed headers (the "Escalation Secret") to talk to Runners.
- [ ] Egress filtering on Runners (deny all, allow-list specific MCP domains).

### Resource Limits
- **CPU/Mem per Runner**: 0.5 vCPU / 512MB RAM.
- **Kill switch**: Auto-terminate Runners idle for > 15 minutes.

---

## 4. Disaster Recovery

- **The Red Button (Revoke All)**: A single API call to the Hub revokes all active escalation tokens. Even if a Runner is compromised, it immediately loses its "Admin" permissions.
- **Manifest Rollback**: Manifests are versioned in Git. If a new tool introduction breaks a security invariant, we revert the manifest, and the Hub instantly stops granting access to that tool.

---

> [!TIP]
> Use Render's "Background Worker" for Runners and "Web Service" for the Hub. This allows scaling the workers (the heavy LLM processing) independently of the controller.
