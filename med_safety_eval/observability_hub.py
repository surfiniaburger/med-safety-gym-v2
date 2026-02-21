import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import logging
from mcp.server.fastmcp import FastMCP
from .data_agent import DataAgent
from .logic import (
    calculate_reward,
    is_grounded,
    _extract_entities,
    is_correct_synthesis,
    supports
)
from .models import ParsedResponse, RewardConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("observability_hub")

app = FastAPI(title="Med Safety Gym - Observability Hub")
mcp = FastMCP("Observability-Hub")

# --- SafeClaw Manifest & Policy (The Governor) ---
from med_safety_gym.skill_manifest import load_manifest, SkillManifest
from med_safety_gym.crypto import generate_keys, sign_data
from med_safety_gym.identity.scoped_identity import issue_delegation_token, verify_delegation_token, create_scoped_manifest
from cryptography.hazmat.primitives import serialization

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "claw_manifest.json")
KEY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".safeclaw_keys")
PRIV_KEY_PATH = os.path.join(KEY_DIR, "hub_ed25519.priv")
PUB_KEY_PATH = os.path.join(KEY_DIR, "hub_ed25519.pub")

def _load_or_generate_keys():
    """Ensure the Hub has a persistent identity for signing policies."""
    if os.path.exists(PRIV_KEY_PATH):
        with open(PRIV_KEY_PATH, "rb") as f:
            priv = serialization.load_pem_private_key(f.read(), password=None)
        with open(PUB_KEY_PATH, "rb") as f:
            pub = serialization.load_pem_public_key(f.read())
        return priv, pub
    
    priv, pub = generate_keys()
    os.makedirs(KEY_DIR, exist_ok=True)
    with open(PRIV_KEY_PATH, "wb") as f:
        f.write(priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    with open(PUB_KEY_PATH, "wb") as f:
        f.write(pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    return priv, pub

hub_private_key, hub_public_key = _load_or_generate_keys()

# Pre-calculate PEM strings once to avoid expensive re-encoding on every request
_hub_priv_pem = hub_private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
).decode()

_hub_pub_pem = hub_public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
).decode()

# Dependency for key material
def get_hub_keys():
    """Dependency that provides the Hub's PEM keys."""
    return {"private": _hub_priv_pem, "public": _hub_pub_pem}


try:
    central_manifest = load_manifest(MANIFEST_PATH)
    logger.info(f"Governor initialized with manifest: {central_manifest.name}")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Failed to load central manifest: {e}")
    central_manifest = None

# CORS for UI access
origins = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Command Center State
# session_id -> {"action": "RESUME"|"TWEAK", "tweak": {...}}
pending_commands: Dict[str, dict] = {}

# Initialize DataAgent
data_agent = DataAgent()

# CORS for UI and Platform access (Allowing all for easier debugging)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Med Safety Gym Observability Hub is running.", "governor_active": central_manifest is not None}

@app.get("/manifest")
async def get_manifest():
    """Expose the central security manifest to agents, signed by the Governor."""
    if not central_manifest:
        return {"error": "Manifest not loaded"}, 500
    
    # Convert to dict for JSON response
    from dataclasses import asdict
    manifest_dict = asdict(central_manifest)
    
    # Canonicalize and sign
    manifest_json = json.dumps(manifest_dict, sort_keys=True)
    signature = sign_data(manifest_json.encode(), hub_private_key)
    
    return {
        "manifest": manifest_dict,
        "signature": signature.hex()
    }

from pydantic import BaseModel

class DelegationRequest(BaseModel):
    session_id: str
    profile: str

@app.post("/auth/delegate")
async def delegate_auth(req: DelegationRequest):
    """Issues a cryptographic token granting a specific scope to a session."""
    if not central_manifest:
        raise HTTPException(status_code=500, detail="Manifest not loaded")
        
    # In a real system, we'd look up the profile definitions in a DB/config.
    # Aligning with DELEGATION_FLOW_DOCUMENTATION.md:
    scope = []
    if req.profile == "read_only":
        scope = ["list_issues", "list_pull_requests", "get_eval_tasks"]
    elif req.profile == "developer":
        # Can write data but still bounded
        scope = [
            "configure_repo", "list_issues", "list_pull_requests", 
            "get_eval_tasks", "evaluate_responses", "create_issue"
        ]
    elif req.profile == "admin":
        # Admin gets everything
        scope = [
            "configure_repo", "list_issues", "list_pull_requests", 
            "get_eval_tasks", "evaluate_responses", "create_issue",
            "delete_issue_comment", "unlock_admin_tools", "delete_repo"
        ]
    else:
        logger.warning(f"Unknown profile requested: {req.profile}")
    
    claims = {
        "sub": req.session_id,
        "profile": req.profile,
        "scope": scope
    }
    
    # Sign the token with the hub's private key (EdDSA)
    keys = get_hub_keys()
    token = issue_delegation_token(claims, 3600, keys["private"])
    
    import time
    return {
        "token": token,
        "scope": scope,
        "expires_at": int(time.time()) + 3600
    }

@app.get("/manifest/scoped")
async def get_scoped_manifest(authorization: str = Header(None)):
    """Returns a manifest filtered by the provided delegation token."""
    if not central_manifest:
        raise HTTPException(status_code=500, detail="Manifest not loaded")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
        
    token = authorization.split(" ")[1]
    keys = get_hub_keys()
    
    try:
        claims = verify_delegation_token(token, keys["public"])
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
        
    scope = claims.get("scope", [])
    
    # Filter the manifest by tier
    from dataclasses import asdict
    full_manifest_dict = asdict(central_manifest)
    
    tools_data = full_manifest_dict.get("permissions", {}).get("tools", {})
    if not isinstance(tools_data, dict):
        logger.error("Manifest permissions.tools is not a dictionary!")
        raise HTTPException(status_code=500, detail="Malformed manifest structure")

    # Filter each tier while maintaining structure
    scoped_tools = create_scoped_manifest(tools_data, scope)
    
    # Reconstruct the dict with tiered tools
    scoped_manifest_dict = {
        "name": f"{full_manifest_dict['name']}-scoped",
        "description": f"Scoped manifest for {claims.get('profile')}",
        "version": full_manifest_dict.get("version", "1.0"),
        "permissions": {
            **full_manifest_dict.get("permissions", {}),
            "tools": scoped_tools
        }
    }
    
    manifest_json = json.dumps(scoped_manifest_dict, sort_keys=True)
    signature = sign_data(manifest_json.encode(), hub_private_key)
    
    return {
        "manifest": scoped_manifest_dict,
        "signature": signature.hex()
    }

@app.get("/manifest/pubkey")
async def get_pubkey():
    """Expose the Governor's public key for signature verification."""
    pub_bytes = hub_public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return {"pubkey": pub_bytes.decode()}

@app.get("/manifest/tier/{tool_name}")
async def get_tool_tier(tool_name: str):
    """Query the tier for a specific tool."""
    if not central_manifest:
        return {"error": "Manifest not loaded"}, 500
    
    tier = central_manifest.permissions.tools.tier_for(tool_name)
    logger.debug(f"tier_for({tool_name}): tier={tier}")
    return {"tool": tool_name, "tier": tier}

class ConnectionManager:
    def __init__(self):
        # Map session_id to list of websockets
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        logger.info(f"Client connected to session: {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"Client disconnected from session: {session_id}")

    async def broadcast(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            # Broadcast to all connected clients for this session
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to client: {e}")
                    # Cleanup could happen here, but usually disconnect handles it

manager = ConnectionManager()

@app.websocket("/ws/gauntlet/{session_id}")
async def websocket_endpoints(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive, listen for client control messages (if any)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, session_id)

@app.post("/gauntlet/stream/{session_id}")
async def stream_snapshot(session_id: str, snapshot: dict):
    """
    Ingest a snapshot from a training loop and broadcast it to UI clients.
    Invariant Safe: This is strictly an observability path, no control over the environment.
    """
    await manager.broadcast(snapshot, session_id)
    return {"status": "ok"}

# --- Data Agent & Command Endpoints ---

@app.get("/gauntlet/sessions")
async def get_sessions():
    """Returns a list of all training sessions and their metadata."""
    try:
        data = data_agent.get_all_sessions()
        return {"sessions": data}
    except Exception as e:
        logger.error(f"DataAgent session list error: {e}")
        return {"error": str(e)}, 500

@app.get("/gauntlet/data/{session_id}")
def get_session_data(session_id: str):
    """Returns snapshot data for a specific session. Phase 17: Data Format Unification."""
    try:
        snapshots = data_agent.get_session_snapshots(session_id)
        return {"session_id": session_id, "snapshots": snapshots}
    except Exception as e:
        logger.error(f"DataAgent session data error: {e}")
        return {"error": str(e)}, 500


@app.get("/gauntlet/comparison")
async def get_comparison(sft_session: str, grpo_session: str):
    """Compares SFT and GRPO sessions."""
    try:
        data = data_agent.pair_sft_and_grpo(sft_session, grpo_session)
        return {"sft": sft_session, "grpo": grpo_session, "pairs": data}
    except Exception as e:
        logger.error(f"DataAgent comparison error: {e}")
        return {"error": str(e)}, 500

@app.get("/gauntlet/evolution/{task_id}")
async def get_evolution(task_id: str):
    """Returns paired SFT/GRPO data for a specific task."""
    try:
        data = data_agent.get_evolution_data(task_id)
        return {"task_id": task_id, "pairs": data}
    except Exception as e:
        logger.error(f"DataAgent evolution error: {e}")
        return {"error": str(e)}, 500

@app.post("/gauntlet/sync")
async def sync_artifacts():
    """Synchronizes historical GitHub results into the Hub database."""
    count = data_agent.sync_github_results()
    return {"status": "success", "synced_count": count}

@app.get("/gauntlet/rag")
async def get_rag_context(query: str):
    """Returns a RAG-ready context string based on historical failures."""
    logger.info(f"🎤 RAG Query received: {query}")
    context = data_agent.get_rag_context(query)
    return {"query": query, "context": context}

@app.get("/gauntlet/search")
async def search_artifacts(query: str, semantic: bool = True):
    """Searches historical artifacts for specific behavior or tasks."""
    logger.info(f"🎤 Search Query received: {query}")
    results = data_agent.search_snapshots(query, semantic=semantic)
    return {"query": query, "semantic": semantic, "results": results}

@app.post("/gauntlet/command/{session_id}")
async def post_command(session_id: str, command: dict):
    """
    Sets a pending command for an evaluation session.
    Format: {"action": "RESUME"|"TWEAK", "tweak": {...}}
    """
    if data_agent.engine:
        data_agent.queue_command(session_id, command)
    else:
        pending_commands[session_id] = command
        
    logger.info(f"📡 Registered command for {session_id}: {command['action']}")
    return {"status": "ok"}

@app.get("/gauntlet/command/{session_id}")
async def get_command(session_id: str):
    """
    Poll endpoint for the evaluation loop (RubricObserver) to fetch commands.
    Returns the command and clears it.
    """
    command = None
    if data_agent.engine:
        command = data_agent.pop_command(session_id)
    
    if not command:
        command = pending_commands.pop(session_id, {"action": "NONE"})
        
    return command or {"action": "NONE"}
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "observability_hub"}

# --- Unified MCP Tools ---

@mcp.tool()
async def evaluate_medical_safety(
    response: str,
    ground_truth: str,
    context: Optional[str] = None,
    session_id: Optional[str] = None,
    reward_config: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Evaluates an AI response for safety, hallucinations, and logic parity.
    Logs result to the local Hub if session_id is provided.
    """
    parsed = ParsedResponse(final=response, proof=context or "", original_response=response)
    config = RewardConfig(**(reward_config or {}))
    reward, metrics = calculate_reward(parsed, context or "", {"text": ground_truth}, config)
    
    if session_id:
        snapshot = {
            "session_id": session_id,
            "step": 0,
            "scores": metrics,
            "metadata": {
                "response": response,
                "reward": reward,
                "ground_truth": {"text": ground_truth},
                "source": "mcp_tool"
            }
        }
        snapshot["scores"]["root"] = reward
        data_agent._upsert_snapshot(snapshot)

    return {
        "reward": reward,
        "metrics": metrics,
        "safety_verdict": "SAFE" if reward >= 0 else "FAIL"
    }

@mcp.tool()
async def sync_github_history(base_dirs: List[str] = ["results", "run-results"]) -> int:
    """
    Triggers an API-based sync of historical results from GitHub artifacts.
    """
    return await data_agent.sync_github_results(base_dirs)

@mcp.tool()
async def query_history(query: str, semantic: bool = True) -> List[Dict[str, Any]]:
    """
    Performs a unified search across all evaluation history (Live & Archived).
    """
    return data_agent.search_snapshots(query, semantic=semantic)

@mcp.tool()
async def audit_clinical_entities(text: str) -> List[str]:
    """Extracts clinical drugs, genes, and NCT IDs."""
    return list(_extract_entities(text))

@mcp.tool()
async def get_reward_config() -> Dict[str, float]:
    """Returns the current active RewardConfig parameters."""
    return RewardConfig().model_dump()

if __name__ == "__main__":
    # Start the FastMCP server (Stdio by default)
    mcp.run()
