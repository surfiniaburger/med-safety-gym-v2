import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import logging
from .data_agent import DataAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("observability_hub")

app = FastAPI(title="Med Safety Gym - Observability Hub")

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
    return {"status": "ok", "message": "Med Safety Gym Observability Hub is running."}

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
