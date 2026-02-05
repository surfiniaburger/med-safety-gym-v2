import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("observability_hub")

app = FastAPI(title="Med Safety Gym - Observability Hub")

# CORS for UI access
origins = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "observability_hub"}
