from typing import Dict, List, Any, Optional, Protocol, runtime_checkable
import json
import time
import threading
import requests
import requests.exceptions
from .rubric import Rubric
from .schemas import NeuralSnapshot
from .utils.logging import get_logger

logger = get_logger(__name__)



@runtime_checkable
class DataSink(Protocol):
    """Interface for where observability data is sent."""
    def emit(self, snapshot: NeuralSnapshot) -> None:
        ...

class ConsoleSink:
    """Simple sink that prints to console."""
    def emit(self, snapshot: NeuralSnapshot) -> None:
        print(f"[RubricObserver] {snapshot.model_dump_json(indent=2)}")

class WandBSink:
    """Sink that logs to Weights & Biases."""
    def __init__(self, project: str = "med-safety-gym", config: Optional[Dict] = None):
        try:
            import wandb
            self.wandb = wandb
            if not self.wandb.run:
                self.wandb.init(project=project, config=config)
        except ImportError:
            logger.warning("wandb not installed. WandBSink will be a no-op.")
            self.wandb = None

    def emit(self, snapshot: NeuralSnapshot) -> None:
        if self.wandb and self.wandb.run:
            # Flatten scores for WandB: "scores.grounding" -> "rubric/grounding"
            log_data = {f"rubric/{k}": v for k, v in snapshot.scores.items()}
            # WandB handles step automatically if we log strictly sequentially, 
            # or we can pass step explicitly
            self.wandb.log(log_data, step=snapshot.step)

class DatabaseSink:
    """Sink that writes to a PostgreSQL database (Stub)."""
    def __init__(self, connection_string: Optional[str] = None, table_name: str = "neural_snapshots"):
        import os
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        self.table_name = table_name
        self.engine = None
        self.snapshots_table = None
        
        # SQLAlchemy 1.4+ requires postgresql:// instead of postgres://
        if self.connection_string and self.connection_string.startswith("postgres://"):
            self.connection_string = self.connection_string.replace("postgres://", "postgresql://", 1)
            logger.info("Normalized connection string to use 'postgresql://' protocol.")

        try:
            from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, JSON
            from urllib.parse import urlparse
            
            if not self.connection_string:
                logger.info("DATABASE_URL not found. DatabaseSink will be a no-op.")
                return

            # Sanitize URL for logging (hide password)
            parsed = urlparse(self.connection_string)
            sanitized_url = f"{parsed.scheme}://{parsed.username}:***@{parsed.hostname}:{parsed.port}{parsed.path}"
            logger.info(f"Attempting to connect to database: {sanitized_url}")

            self.engine = create_engine(self.connection_string)
            metadata = MetaData()
            
            # Define schema
            self.snapshots_table = Table(
                self.table_name, metadata,
                Column('id', Integer, primary_key=True),
                Column('session_id', String, index=True),
                Column('step', Integer, index=True),
                Column('scores', JSON), # Store flattened scores
                Column('metadata', JSON)
            )
            
            # Create table if not exists
            metadata.create_all(self.engine)
            logger.info("Database connection established and schema verified.")
            
        except ImportError:
            logger.info("sqlalchemy not installed. DatabaseSink will be a no-op.")
        except Exception as e:
            error_msg = str(e)
            hint = ""
            if "could not translate host name" in error_msg and "@" in error_msg:
                hint = "\nHINT: Your password may contain special characters (like '@' or ':'). " \
                       "Please ensure your DATABASE_URL is properly URL-encoded. " \
                       "You can use 'urllib.parse.quote_plus' for the password part."
            elif "Cannot assign requested address" in error_msg or "6432" in error_msg:
                hint = "\nHINT: This often indicates an IPv6 connection attempt in an IPv4-only environment. " \
                       "If using Supabase, try using the Transaction Pooler (Session Pooler) hostname " \
                       "and port (6543 or 5432) which supports IPv4."
            
            logger.warning(f"Database connection failed: {error_msg}{hint}. DatabaseSink will be a no-op.")

    def emit(self, snapshot: NeuralSnapshot) -> None:
        if self.snapshots_table is not None and self.engine is not None:
            try:
                from sqlalchemy import insert
                stmt = insert(self.snapshots_table).values(
                    session_id=snapshot.session_id,
                    step=snapshot.step,
                    scores=snapshot.scores,
                    metadata=snapshot.metadata
                )
                with self.engine.begin() as conn:
                    conn.execute(stmt)
            except Exception as e:
                logger.error(f"Error writing to database: {e}")

class WebsocketSink:
    """Sink that sends snapshots to a Gauntlet UI via a broadcast server."""
    def __init__(self, session_id: str, base_url: Optional[str] = None):
        import os
        self.session_id = session_id
        # Source from env, then fall back to Render Hub, then localhost
        self.base_url = base_url or os.getenv("GAUNTLET_HUB_URL", "https://med-safety-hub.onrender.com")
        self.url = f"{self.base_url}/gauntlet/stream/{session_id}"

    def emit(self, snapshot: NeuralSnapshot) -> None:

        def _post_in_thread():
            import time
            max_retries = 3
            backoff = 1.0 # Initial backoff for Render cold start
            
            for attempt in range(max_retries):
                try:
                    # We use a simple POST to the broadcast endpoint
                    response = requests.post(self.url, json=snapshot.model_dump(mode='json'), timeout=2.0)
                    if response.status_code == 200:
                        return
                    # If not 200, maybe it's still waking up (e.g. 503/502/504)
                    logger.debug(f"WebsocketSink attempt {attempt+1} failed with status {response.status_code}")
                except Exception as e:
                    logger.debug(f"WebsocketSink attempt {attempt+1} error: {e}")
                
                # Exponential backoff
                time.sleep(backoff)
                backoff *= 2
        
        # Run the blocking request in a separate thread to avoid blocking the event loop.
        thread = threading.Thread(target=_post_in_thread, daemon=True)
        thread.start()


class RubricObserver:
    """
    Observes a Rubric hierarchy and aggregates scores into snapshots.
    """
    def __init__(self, root_rubric: Rubric, sinks: List[DataSink], session_id: str = "default_session", pause_on_indices: Optional[List[int]] = None, base_metadata: Optional[Dict[str, Any]] = None):
        self.root_rubric = root_rubric
        self.sinks = sinks
        self.session_id = session_id
        self._step_count = 0
        self.pause_on_indices = pause_on_indices or []
        self.base_metadata = base_metadata or {}  # Store metadata to merge with each snapshot
        self._resume_event = threading.Event()
        self._resume_event.set() # Unblocked by default
        self.is_paused = False
        self._setup_hooks()
        self._command_thread = None
        self._stop_listening = threading.Event()

    def _setup_hooks(self):
        """Attaches post-forward hooks to all rubrics in the hierarchy."""
        for path, rubric in self.root_rubric.named_rubrics():
            # We use a closure to capture the path
            def hook_factory(p):
                def hook(r, action, observation, score):
                    self._on_score(p, action, observation, score)
                return hook
            
            rubric.register_forward_hook(hook_factory(path))

    def _on_score(self, path: str, action: Any, observation: Any, score: float):
        """Called whenever a rubric in the hierarchy produces a score."""
        # In a real implementation, we might want to buffer these 
        # or only emit when the root rubric finishes.
        # For now, we'll emit a snapshot if it's the root rubric.
        if path == "":
            self._step_count += 1
            
            # Check for Partial Stop (Intervention required)
            should_pause = (self._step_count - 1) in self.pause_on_indices
            
            if should_pause:
                self.is_paused = True
                self._resume_event.clear()
                logger.info(f"⏸️ Partial Stop triggered at index {self._step_count - 1}. Waiting for user intervention...")
                self._start_command_listener()

            snapshot = self.capture_snapshot(action, observation)
            for sink in self.sinks:
                sink.emit(snapshot)

            if should_pause:
                # Block here until resume() is called (likely via WebSocket/API)
                self._resume_event.wait()
                self.is_paused = False
                self._stop_command_listener()
                logger.info(f"▶️ Resuming evaluation from index {self._step_count - 1}")

    def _start_command_listener(self):
        """Starts a background thread to poll for RESUME commands from the Hub."""
        if self._command_thread and self._command_thread.is_alive():
            return
            
        self._stop_listening.clear()
        self._command_thread = threading.Thread(target=self._poll_for_commands, daemon=True)
        self._command_thread.start()

    def _stop_command_listener(self):
        """Stops the command listener thread."""
        self._stop_listening.set()

    def _poll_for_commands(self):
        """Polls the Gauntlet Hub for commands (RESUME, TWEAK)."""
        # Find the hub URL from the sinks
        hub_url = None
        for sink in self.sinks:
            if hasattr(sink, 'base_url'):
                hub_url = sink.base_url
                break
        
        if not hub_url:
            logger.warning("No Hub URL found in sinks. Command listener disabled.")
            return

        command_url = f"{hub_url}/gauntlet/command/{self.session_id}"
        
        while not self._stop_listening.is_set():
            try:
                response = requests.get(command_url, timeout=5.0)
                if response.status_code == 200:
                    cmd_data = response.json()
                    action = cmd_data.get("action")
                    
                    if action == "RESUME":
                        logger.info("📡 Received RESUME command from Hub.")
                        self.resume()
                        break
                    elif action == "TWEAK":
                        tweak_data = cmd_data.get("tweak", {})
                        logger.info(f"📡 Received TWEAK command: {tweak_data}")
                        # Apply tweak to root rubric if it supports it
                        if hasattr(self.root_rubric, 'update_config'):
                            self.root_rubric.update_config(tweak_data)
                        
                        # Auto-resume after tweak? For now, we wait for RESUME
                        # self.resume()
                
            except requests.exceptions.RequestException:
                pass # Silent retry
            except Exception as e:
                logger.debug(f"Command listener error: {e}")
                
            time.sleep(2.0)

    def resume(self):
        """Unblocks the evaluation loop."""
        self._resume_event.set()

    def capture_snapshot(self, action: Any = None, observation: Any = None) -> NeuralSnapshot:
        """Traverses the rubric tree and captures all current scores."""
        from .logic import generate_safety_challenge
        
        scores = {}
        for path, rubric in self.root_rubric.named_rubrics():
            # We assume last_score is set by the rubric logic or hook
            scores[path or "root"] = getattr(rubric, "last_score", 0.0)
            
        challenge = None
        if self.is_paused:
            # Create a shallow dict for the generator to avoid circular dependencies if any
            temp_snap = {
                "scores": scores,
                "metadata": {
                    "action": str(action) if action is not None else "",
                    "observation": str(observation) if observation is not None else ""
                }
            }
            challenge = generate_safety_challenge(temp_snap)
        
        # Merge base_metadata (task_id, run_type, etc.) with per-snapshot metadata
        snapshot_metadata = {
            **self.base_metadata,  # Global metadata (task_id, run_type, model, etc.)
            "action": str(action) if action is not None else "",
            "observation": str(observation) if observation is not None else ""
        }
            
        return NeuralSnapshot(
            session_id=self.session_id,
            step=self._step_count,
            scores=scores,
            is_paused=self.is_paused,
            challenge=challenge,
            metadata=snapshot_metadata
        )
