from typing import Dict, List, Any, Optional, Protocol, runtime_checkable
import json
import time
import threading
import requests
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
        
        try:
            from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, JSON
            
            self.engine = create_engine(connection_string)
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
            
        except ImportError:
            logger.warning("sqlalchemy not installed. DatabaseSink will be a no-op.")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}. DatabaseSink will be a no-op.")

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
    def __init__(self, root_rubric: Rubric, sinks: List[DataSink], session_id: str = "default_session"):
        self.root_rubric = root_rubric
        self.sinks = sinks
        self.session_id = session_id
        self._step_count = 0
        self._setup_hooks()

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
            snapshot = self.capture_snapshot(action, observation)
            for sink in self.sinks:
                sink.emit(snapshot)

    def capture_snapshot(self, action: Any = None, observation: Any = None) -> NeuralSnapshot:
        """Traverses the rubric tree and captures all current scores."""
        scores = {}
        for path, rubric in self.root_rubric.named_rubrics():
            # We assume last_score is set by the rubric logic or hook
            scores[path or "root"] = getattr(rubric, "last_score", 0.0)
            
        return NeuralSnapshot(
            session_id=self.session_id,
            step=self._step_count,
            scores=scores,
            metadata={
                "action": str(action) if action is not None else "",
                "observation": str(observation) if observation is not None else ""
            }
        )
