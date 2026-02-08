import os
import json
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from .utils.logging import get_logger

logger = get_logger(__name__)

class DataAgent:
    """
    UI Data Agent: Aggregates evaluation results for the Gauntlet UI.
    Responsible for:
    1. Finding SFT vs GRPO session pairs.
    2. Selecting 'Interesting Indices' based on reward deltas and failures.
    3. Formatting data for the React-based Gauntlet dashboard.
    """
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            logger.warning("DATABASE_URL not set. DataAgent will run in mock mode.")
            self.engine = None
        else:
            if self.db_url.startswith("postgres://"):
                self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)
            self.engine = create_engine(self.db_url)

    def get_interesting_indices(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Queries the database for snapshots in a session and scores them for 'interest'.
        Interest is defined by:
        - Negative rewards (failures)
        - Specific metric flags (hallucination, inconsistency)
        - Format errors
        """
        if not self.engine:
            return []

        query = text("""
            SELECT step, scores, metadata 
            FROM neural_snapshots 
            WHERE session_id = :session_id
            ORDER BY step ASC
        """)

        interesting_indices = []
        with self.engine.connect() as conn:
            result = conn.execute(query, {"session_id": session_id})
            for row in result:
                step = row[0]
                scores = row[1]
                meta = row[2]
                
                # Robust JSON parsing for SQLite compatibility
                if isinstance(scores, str):
                    scores = json.loads(scores)
                if isinstance(meta, str):
                    meta = json.loads(meta)
                
                # Scoring Rubric for "Interest"
                interest_score = 0
                
                # 1. Negative total reward is interesting
                root_score = scores.get("root", 0.0)
                if root_score < 0:
                    interest_score += 30
                
                # 2. Hallucinations are high priority
                if scores.get("grounding", 0.0) < 0:
                    interest_score += 50
                    
                # 3. Inconsistencies (Mental Model mismatch)
                if scores.get("inconsistency", 0.0) < 0:
                    interest_score += 40
                
                # 4. Format errors (LLM following instructions)
                if "format" in scores and scores.get("format", 0.0) <= 0:
                    interest_score += 20

                if interest_score > 0:
                    interesting_indices.append({
                        "step": step,
                        "interest_score": interest_score,
                        "scores": scores,
                        "metadata": meta
                    })

        # Sort by interest score descending
        interesting_indices.sort(key=lambda x: x["interest_score"], reverse=True)
        return interesting_indices[:limit]

    def pair_sft_and_grpo(self, sft_session: str, grpo_session: str) -> List[Dict[str, Any]]:
        """
        Compares an SFT baseline with a GRPO run.
        Returns pairs of indices where delta is significant.
        """
        if not self.engine:
            return []

        # This logic assumes steps align 1:1 by index which is true for standard evals
        sft_data = self.get_session_snapshots(sft_session)
        grpo_data = self.get_session_snapshots(grpo_session)

        paired = []
        # Index data by step for easy lookup
        sft_map = {d["step"]: d for d in sft_data}
        
        for g in grpo_data:
            step = g["step"]
            if step in sft_map:
                s = sft_map[step]
                delta = g["scores"].get("root", 0) - s["scores"].get("root", 0)
                
                # Only include if there is a delta or both are interesting
                if abs(delta) > 5.0 or g["scores"].get("root", 0) < 0:
                    paired.append({
                        "step": step,
                        "sft": s,
                        "grpo": g,
                        "delta": delta
                    })
        
        return paired

    def get_evolution_data(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Finds the latest SFT and GRPO sessions for a task_id and pairs them.
        """
        if not self.engine: return []
        
        # Simple scan for session types (can be optimized with JSON indexing in PG)
        query = text("SELECT DISTINCT session_id, metadata FROM neural_snapshots")
        sft_session = None
        grpo_session = None
        
        with self.engine.connect() as conn:
            result = conn.execute(query)
            for row in result:
                sid, meta = row[0], row[1]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                
                if meta.get("task_id") == task_id:
                    run_type = meta.get("run_type")
                    if run_type == "sft":
                        sft_session = sid
                    elif run_type == "grpo":
                        grpo_session = sid
                        
        if sft_session and grpo_session:
            return self.pair_sft_and_grpo(sft_session, grpo_session)
        
        return []

    def get_session_snapshots(self, session_id: str) -> List[Dict[str, Any]]:
        """Utility to get all snapshots for a session."""
        if not self.engine: return []
        query = text("SELECT step, scores, metadata FROM neural_snapshots WHERE session_id = :session_id")
        with self.engine.connect() as conn:
            result = conn.execute(query, {"session_id": session_id})
            items = []
            for r in result:
                step, scores, meta = r[0], r[1], r[2]
                if isinstance(scores, str): scores = json.loads(scores)
                if isinstance(meta, str): meta = json.loads(meta)
                items.append({"step": step, "scores": scores, "metadata": meta})
            return items

    def queue_command(self, session_id: str, command: Dict[str, Any]):
        """Queues a command for a session."""
        if not self.engine: return
        import time
        # Upsert command (replace existing if any)
        # Using simple delete+insert for compatibility
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM gauntlet_commands WHERE session_id = :sid"), {"sid": session_id})
            conn.execute(
                text("INSERT INTO gauntlet_commands (session_id, command, timestamp) VALUES (:sid, :cmd, :ts)"),
                {"sid": session_id, "cmd": json.dumps(command), "ts": time.time()}
            )

    def pop_command(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves and clears a command for a session."""
        if not self.engine: return None
        
        command = None
        with self.engine.begin() as conn:
            # Select first
            result = conn.execute(
                text("SELECT command FROM gauntlet_commands WHERE session_id = :sid"),
                {"sid": session_id}
            ).fetchone()
            
            if result:
                cmd_str = result[0]
                command = json.loads(cmd_str) if isinstance(cmd_str, str) else cmd_str
                # Clear command
                conn.execute(text("DELETE FROM gauntlet_commands WHERE session_id = :sid"), {"sid": session_id})
                
        return command
