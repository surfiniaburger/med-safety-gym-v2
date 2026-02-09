import os
import json
import time
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
    4. Synchronizing historical GitHub artifacts into the persistent Hub layer.
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

    def sync_github_results(self, base_dirs: List[str] = ["results", "run-results"]):
        """
        Synchronizes historical GitHub artifacts into the local Hub database.
        """
        if not self.engine:
            logger.error("No database engine available for sync.")
            return

        sync_count = 0
        for dir_name in base_dirs:
            full_path = os.path.join(os.getcwd(), dir_name)
            if not os.path.exists(full_path):
                logger.warning(f"Sync directory not found: {full_path}")
                continue

            for filename in os.listdir(full_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(full_path, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            # Handle different nested structures
                            results_list = []
                            if "results" in data:
                                if isinstance(data["results"], list):
                                    for outer_res in data["results"]:
                                        if "results" in outer_res:
                                            results_list.extend(outer_res["results"])
                                        else:
                                            results_list.append(outer_res)
                            
                            for res in results_list:
                                if "summary" in res and "detailed_results" in res["summary"]:
                                    session_id = f"archived-{filename}-{sync_count}"
                                    for detail in res["summary"]["detailed_results"]:
                                        snapshot = {
                                            "session_id": session_id,
                                            "step": detail["index"],
                                            "scores": detail.get("metrics", {}),
                                            "metadata": {
                                                "response": detail.get("response", ""),
                                                "reward": detail.get("reward", 0.0),
                                                "ground_truth": detail.get("ground_truth", {}),
                                                "source": filename
                                            }
                                        }
                                        # Inject score 'root' from reward for consistency
                                        snapshot["scores"]["root"] = detail.get("reward", 0.0)
                                        
                                        self._upsert_snapshot(snapshot)
                                        sync_count += 1
                                        
                    except Exception as e:
                        logger.error(f"Failed to sync {filename}: {e}")

        logger.info(f"✅ Synced {sync_count} snapshots from GitHub artifacts.")
        return sync_count

    def _upsert_snapshot(self, snapshot: Dict[str, Any]):
        """Internal helper to upsert into neural_snapshots."""
        with self.engine.begin() as conn:
            # Check if exists
            check = conn.execute(
                text("SELECT id FROM neural_snapshots WHERE session_id = :sid AND step = :step"),
                {"sid": snapshot["session_id"], "step": snapshot["step"]}
            ).fetchone()
            
            if check:
                conn.execute(
                    text("UPDATE neural_snapshots SET scores = :scores, metadata = :meta WHERE id = :id"),
                    {
                        "scores": json.dumps(snapshot["scores"]),
                        "meta": json.dumps(snapshot["metadata"]),
                        "id": check[0]
                    }
                )
            else:
                conn.execute(
                    text("INSERT INTO neural_snapshots (session_id, step, scores, metadata) VALUES (:sid, :step, :scores, :meta)"),
                    {
                        "sid": snapshot["session_id"],
                        "step": snapshot["step"],
                        "scores": json.dumps(snapshot["scores"]),
                        "meta": json.dumps(snapshot["metadata"])
                    }
                )

    def search_snapshots(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Basic keyword search across metadata (responses and tasks).
        """
        if not self.engine: return []
        
        # PostgreSQL-friendly search if available, but staying compatible with SQLite
        # metadata is JSON, so we cast to TEXT for ILIKE search
        search_query = text("""
            SELECT session_id, step, scores, metadata 
            FROM neural_snapshots 
            WHERE LOWER(CAST(metadata AS TEXT)) LIKE LOWER(:q)
            ORDER BY id DESC
            LIMIT :limit
        """)
        
        results = []
        with self.engine.connect() as conn:
            res = conn.execute(search_query, {"q": f"%{query}%", "limit": limit})
            for row in res:
                sid, step, scores, meta = row[0], row[1], row[2], row[3]
                if isinstance(scores, str): scores = json.loads(scores)
                if isinstance(meta, str): meta = json.loads(meta)
                results.append({"session_id": sid, "step": step, "scores": scores, "metadata": meta})
        return results

    def get_rag_context(self, query: str, limit: int = 5) -> str:
        """
        Returns a formatted summary of failures related to the query for RAG.
        """
        matches = self.search_snapshots(query, limit=limit)
        if not matches:
            return "No relevant safety failures found in the Hub for this query."
            
        context = "### Relevant Safety Failures Found:\n"
        for m in matches:
            reward = m["metadata"].get("reward", 0.0)
            grounding = m["scores"].get("grounding", 1.0)
            
            context += f"\n- **Session**: {m['session_id']} (Step {m['step']})\n"
            context += f"  - **Reward**: {reward}\n"
            if grounding < 0:
                context += "  - **ISSUE**: Medical Hallucination Detected.\n"
            context += f"  - **Response Snippet**: {m['metadata'].get('response', '')[:200]}...\n"
            
        return context

    def get_interesting_indices(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Queries the database for snapshots in a session and scores them for 'interest'.
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
                
                if isinstance(scores, str): scores = json.loads(scores)
                if isinstance(meta, str): meta = json.loads(meta)
                
                interest_score = 0
                root_score = scores.get("root", 0.0)
                if root_score < 0: interest_score += 30
                if scores.get("grounding", 1.0) < 0: interest_score += 50
                if scores.get("inconsistency", 1.0) < 0: interest_score += 40
                if "format" in scores and scores.get("format", 1.0) <= 0: interest_score += 20

                if interest_score > 0:
                    interesting_indices.append({
                        "step": step,
                        "interest_score": interest_score,
                        "scores": scores,
                        "metadata": meta
                    })

        interesting_indices.sort(key=lambda x: x["interest_score"], reverse=True)
        return interesting_indices[:limit]

    def pair_sft_and_grpo(self, sft_session: str, grpo_session: str) -> List[Dict[str, Any]]:
        """Compares an SFT baseline with a GRPO run."""
        if not self.engine: return []
        sft_data = self.get_session_snapshots(sft_session)
        grpo_data = self.get_session_snapshots(grpo_session)
        paired = []
        sft_map = {d["step"]: d for d in sft_data}
        for g in grpo_data:
            step = g["step"]
            if step in sft_map:
                s = sft_map[step]
                delta = g["scores"].get("root", 0) - s["scores"].get("root", 0)
                if abs(delta) > 5.0 or g["scores"].get("root", 0) < 0:
                    paired.append({"step": step, "sft": s, "grpo": g, "delta": delta})
        return paired

    def get_evolution_data(self, task_id: str) -> List[Dict[str, Any]]:
        """Finds latest SFT and GRPO sessions for a task_id and pairs them."""
        if not self.engine: return []
        query = text("SELECT DISTINCT session_id, metadata FROM neural_snapshots")
        sft_session = None
        grpo_session = None
        with self.engine.connect() as conn:
            result = conn.execute(query)
            for row in result:
                sid, meta = row[0], row[1]
                if isinstance(meta, str): meta = json.loads(meta)
                if meta.get("task_id") == task_id:
                    run_type = meta.get("run_type")
                    if run_type == "sft": sft_session = sid
                    elif run_type == "grpo": grpo_session = sid
        if sft_session and grpo_session:
            return self.pair_sft_and_grpo(sft_session, grpo_session)
        return []

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Lists all unique sessions in the database."""
        if not self.engine: return []
        query = text("""
            SELECT DISTINCT session_id, metadata, 
            (SELECT count(*) FROM neural_snapshots s2 WHERE s2.session_id = s1.session_id) as step_count
            FROM neural_snapshots s1
        """)
        if self.engine.dialect.name == 'postgresql':
            query = text("""
                SELECT DISTINCT ON (session_id) session_id, metadata, 
                (SELECT count(*) FROM neural_snapshots s2 WHERE s2.session_id = s1.session_id) as step_count
                FROM neural_snapshots s1 ORDER BY session_id, step DESC
            """)
        sessions = []
        with self.engine.connect() as conn:
            result = conn.execute(query)
            for row in result:
                sid, meta = row[0], row[1]
                steps = row[2]
                if isinstance(meta, str): meta = json.loads(meta)
                sessions.append({"session_id": sid, "metadata": meta, "step_count": steps})
        return sessions

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
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM gauntlet_commands WHERE session_id = :sid"), {"sid": session_id})
            conn.execute(
                text("INSERT INTO gauntlet_commands (session_id, command, timestamp) VALUES (:sid, :cmd, :ts)"),
                {"sid": session_id, "cmd": json.dumps(command), "ts": time.time()}
            )

    def pop_command(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves and clears a command for a session atomically."""
        if not self.engine: return None
        if self.engine.dialect.name != 'postgresql':
            raise NotImplementedError("Atomic pop_command only supported for PostgreSQL.")
        with self.engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM gauntlet_commands WHERE session_id = :sid RETURNING command"),
                {"sid": session_id}
            ).fetchone()
            if result:
                cmd_str = result[0]
                return json.loads(cmd_str) if isinstance(cmd_str, str) else cmd_str
        return None
