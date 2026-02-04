from typing import Dict, Any
from sqlalchemy.orm import Session
from med_safety_eval.observer import DataSink
from med_safety_eval.schemas import NeuralSnapshot

from med_safety_gym.database import TrajectoryStep, RubricScore, SessionLocal, TrainingSession
import json

class SQLAlchemySink:
    """
    Writes rubric snapshots to a SQL database.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._ensure_session_exists()

    def _ensure_session_exists(self):
        db = SessionLocal()
        try:
            session = db.query(TrainingSession).filter_by(id=self.session_id).first()
            if not session:
                session = TrainingSession(id=self.session_id, name=f"Run {self.session_id}")
                db.add(session)
                db.commit()
        finally:
            db.close()

    def emit(self, snapshot: NeuralSnapshot) -> None:

        db = SessionLocal()
        try:
            # Create Step
            # We assume snapshot has 'step_index' if provided, else auto-increment logic might be needed
            # For now, let's just append.
            
            # Extract action/obs if they were captured (RubricObserver might need to be updated to capture them)
            # The current RubricObserver only captures scores.
            # To fully implement the "Film", we need action/obs.
            # Let's assume the snapshot might contain them if the observer was configured to capture them.
            
            step = TrajectoryStep(
                session_id=self.session_id,
                step_index=snapshot.step,
                total_reward=snapshot.scores.get("root", 0.0),
                action=str(snapshot.metadata.get("action", "")),
                observation=str(snapshot.metadata.get("observation", ""))
            )
            db.add(step)
            db.flush() # Get ID

            # Create Rubric Scores
            for path, score in snapshot.scores.items():

                r_score = RubricScore(
                    step_id=step.id,
                    path=path,
                    score=float(score) if score is not None else 0.0
                )
                db.add(r_score)
            
            db.commit()
        except Exception as e:
            print(f"Error writing to DB: {e}")
            db.rollback()
        finally:
            db.close()
