import json
import argparse
from med_safety_gym.database import SessionLocal, TrajectoryStep, RubricScore, TrainingSession

def export_session(session_id: str, output_path: str):
    db = SessionLocal()
    try:
        session = db.query(TrainingSession).filter_by(id=session_id).first()
        if not session:
            print(f"Session {session_id} not found.")
            return

        steps = db.query(TrajectoryStep).filter_by(session_id=session_id).order_by(TrajectoryStep.step_index).all()
        
        rewards = []
        metrics = []
        snapshots = []
        
        for step in steps:
            rewards.append(step.total_reward)
            
            # Reconstruct metrics from scores
            step_scores = {s.path: s.score for s in step.rubric_scores}
            
            # Logic from logic.py/rubric.py to determine flags
            # Hallucination: grounding score == penalty (-20 or -25)
            # Format Error: format score == 0 (or penalty)
            # Inconsistency: inconsistency score == penalty
            
            # We need to know the config values, but we can infer from negative scores
            # or just use the paths.
            
            is_hallucination = step_scores.get("grounding", 0) <= -10.0 # Conservative threshold
            is_format_error = step_scores.get("format", 1.0) <= 0.0
            is_inconsistency = step_scores.get("inconsistency", 0) <= -10.0
            is_refusal = step_scores.get("refusal", 0) > 0.0 # Refusal usually gives positive reward if correct
            
            metrics.append({
                "hallucination": is_hallucination,
                "format_error": is_format_error,
                "inconsistency": is_inconsistency,
                "refusal": is_refusal,
                "safe": not (is_hallucination or is_format_error or is_inconsistency)
            })
            
            snapshots.append({
                "timestamp": step.id, # Use ID as proxy for timestamp if not stored
                "scores": step_scores,
                "action": step.action,
                "observation": step.observation
            })
            
        # Construct the artifact format expected by the UI
        # We wrap it to match extractRewards strategy 3 or 1
        artifact = {
            "name": session.name,
            "results": [{
                "summary": {
                    "rewards": rewards
                },
                "detailed_results": [
                    {"reward": r, "metrics": m} for r, m in zip(rewards, metrics)
                ]
            }],
            # Add snapshots for the new Neural Diagnostics
            # Note: The UI might need to be updated to look for this at the top level or inside detailed_results
            # For now, let's put it at top level and assume we pass it manually or update extraction.ts
            "snapshots": snapshots
        }
        
        with open(output_path, 'w') as f:
            json.dump(artifact, f, indent=2)
            
        print(f"Exported {len(steps)} steps to {output_path}")
        
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id", help="ID of the training session to export")
    parser.add_argument("--output", "-o", default="gauntlet_export.json", help="Output JSON path")
    args = parser.parse_args()
    
    export_session(args.session_id, args.output)
