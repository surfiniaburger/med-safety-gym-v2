
import json
import argparse
from typing import Optional
from sqlalchemy import create_engine, text

def export_results(connection_string: str, output_path: str, session_id: Optional[str] = None):
    """
    Exports evaluation results from the database to a JSON file.
    
    Args:
        connection_string: Database connection string (e.g. postgresql://...)
        output_path: Path to save result.json
        session_id: Optional session ID to filter by
    """
    engine = create_engine(connection_string)
    
    query = "SELECT * FROM neural_snapshots"
    params = {}
    if session_id:
        query += " WHERE session_id = :session_id"
        params["session_id"] = session_id
        
    query += " ORDER BY step ASC"
    
    with engine.connect() as conn:
        result = conn.execute(text(query), params).mappings().all()
        
    # Format for Gauntlet/Analysis
    snapshots = []
    for row in result:
        # Handle SQLite JSON string vs Postgres JSON dict
        scores = row["scores"]
        if isinstance(scores, str):
            scores = json.loads(scores)
            
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
            
        snapshots.append({
            "step": row["step"],
            "session_id": row["session_id"],
            "scores": scores,
            "metadata": metadata
        })
        
    with open(output_path, "w") as f:
        json.dump(snapshots, f, indent=2)
        
    print(f"Exported {len(snapshots)} snapshots to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export med-safety-eval results")
    parser.add_argument("--db", required=True, help="Database connection string")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--session", help="Session ID filter")
    
    args = parser.parse_args()
    export_results(args.db, args.out, args.session)
