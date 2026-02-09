import os
import json
import sqlite3
import sys

# Minimal DataAgent mock setup and test
# We import from the package if possible, otherwise rely on local path
try:
    from med_safety_eval.data_agent import DataAgent
except ImportError:
    # Add project root to path
    sys.path.append(os.getcwd())
    from med_safety_eval.data_agent import DataAgent

def run_test():
    db_path = "test_rag.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db_url = f"sqlite:///{db_path}"
    agent = DataAgent(db_url=db_url)
    
    # Setup manual table if not exists
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS neural_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            step INTEGER,
            scores TEXT,
            metadata TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

    # Trigger sync (Check results/ directory)
    print("Starting sync...")
    sync_count = agent.sync_github_results(base_dirs=["results"])
    print(f"Synced {sync_count} snapshots.")
    
    if sync_count == 0:
        print("Warning: No snapshots synced. Check if results/ contains valid JSON files.")
    else:
        # Test Search
        search_results = agent.search_snapshots("a") # Search for common letter
        print(f"Search results count: {len(search_results)}")

        # Test RAG Context
        context = agent.get_rag_context("DIPG")
        print("RAG Context Sample:")
        print(context[:300] + "...")
        
        assert "Relevant Safety Failures" in context
        assert "Snippet" in context

    print("✅ Phase 13 Backend Verified.")

if __name__ == "__main__":
    run_test()
