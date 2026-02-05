
import pytest
from med_safety_eval.observer import DatabaseSink
from med_safety_eval.schemas import NeuralSnapshot
from sqlalchemy import text

@pytest.fixture
def sqlite_sink():
    # Use in-memory SQLite for testing
    sink = DatabaseSink("sqlite:///:memory:", table_name="test_snapshots")
    return sink

def test_database_sink_emit(sqlite_sink):
    """Verify DatabaseSink writes to the database."""
    snapshot = NeuralSnapshot(
        session_id="test_db_session",
        step=1,
        scores={"root": 1.0, "grounding": 0.5},
        metadata={"action": "test_action"}
    )
    
    sqlite_sink.emit(snapshot)
    
    # Verify data
    with sqlite_sink.engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM test_snapshots")).mappings().all()
        assert len(result) == 1
        row = result[0]
        assert row["session_id"] == "test_db_session"
        assert row["step"] == 1
        # SQLite returns JSON as string, so we need to parse it
        import json
        scores = json.loads(row["scores"])
        assert scores["root"] == 1.0
