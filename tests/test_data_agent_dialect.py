
import pytest
from unittest.mock import MagicMock
from med_safety_eval.data_agent import DataAgent
import json

def test_pop_command_dialect_check():
    """Verifies that pop_command raises NotImplementedError for non-PostgreSQL dialects."""
    agent = DataAgent()
    agent.engine = MagicMock()
    # Simulate a SQLite dialect
    agent.engine.dialect.name = "sqlite"
    
    with pytest.raises(NotImplementedError) as excinfo:
        agent.pop_command("test_session")
    
    assert "Atomic pop_command is only supported for PostgreSQL backends" in str(excinfo.value)

def test_pop_command_postgres_call():
    """Verifies that pop_command proceeds with PostgreSQL dialect."""
    agent = DataAgent()
    agent.engine = MagicMock()
    # Simulate Postgres dialect
    agent.engine.dialect.name = "postgresql"
    
    # Mock connection and execution
    mock_conn = MagicMock()
    agent.engine.begin.return_value.__enter__.return_value = mock_conn
    
    # Mock result
    mock_result = MagicMock()
    mock_result.fetchone.return_value = ['{"action": "TEST"}']
    mock_conn.execute.return_value = mock_result
    
    command = agent.pop_command("test_session")
    
    assert command == {"action": "TEST"}
    # Verify the correct SQL was executed
    args, _ = mock_conn.execute.call_args
    assert str(args[0]) == "DELETE FROM gauntlet_commands WHERE session_id = :sid RETURNING command"
    assert args[1] == {"sid": "test_session"}
