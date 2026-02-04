import pytest
from pydantic import ValidationError

# These imports will fail initially (TDD Red)
try:
    from med_safety_eval.schemas import NeuralSnapshot, StepResult, RubricScore
except ImportError:
    NeuralSnapshot = None
    StepResult = None
    RubricScore = None

def test_schemas_exist():
    """TDD: Verify schemas module exists."""
    assert NeuralSnapshot is not None, "med_safety_eval.schemas module not found"

@pytest.mark.skipif(NeuralSnapshot is None, reason="Schemas not implemented")
def test_neural_snapshot_validation_success():
    """Verify strictly valid data passes."""
    snapshot = NeuralSnapshot(
        session_id="test_session_1",
        step=10,
        scores={"safety": 0.95, "helpfulness": 0.8},
        metadata={"model": "gemma-2b"}
    )
    assert snapshot.session_id == "test_session_1"
    assert snapshot.scores["safety"] == 0.95

@pytest.mark.skipif(NeuralSnapshot is None, reason="Schemas not implemented")
def test_neural_snapshot_validation_failure():
    """Verify malformed data is rejected (Zero Trust)."""
    with pytest.raises(ValidationError):
        NeuralSnapshot(
            session_id=123,  # Wrong type, should be str
            step="ten",      # Wrong type, should be int
            scores="high"    # Wrong type, should be dict
        )

@pytest.mark.skipif(NeuralSnapshot is None, reason="Schemas not implemented")
def test_sanitization_xss():
    """Verify HTML tags are stripped from metadata strings (XSS Prevention)."""
    snapshot = NeuralSnapshot(
        session_id="test_xss",
        step=1,
        scores={"safety": 0.0},
        metadata={"comment": "<script>alert('pwned')</script>Safe Text"}
    )
    # The exact behavior depends on implementation (strip vs escape).
    # We'll assert it's NOT the raw script.
    assert "<script>" not in snapshot.metadata["comment"], "HTML tags were not sanitized!"
