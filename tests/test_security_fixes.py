
import pytest
from med_safety_gym.app import get_environment
from med_safety_gym.dipg_environment import DIPGEnvironment

def test_environment_isolation():
    """Verify that get_environment creates distinct instances."""
    env1 = get_environment()
    env2 = get_environment()
    
    assert env1 is not env2
    assert isinstance(env1, DIPGEnvironment)
    assert isinstance(env2, DIPGEnvironment)
    
    # Verify they share the same dataset (optimization check)
    assert env1.dataset is env2.dataset
