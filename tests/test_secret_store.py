import pytest
from med_safety_gym.identity.secret_store import InMemorySecretStore

def test_in_memory_secret_store():
    store = InMemorySecretStore()
    
    # Test setting and getting
    store.set_secret("test_key", "test_value")
    assert store.get_secret("test_key") == "test_value"
    
    # Test missing key
    assert store.get_secret("nonexistent") is None
    
    # Test clearing
    store.clear_secrets()
    assert store.get_secret("test_key") is None

def test_in_memory_multiple_keys():
    store = InMemorySecretStore()
    store.set_secret("k1", "v1")
    store.set_secret("k2", "v2")
    
    assert store.get_secret("k1") == "v1"
    assert store.get_secret("k2") == "v2"
