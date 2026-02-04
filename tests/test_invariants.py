import pytest
from fastapi.testclient import TestClient
from med_safety_gym.app import app

client = TestClient(app)

def test_no_custom_websocket_routes():
    """
    INVARIANT CHECK: The Environment Server must NOT expose custom WebSocket routes
    for observability or control. These belong in the Infrastructure layer (Hub).
    """
    routes = [route.path for route in app.routes]
    
    # Check for the specific forbidden routes we identified
    forbidden_routes = [
        "/ws/gauntlet/{session_id}",
        "/gauntlet/stream/{session_id}"
    ]
    
    violations = [r for r in routes if any(f in r for f in forbidden_routes)]
    
    assert not violations, \
        f"INVARIANT VIOLATION: Env Server has custom routes: {violations}. " \
        "These must be moved to the Observability Hub."

def test_no_broadcasting_dependencies():
    """
    INVARIANT CHECK: app.py should not import or use 'ConnectionManager' 
    or have broadcasting logic.
    """
    import inspect
    import med_safety_gym.app as app_module
    
    # Check if 'manager' or 'ConnectionManager' is present in the module's global scope
    has_manager = hasattr(app_module, "manager")
    has_class = hasattr(app_module, "ConnectionManager")
    
    assert not has_manager, "INVARIANT VIOLATION: 'manager' instance found in app.py"
    assert not has_class, "INVARIANT VIOLATION: 'ConnectionManager' class found in app.py"
