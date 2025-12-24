import openenv_core
import pkgutil

print(f"openenv_core path: {openenv_core.__path__}")
print("Submodules:")
for loader, module_name, is_pkg in pkgutil.walk_packages(openenv_core.__path__):
    print(f"- {module_name} (is_pkg: {is_pkg})")

try:
    from openenv_core import http_env_client
    print("SUCCESS: imported openenv_core.http_env_client")
except ImportError as e:
    print(f"FAILURE: could not import openenv_core.http_env_client: {e}")

try:
    from openenv_core.http_env_client import StepResult
    print("SUCCESS: imported StepResult from openenv_core.http_env_client")
except ImportError as e:
    print(f"FAILURE: could not import StepResult: {e}")
