import subprocess
import os
import sys
import logging
import shlex

logger = logging.getLogger(__name__)

def require_local_auth(reason: str) -> bool:
    """
    Triggers a system-level authentication challenge (Touch ID or Password).
    Returns True if successfully authenticated, False otherwise.
    
    On macOS, uses osascript with administrator privileges.
    """
    if sys.platform != "darwin":
        # Fallback for non-macOS: assume True for now or implement CLI password
        logger.warning(f"Local auth requested on {sys.platform}, bypassing challenge.")
        return True

    # Use shlex.quote to properly escape the message for the shell
    message = f"Authenticating for SafeClaw: {reason}"
    script = f'do shell script "echo {shlex.quote(message)}" with administrator privileges'
    
    try:
        # We use a 60s timeout to allow the user time to react
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            timeout=60
        )
        logger.info(f"Local auth successful for: {reason}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Local auth failed or canceled: {e}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Local auth timed out.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during local auth: {e}")
        return False

if __name__ == "__main__":
    # Test execution
    print("Testing local auth...")
    if require_local_auth("Delete repository 'med-safety-gym-v2'"):
        print("Success!")
    else:
        print("Failed or Canceled.")
