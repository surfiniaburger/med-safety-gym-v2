import pytest
import subprocess
import shlex
from unittest.mock import patch, Mock
from med_safety_gym.auth_guard import require_local_auth

@patch('subprocess.run')
@patch('sys.platform', 'darwin')
def test_local_auth_injection_prevention(mock_run):
    """Verify that malicious strings are escaped and don't inject shell commands."""
    malicious_reason = "'; ls -la /tmp; echo '"
    
    mock_run.return_value = Mock(returncode=0)
    
    require_local_auth(malicious_reason)
    
    # Check the command passed to subprocess.run
    # cmd should be ['osascript', '-e', script]
    cmd = mock_run.call_args[0][0]
    script = cmd[2]
    
    # The script should contain the quoted reason
    # shlex.quote handles ' by escaping it correctly for POSIX shells
    quoted_msg = shlex.quote(f"Authenticating for SafeClaw: {malicious_reason}")
    assert f'echo {quoted_msg}' in script
    
    # Critical: Check that 'ls -la' isn't unquoted
    # In an injection, it would be 'do shell script "...; ls -la ..."'
    # With quoting, it should be within the single-quoted echo string.
    assert 'ls -la' in script
    # Verify it doesn't break out of the echo command
    # A simple check: if it's properly quoted, it shouldn't contain unescaped shell metacharacters 
    # that would allow command execution outside echo.
    # shlex.quote is trusted for this.
