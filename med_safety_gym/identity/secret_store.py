from abc import ABC, abstractmethod
from typing import Optional
import keyring

class SecretStore(ABC):
    """Abstract interface for storing agent secrets."""
    
    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def set_secret(self, key: str, value: str) -> None:
        pass

    @abstractmethod
    def clear_secrets(self) -> None:
        pass

class KeyringSecretStore(SecretStore):
    """Production implementation using OS keyring (e.g. macOS Keychain)."""
    
    SERVICE_NAME = "safeclaw"

    def get_secret(self, key: str) -> Optional[str]:
        return keyring.get_password(self.SERVICE_NAME, key)

    def set_secret(self, key: str, value: str) -> None:
        keyring.set_password(self.SERVICE_NAME, key, value)

    def clear_secrets(self) -> None:
        # Note: keyring doesn't have a built-in 'clear all for service' 
        # so we rely on known keys for now or manual cleanup.
        # In a real impl, we might track keys in a nested structure.
        for key in ["auth_token", "hub_pub_key"]:
            try:
                keyring.delete_password(self.SERVICE_NAME, key)
            except keyring.errors.PasswordDeleteError:
                pass

class InMemorySecretStore(SecretStore):
    """Ephemeral storage for testing/CI."""
    
    def __init__(self):
        self._secrets = {}

    def get_secret(self, key: str) -> Optional[str]:
        return self._secrets.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._secrets[key] = value

    def clear_secrets(self) -> None:
        self._secrets.clear()
