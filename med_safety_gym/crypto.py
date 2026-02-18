"""
SafeClaw Crypto Layer — ED25519 Signatures

Provides small, focused functions for manifest signing and verification.
Follows Farley practices: functions < 10 lines, clear communication.
"""
import logging
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

def generate_keys():
    """Generate a new ED25519 private/public key pair."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()

def sign_data(data: bytes, private_key: ed25519.Ed25519PrivateKey) -> bytes:
    """Sign raw bytes with an ED25519 private key."""
    return private_key.sign(data)

def verify_signature(data: bytes, signature: bytes, public_key: ed25519.Ed25519PublicKey) -> bool:
    """Verify an ED25519 signature. Returns True if valid, False otherwise."""
    try:
        public_key.verify(signature, data)
        return True
    except InvalidSignature:
        return False
    except Exception as e:
        logger.error(f"Crypto verification error: {e}")
        return False
