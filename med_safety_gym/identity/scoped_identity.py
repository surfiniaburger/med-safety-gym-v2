from typing import List, Dict, Any
import jwt
import time

def create_scoped_manifest(global_manifest: Dict[str, List[Any]], allowed_tools: List[str]) -> Dict[str, List[Any]]:
    """
    Filters a tiered manifest dictionary to only include tools present in the allowed_tools list.
    Preserves the tier keys (user, write, admin, critical).
    """
    scoped = {}
    for tier, tools in global_manifest.items():
        if isinstance(tools, list):
            scoped[tier] = [t for t in tools if (t.get("name") if isinstance(t, dict) else t) in allowed_tools]
        else:
            scoped[tier] = tools
    return scoped

def issue_delegation_token(claims: Dict[str, Any], ttl_seconds: int, signing_key: str) -> str:
    """
    Issues a signed delegation token (JWT). 
    Supports both HS256 (symmetric) and EdDSA (asymmetric Ed25519).
    """
    payload = claims.copy()
    payload["iat"] = int(time.time())
    payload["exp"] = payload["iat"] + ttl_seconds
    
    algorithm = "HS256"
    if "-----BEGIN" in signing_key:
        algorithm = "EdDSA"
        
    return jwt.encode(payload, signing_key, algorithm=algorithm)

def verify_delegation_token(token: str, verification_key: str) -> Dict[str, Any]:
    """
    Verifies a delegation token and returns its decoded claims.
    """
    algorithms = ["HS256", "EdDSA"]
    return jwt.decode(token, verification_key, algorithms=algorithms)
