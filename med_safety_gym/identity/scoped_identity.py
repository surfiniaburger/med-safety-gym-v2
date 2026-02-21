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

def issue_delegation_token(claims: Dict[str, Any], ttl_seconds: int, secret_key: str) -> str:
    """
    Issues a signed delegation token (JWT) valid for `ttl_seconds`.
    """
    payload = claims.copy()
    payload["iat"] = int(time.time())
    payload["exp"] = payload["iat"] + ttl_seconds
    
    return jwt.encode(payload, secret_key, algorithm="HS256")

def verify_delegation_token(token: str, secret_key: str) -> Dict[str, Any]:
    """
    Verifies a delegation token and returns its decoded claims.
    Raises jwt.ExpiredSignatureError or jwt.InvalidTokenError if invalid.
    """
    return jwt.decode(token, secret_key, algorithms=["HS256"])
