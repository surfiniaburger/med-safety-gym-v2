from typing import List, Dict, Any
import jwt
import time

def create_scoped_manifest(global_manifest: List[Dict[str, Any]], allowed_tools: List[str]) -> List[Dict[str, Any]]:
    """
    Filters a global manifest to only include tools present in the allowed_tools list.
    """
    return [tool for tool in global_manifest if tool.get("name") in allowed_tools]

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
